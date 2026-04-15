/*
 * dbcast_dir.c - Broadcast a directory tree to node-local storage using MPI
 *
 * Based on dbcast from mpifileutils by Adam Moody (LLNL).
 * Extended to handle entire directories with progress reporting.
 *
 * Usage: srun -N <nodes> --ntasks-per-node=<n> dbcast_dir [options] SRC_DIR DST_DIR
 *        Requires at least 2 tasks per node. All nodes must have the same task count.
 */

#define _GNU_SOURCE

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <getopt.h>
#include <sys/mman.h>
#include <ftw.h>

#include "mfu.h"

/* ================================================================== */
/* Shared memory allocation (simplified from dbcast.c GCS_Shmem)      */
/* ================================================================== */

static int anytrue(int value, MPI_Comm comm)
{
    int result;
    MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_LOR, comm);
    return result;
}

/*
 * Allocate a shared memory segment visible to all ranks in comm.
 * All ranks in comm must be on the same physical node.
 * buf_id must be unique per allocation for the same comm.
 * root_world_rank is the MPI_COMM_WORLD rank of rank 0 in comm
 * (used for unique filename generation).
 */
static void* shmem_alloc(size_t size, MPI_Comm comm,
                          int buf_id, int root_world_rank)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    char file[256];
    const char* jobid = getenv("SLURM_JOBID");
    snprintf(file, sizeof(file), "/dev/shm/dbcast_dir-%s-%d-%d.shmem",
             jobid ? jobid : "nojob", root_world_rank, buf_id);

    MPI_Barrier(comm);

    int fd = open(file, O_RDWR | O_CREAT, S_IRWXU);
    if (anytrue((fd < 0), comm)) {
        if (fd >= 0) close(fd);
        if (rank == 0) unlink(file);
        return NULL;
    }

    if (rank == 0) {
        ftruncate(fd, 0);
        ftruncate(fd, (off_t)size);
    }
    MPI_Barrier(comm);

    void* ptr = mmap(0, (off_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (anytrue((ptr == MAP_FAILED), comm)) {
        if (ptr != MAP_FAILED) munmap(ptr, (off_t)size);
        close(fd);
        if (rank == 0) unlink(file);
        return NULL;
    }

    close(fd);
    if (rank == 0) unlink(file);

    return ptr;
}

static void shmem_free(void* ptr, size_t size, MPI_Comm comm)
{
    MPI_Barrier(comm);
    if (ptr != NULL && ptr != MAP_FAILED) {
        munmap(ptr, size);
    }
}

/* ================================================================== */
/* Utility helpers                                                    */
/* ================================================================== */

static void bcast_abort(const char* msg)
{
    if (msg) fprintf(stderr, "FATAL: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static void compute_offset_size(
    uint64_t bytes_read, uint64_t stripe_read,
    uint64_t file_size, uint64_t stripe_size,
    size_t chunk_size, int rdr_rank, uint64_t chunk_id,
    off_t* outpos, size_t* outsize)
{
    size_t read_size = chunk_size;

    uint64_t remainder = stripe_size - stripe_read;
    if (remainder < (uint64_t)read_size) {
        read_size = (size_t)remainder;
    }

    uint64_t offset = bytes_read
                    + (uint64_t)rdr_rank * stripe_size
                    + chunk_id * (uint64_t)chunk_size;
    if (offset < file_size) {
        uint64_t last = offset + (uint64_t)read_size;
        if (last > file_size) {
            read_size = (size_t)(file_size - offset);
        }
    } else {
        read_size = 0;
    }

    *outpos  = (off_t)offset;
    *outsize = read_size;
}

static int mkdirp_r(const char* path, int depth)
{
    if (depth > 128) return -1; /* guard against infinite recursion */

    int rc = 0;
    mfu_path* parent = mfu_path_from_str(path);
    mfu_path_dirname(parent);
    const char* parent_str = mfu_path_strdup(parent);

    /* Stop recursing when parent equals path (filesystem root) */
    if (strcmp(parent_str, path) != 0 && mfu_access(parent_str, R_OK) < 0) {
        rc = mkdirp_r(parent_str, depth + 1);
    }

    errno = 0;
    int tmp_rc = mfu_mkdir(path, S_IRWXU);
    if (tmp_rc < 0 && errno != EEXIST) {
        rc = tmp_rc;
    }

    mfu_free(&parent_str);
    mfu_path_delete(&parent);
    return rc;
}

static int mkdirp(const char* path)
{
    return mkdirp_r(path, 0);
}

/* ================================================================== */
/* File list for directory walk                                       */
/* ================================================================== */

typedef struct {
    char*    relpath;
    mode_t   mode;
    uint64_t size;
    int      type;         /* 0=dir, 1=file, 2=symlink */
    char*    link_target;  /* only for type==2 */
} file_entry_t;

typedef struct {
    file_entry_t* entries;
    int count;
    int capacity;
    uint64_t total_file_bytes;
} file_list_t;

/* Globals for nftw callback (nftw has no user-data pointer) */
static file_list_t* g_walk_list    = NULL;
static size_t       g_walk_root_len = 0;

static void file_list_init(file_list_t* list)
{
    list->entries = NULL;
    list->count = 0;
    list->capacity = 0;
    list->total_file_bytes = 0;
}

static void file_list_add(file_list_t* list, const file_entry_t* entry)
{
    if (list->count >= list->capacity) {
        int new_cap = list->capacity == 0 ? 256 : list->capacity * 2;
        file_entry_t* tmp = realloc(list->entries, new_cap * sizeof(file_entry_t));
        if (!tmp) bcast_abort("realloc failed in file_list_add");
        list->entries = tmp;
        list->capacity = new_cap;
    }
    list->entries[list->count++] = *entry;
}

static void file_list_free(file_list_t* list)
{
    for (int i = 0; i < list->count; i++) {
        free(list->entries[i].relpath);
        free(list->entries[i].link_target);
    }
    free(list->entries);
    memset(list, 0, sizeof(*list));
}

static int nftw_callback(const char* fpath, const struct stat* sb,
                          int typeflag, struct FTW* ftwbuf)
{
    (void)ftwbuf;

    /* Compute relative path */
    const char* rel = fpath + g_walk_root_len;
    if (*rel == '/') rel++;

    file_entry_t entry;
    memset(&entry, 0, sizeof(entry));
    entry.relpath = (*rel == '\0') ? strdup(".") : strdup(rel);
    if (!entry.relpath) return -1; /* OOM */
    entry.mode = sb->st_mode;

    switch (typeflag) {
    case FTW_D:
        entry.type = 0;
        entry.size = 0;
        break;

    case FTW_F:
        if (!S_ISREG(sb->st_mode)) {
            free(entry.relpath);
            return 0; /* skip non-regular files (sockets, fifos, etc.) */
        }
        entry.type = 1;
        entry.size = (uint64_t)sb->st_size;
        g_walk_list->total_file_bytes += entry.size;
        break;

    case FTW_SL:
#ifdef FTW_SLN
    case FTW_SLN:
#endif
    {
        entry.type = 2;
        entry.size = 0;
        char target[PATH_MAX];
        ssize_t len = readlink(fpath, target, sizeof(target) - 1);
        if (len > 0) {
            target[len] = '\0';
            entry.link_target = strdup(target);
        } else {
            entry.link_target = strdup("");
        }
        if (!entry.link_target) { free(entry.relpath); return -1; }
        break;
    }

    default:
        free(entry.relpath);
        return 0;
    }

    file_list_add(g_walk_list, &entry);
    return 0;
}

static int walk_directory(const char* src_root, file_list_t* list)
{
    file_list_init(list);
    g_walk_list = list;
    g_walk_root_len = strlen(src_root);
    /* Strip trailing slashes */
    while (g_walk_root_len > 1 && src_root[g_walk_root_len - 1] == '/') {
        g_walk_root_len--;
    }
    int rc = nftw(src_root, nftw_callback, 64, FTW_PHYS);
    g_walk_list = NULL;
    return rc;
}

/* ------------------------------------------------------------------ */
/* File list serialization                                            */
/* ------------------------------------------------------------------ */

/*
 * Wire format:
 *   uint32_t count
 *   uint64_t total_file_bytes
 *   For each entry:
 *     uint32_t type
 *     uint32_t mode
 *     uint64_t size
 *     uint32_t relpath_len  (includes NUL)
 *     char[relpath_len]     relpath
 *     uint32_t link_len     (0 if not symlink, else includes NUL)
 *     char[link_len]        link_target
 */
static char* serialize_file_list(const file_list_t* list, size_t* out_size)
{
    size_t total = sizeof(uint32_t) + sizeof(uint64_t);
    for (int i = 0; i < list->count; i++) {
        const file_entry_t* e = &list->entries[i];
        total += 3 * sizeof(uint32_t) + sizeof(uint64_t);
        total += strlen(e->relpath) + 1;
        total += sizeof(uint32_t);
        if (e->link_target) total += strlen(e->link_target) + 1;
    }

    char* buf = malloc(total);
    if (!buf) return NULL;
    char* p = buf;

    uint32_t count = (uint32_t)list->count;
    memcpy(p, &count, sizeof(count));                          p += sizeof(count);
    memcpy(p, &list->total_file_bytes, sizeof(uint64_t));      p += sizeof(uint64_t);

    for (int i = 0; i < list->count; i++) {
        const file_entry_t* e = &list->entries[i];
        uint32_t type = (uint32_t)e->type;
        uint32_t mode = (uint32_t)e->mode;
        uint64_t size = e->size;
        uint32_t rlen = (uint32_t)(strlen(e->relpath) + 1);
        uint32_t llen = e->link_target ? (uint32_t)(strlen(e->link_target) + 1) : 0;

        memcpy(p, &type, sizeof(type)); p += sizeof(type);
        memcpy(p, &mode, sizeof(mode)); p += sizeof(mode);
        memcpy(p, &size, sizeof(size)); p += sizeof(size);
        memcpy(p, &rlen, sizeof(rlen)); p += sizeof(rlen);
        memcpy(p, e->relpath, rlen);    p += rlen;
        memcpy(p, &llen, sizeof(llen)); p += sizeof(llen);
        if (llen > 0) {
            memcpy(p, e->link_target, llen); p += llen;
        }
    }

    *out_size = (size_t)(p - buf);
    return buf;
}

static void deserialize_file_list(const char* buf, file_list_t* list)
{
    file_list_init(list);
    const char* p = buf;

    uint32_t count;
    memcpy(&count, p, sizeof(count));                     p += sizeof(count);
    memcpy(&list->total_file_bytes, p, sizeof(uint64_t)); p += sizeof(uint64_t);

    for (uint32_t i = 0; i < count; i++) {
        file_entry_t entry;
        memset(&entry, 0, sizeof(entry));

        uint32_t type, mode, rlen, llen;
        uint64_t size;

        memcpy(&type, p, sizeof(type)); p += sizeof(type);
        memcpy(&mode, p, sizeof(mode)); p += sizeof(mode);
        memcpy(&size, p, sizeof(size)); p += sizeof(size);
        memcpy(&rlen, p, sizeof(rlen)); p += sizeof(rlen);
        entry.relpath = strdup(p);
        if (!entry.relpath) bcast_abort("strdup failed in deserialize");
        p += rlen;
        memcpy(&llen, p, sizeof(llen)); p += sizeof(llen);
        if (llen > 0) {
            entry.link_target = strdup(p);
            if (!entry.link_target) bcast_abort("strdup failed in deserialize");
            p += llen;
        }

        entry.type = (int)type;
        entry.mode = (mode_t)mode;
        entry.size = size;
        file_list_add(list, &entry);
    }
}

static void bcast_file_list(file_list_t* list, int rank, MPI_Comm comm)
{
    size_t buf_size = 0;
    char* buf = NULL;

    if (rank == 0) {
        buf = serialize_file_list(list, &buf_size);
        if (!buf) bcast_abort("serialize_file_list: malloc failed");
    }

    uint64_t bsize = (uint64_t)buf_size;
    MPI_Bcast(&bsize, 1, MPI_UINT64_T, 0, comm);
    buf_size = (size_t)bsize;

    if (rank != 0) {
        buf = malloc(buf_size);
        if (!buf) bcast_abort("bcast_file_list: malloc failed");
    }

    /* Broadcast in chunks if needed (MPI count is int) */
    size_t remaining = buf_size;
    char* ptr = buf;
    while (remaining > 0) {
        int chunk = (remaining > (size_t)INT_MAX) ? INT_MAX : (int)remaining;
        MPI_Bcast(ptr, chunk, MPI_BYTE, 0, comm);
        ptr += chunk;
        remaining -= chunk;
    }

    if (rank != 0) {
        deserialize_file_list(buf, list);
    }
    free(buf);
}

/* ================================================================== */
/* Progress bar                                                       */
/* ================================================================== */

typedef struct {
    double   time_start;
    uint64_t total_bytes;
    uint64_t bytes_done;
    int      is_interactive;
    int      last_percent;
    char     current_file[256];
} progress_t;

static void progress_init(progress_t* p, uint64_t total_bytes, double t0)
{
    p->time_start = t0;
    p->total_bytes = total_bytes;
    p->bytes_done = 0;
    p->is_interactive = isatty(STDERR_FILENO);
    p->last_percent = -1;
    p->current_file[0] = '\0';
}

static void progress_set_file(progress_t* p, const char* relpath)
{
    const char* base = strrchr(relpath, '/');
    base = base ? base + 1 : relpath;
    snprintf(p->current_file, sizeof(p->current_file), "%s", base);
}

static void format_eta(double secs, char* buf, size_t sz)
{
    if (secs < 0 || secs > 86400 * 365) {
        snprintf(buf, sz, "--:--");
        return;
    }
    int s = (int)(secs + 0.5);
    if (s < 60)       snprintf(buf, sz, "%ds", s);
    else if (s < 3600) snprintf(buf, sz, "%dm %02ds", s / 60, s % 60);
    else               snprintf(buf, sz, "%dh %02dm", s / 3600, (s % 3600) / 60);
}

static void progress_update(progress_t* p)
{
    if (p->total_bytes == 0) return;

    int percent = (int)((double)p->bytes_done / (double)p->total_bytes * 100.0);
    if (percent > 100) percent = 100;

    double elapsed = MPI_Wtime() - p->time_start;
    double rate = (elapsed > 0.01) ? (double)p->bytes_done / elapsed : 0.0;
    double remaining = (rate > 0) ? (double)(p->total_bytes - p->bytes_done) / rate : 0.0;

    double rate_val;
    const char* rate_units;
    mfu_format_bw(rate, &rate_val, &rate_units);

    char eta[32];
    format_eta(remaining, eta, sizeof(eta));

    double done_val, total_val;
    const char *done_u, *total_u;
    mfu_format_bytes(p->bytes_done,  &done_val,  &done_u);
    mfu_format_bytes(p->total_bytes, &total_val, &total_u);

    if (p->is_interactive) {
        int bar_w = 32;
        int filled = (int)((double)percent / 100.0 * bar_w);
        char bar[33];
        for (int i = 0; i < bar_w; i++) bar[i] = (i < filled) ? '#' : ' ';
        bar[bar_w] = '\0';

        fprintf(stderr,
            "\r%3d%% |%s| %.1f %s / %.1f %s  %.1f %s  eta %s  %s   ",
            percent, bar,
            done_val, done_u, total_val, total_u,
            rate_val, rate_units, eta, p->current_file);
        fflush(stderr);
    } else {
        if (percent > p->last_percent) {
            fprintf(stderr, "%3d%%  %.1f %s  eta %s\n",
                    percent, rate_val, rate_units, eta);
            fflush(stderr);
            p->last_percent = percent;
        }
    }
}

static void progress_finish(progress_t* p)
{
    double elapsed = MPI_Wtime() - p->time_start;
    double rate = (elapsed > 0.01) ? (double)p->total_bytes / elapsed : 0.0;
    double rate_val;
    const char* rate_units;
    mfu_format_bw(rate, &rate_val, &rate_units);

    double total_val;
    const char* total_u;
    mfu_format_bytes(p->total_bytes, &total_val, &total_u);

    char el[32];
    format_eta(elapsed, el, sizeof(el));

    if (p->is_interactive) {
        char bar[33];
        memset(bar, '#', 32);
        bar[32] = '\0';
        fprintf(stderr,
            "\r100%% |%s| %.1f %s / %.1f %s  %.1f %s  elapsed %s\n",
            bar, total_val, total_u, total_val, total_u,
            rate_val, rate_units, el);
    } else {
        if (p->last_percent < 100) {
            fprintf(stderr, "100%%  %.1f %s  elapsed %s\n",
                    rate_val, rate_units, el);
        }
    }
    fflush(stderr);
}

/* ================================================================== */
/* Per-file broadcast (adapted from dbcast.c reader/writer loops)     */
/* ================================================================== */

/*
 * Broadcast a single file from src_path (on shared FS) to dst_path
 * (on node-local storage) using the ring-based shared-memory algorithm.
 *
 * All MPI ranks in COMM_WORLD must call this collectively.
 *
 * Returns 0 on success, 1 if any writer had an error.
 */
static int broadcast_one_file(
    const char* src_path, const char* dst_path,
    uint64_t file_size, mode_t file_mode,
    int global_rank,
    MPI_Comm node_comm, int node_rank, int node_size,
    MPI_Comm level_comm, int level_rank, int level_size,
    void** shmbuf,
    uint64_t stripe_size, size_t chunk_size,
    progress_t* progress, uint64_t base_bytes)
{
    int write_error = 0;
    int in_file = -1;
    int out_file = -1;

    int reader_size = level_size * (node_size - 1);
    int reader_rank = level_rank * (node_size - 1) + (node_rank - 1);

    /* --- Zero-size file: create and return --- */
    if (file_size == 0) {
        if (node_rank == 0) {
            int fd = mfu_open(dst_path, O_CREAT | O_TRUNC | O_WRONLY,
                              S_IRWXU | S_IRWXG | S_IRWXO);
            if (fd >= 0) {
                mfu_close(dst_path, fd);
                mfu_chmod(dst_path, file_mode);
            } else {
                write_error = 1;
            }
        }
        if (global_rank == 0 && progress) {
            progress->bytes_done = base_bytes;
            progress_update(progress);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        int any;
        MPI_Allreduce(&write_error, &any, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        return any;
    }

    /* --- Open files --- */
    if (node_rank == 0) {
        out_file = mfu_open(dst_path, O_CREAT | O_TRUNC | O_WRONLY,
                            S_IRWXU | S_IRWXG | S_IRWXO);
        if (out_file < 0) {
            write_error = 1;
        }
    } else {
        errno = 0;
        in_file = mfu_open(src_path, O_RDONLY);
        if (in_file < 0) {
            fprintf(stderr, "FATAL: cannot open source '%s': %s\n",
                    src_path, strerror(errno));
            bcast_abort(NULL);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Ring topology --- */
    int left  = level_rank - 1;
    if (left < 0) left = level_size - 1;
    int right = level_rank + 1;
    if (right == level_size) right = 0;

    /* --- READERS --- */
    if (node_rank != 0) {
        MPI_Request request[2];
        MPI_Status  status[2];
        size_t bytes_read = 0;

        while (bytes_read < file_size) {
            uint64_t chunk_id = 0;
            uint64_t stripe_read = 0;
            while (stripe_read < stripe_size) {
                int shmid = (node_rank - 1) * 2;
                void* buf1 = shmbuf[shmid + 0];
                void* buf2 = shmbuf[shmid + 1];

                off_t  pos1;
                size_t size1;
                compute_offset_size(bytes_read, stripe_read,
                    file_size, stripe_size, chunk_size,
                    reader_rank, chunk_id, &pos1, &size1);

                /* Read chunk from source into shmem buf1 */
                if (size1 > 0) {
                    errno = 0;
                    if (mfu_lseek(src_path, in_file, pos1, SEEK_SET) == (off_t)-1) {
                        bcast_abort("reader seek failed");
                    }
                    ssize_t ret = mfu_read(src_path, in_file, buf1, size1);
                    if (ret != (ssize_t)size1) {
                        bcast_abort("reader read failed");
                    }
                }

                /* Ring propagation */
                int lev;
                for (lev = 1; lev < level_size; lev++) {
                    int lev_in = level_rank + lev;
                    if (lev_in >= level_size) lev_in -= level_size;
                    int rdr_in = lev_in * (node_size - 1) + (node_rank - 1);

                    off_t  pos2;
                    size_t size2;
                    compute_offset_size(bytes_read, stripe_read,
                        file_size, stripe_size, chunk_size,
                        rdr_in, chunk_id, &pos2, &size2);

                    /* Signal writer that buf1 is ready */
                    MPI_Send(&shmid, 1, MPI_INT, 0, 0, node_comm);

                    /* Exchange with ring neighbors */
                    MPI_Irecv(buf2, (int)size2, MPI_BYTE, right, 0,
                              level_comm, &request[0]);
                    MPI_Isend(buf1, (int)size1, MPI_BYTE, left,  0,
                              level_comm, &request[1]);
                    MPI_Waitall(2, request, status);

                    /* Wait for writer to finish reading buf1 */
                    MPI_Recv(&shmid, 1, MPI_INT, 0, 0, node_comm, &status[0]);

                    /* Swap buffers */
                    size1 = size2;
                    void* tmp = buf1; buf1 = buf2; buf2 = tmp;
                    shmid = (shmid & 0x1) ? (shmid - 1) : (shmid + 1);
                }

                /* Final handshake for the last buffer */
                MPI_Send(&shmid, 1, MPI_INT, 0, 0, node_comm);
                MPI_Recv(&shmid, 1, MPI_INT, 0, 0, node_comm, &status[0]);

                stripe_read += chunk_size;
                chunk_id++;
            }
            bytes_read += stripe_size * (uint64_t)reader_size;
        }
    }

    /* --- WRITERS --- */
    if (node_rank == 0) {
        MPI_Status status[2];
        size_t bytes_read = 0;

        while (bytes_read < file_size) {
            uint64_t chunk_id = 0;
            uint64_t stripe_read = 0;
            while (stripe_read < stripe_size) {
                int lev;
                for (lev = 0; lev < level_size; lev++) {
                    int node;
                    for (node = 1; node < node_size; node++) {
                        int lev_in = level_rank + lev;
                        if (lev_in >= level_size) lev_in -= level_size;
                        int rdr_in = lev_in * (node_size - 1) + (node - 1);

                        off_t  pos;
                        size_t size;
                        compute_offset_size(bytes_read, stripe_read,
                            file_size, stripe_size, chunk_size,
                            rdr_in, chunk_id, &pos, &size);

                        /* Wait for reader to signal buffer ready */
                        int shmid;
                        MPI_Recv(&shmid, 1, MPI_INT, node, 0,
                                 node_comm, &status[0]);
                        void* copybuf = shmbuf[shmid];

                        /* Write data to output file */
                        if (size > 0 && !write_error) {
                            errno = 0;
                            if (mfu_lseek(dst_path, out_file, pos, SEEK_SET)
                                    == (off_t)-1) {
                                write_error = 1;
                            } else {
                                ssize_t ret = mfu_write(dst_path, out_file,
                                                        copybuf, size);
                                if (ret == -1) {
                                    write_error = 1;
                                }
                            }
                        }

                        /* Signal reader we're done with buffer */
                        MPI_Send(&shmid, 1, MPI_INT, node, 0, node_comm);
                    }
                }
                stripe_read += chunk_size;
                chunk_id++;
            }

            bytes_read += stripe_size * (uint64_t)reader_size;

            /* Update progress (only global rank 0) */
            if (global_rank == 0 && progress) {
                uint64_t done = (bytes_read < file_size) ? bytes_read : file_size;
                progress->bytes_done = base_bytes + done;
                progress_update(progress);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Close files, set metadata, clean up on error --- */
    if (node_rank == 0) {
        if (out_file >= 0) {
            mfu_fsync(dst_path, out_file);
            mfu_close(dst_path, out_file);
        }
        errno = 0;
        mfu_truncate(dst_path, (off_t)file_size);
        mfu_chmod(dst_path, file_mode);
        if (write_error) {
            mfu_unlink(dst_path);
        }
    } else {
        if (in_file >= 0) {
            mfu_close(src_path, in_file);
        }
    }

    int any_error = 0;
    MPI_Allreduce(&write_error, &any_error, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return any_error;
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */

static void print_usage(void)
{
    fprintf(stderr,
        "\nUsage: dbcast_dir [options] <SRC_DIR> <DST_DIR>\n\n"
        "Broadcast a directory tree from a shared filesystem to\n"
        "node-local storage on every node.\n\n"
        "Options:\n"
        "  -s, --size <SIZE>  block/stripe size (default 1MB)\n"
        "  -h, --help         print this message\n\n");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    mfu_init();

    /* Suppress verbose mfu logging; we do our own progress reporting */
    mfu_debug_level = MFU_LOG_ERR;

    int rank, ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint64_t stripe_size = 1024 * 1024;

    /* --- Parse options --- */
    static struct option long_opts[] = {
        {"size", 1, 0, 's'},
        {"help", 0, 0, 'h'},
        {0, 0, 0, 0}
    };

    int usage = 0;
    int opt_idx = 0;
    while (1) {
        int c = getopt_long(argc, argv, "s:h", long_opts, &opt_idx);
        if (c == -1) break;
        switch (c) {
        case 's': {
            unsigned long long bval;
            if (mfu_abtoull(optarg, &bval) != MFU_SUCCESS) {
                if (rank == 0) fprintf(stderr, "Error: bad size '%s'\n", optarg);
                usage = 1;
            }
            stripe_size = (uint64_t)bval;
            break;
        }
        case 'h':
            usage = 1;
            break;
        default:
            usage = 1;
            break;
        }
    }

    if (!usage && (argc - optind) < 2) {
        if (rank == 0) fprintf(stderr, "Error: need SRC_DIR and DST_DIR\n");
        usage = 1;
    }
    if (usage) {
        if (rank == 0) print_usage();
        mfu_finalize();
        MPI_Finalize();
        return 0;
    }

    char* src_dir = mfu_path_strdup_abs_reduce_str(argv[optind]);
    char* dst_dir = mfu_path_strdup_abs_reduce_str(argv[optind + 1]);

    /* --- Split communicators --- */

    /* node_comm: all ranks on the same physical node */
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    /* Use node leader's world rank as key so levels are ordered by node */
    int key = rank;
    MPI_Bcast(&key, 1, MPI_INT, 0, node_comm);

    /* level_comm: same-position ranks across nodes */
    MPI_Comm level_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_rank, key, &level_comm);

    int level_rank, level_size;
    MPI_Comm_rank(level_comm, &level_rank);
    MPI_Comm_size(level_comm, &level_size);

    /* Validate: same number of ranks on every node */
    int max_ns;
    MPI_Allreduce(&node_size, &max_ns, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int same = (node_size == max_ns);
    int all_same;
    MPI_Allreduce(&same, &all_same, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_same) {
        if (rank == 0)
            fprintf(stderr, "Error: all nodes must have the same number of tasks\n");
        mfu_free(&src_dir); mfu_free(&dst_dir);
        MPI_Comm_free(&level_comm); MPI_Comm_free(&node_comm);
        mfu_finalize(); MPI_Finalize();
        return 1;
    }
    if (node_size < 2) {
        if (rank == 0)
            fprintf(stderr, "Error: need at least 2 tasks per node\n");
        mfu_free(&src_dir); mfu_free(&dst_dir);
        MPI_Comm_free(&level_comm); MPI_Comm_free(&node_comm);
        mfu_finalize(); MPI_Finalize();
        return 1;
    }

    /* Validate source directory */
    int src_ok = 1;
    if (rank == 0) {
        struct stat st;
        if (stat(src_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Error: cannot read source directory '%s'\n", src_dir);
            src_ok = 0;
        }
    }
    MPI_Bcast(&src_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!src_ok) {
        mfu_free(&src_dir); mfu_free(&dst_dir);
        MPI_Comm_free(&level_comm); MPI_Comm_free(&node_comm);
        mfu_finalize(); MPI_Finalize();
        return 1;
    }

    /* --- Walk directory on rank 0, broadcast to all --- */
    file_list_t flist;
    file_list_init(&flist);
    if (rank == 0) {
        if (walk_directory(src_dir, &flist) != 0) {
            fprintf(stderr, "Error: failed to walk '%s'\n", src_dir);
        }
    }
    bcast_file_list(&flist, rank, MPI_COMM_WORLD);

    /* Count entry types */
    int n_dirs = 0, n_files = 0, n_links = 0;
    for (int i = 0; i < flist.count; i++) {
        switch (flist.entries[i].type) {
        case 0: n_dirs++;  break;
        case 1: n_files++; break;
        case 2: n_links++; break;
        }
    }

    if (rank == 0) {
        double tv; const char* tu;
        mfu_format_bytes(flist.total_file_bytes, &tv, &tu);
        fprintf(stderr,
            "Broadcasting %d files, %d dirs, %d symlinks (%.1f %s) "
            "from %s to %s\n",
            n_files, n_dirs, n_links, tv, tu, src_dir, dst_dir);
    }

    /* --- Allocate shared memory buffers (once, reused for all files) --- */
    size_t chunk_size = 1 * 1024 * 1024;
    size_t alignment  = 1024 * 1024;
    int bufcounts = (node_size - 1) * 2;

    /* Get node leader's world rank for shmem filenames */
    int root_world_rank = rank;
    MPI_Bcast(&root_world_rank, 1, MPI_INT, 0, node_comm);

    void**  shmbuf_base  = malloc(bufcounts * sizeof(void*));
    void**  shmbuf       = malloc(bufcounts * sizeof(void*));
    size_t* shmbuf_sizes = malloc(bufcounts * sizeof(size_t));
    if (!shmbuf_base || !shmbuf || !shmbuf_sizes)
        bcast_abort("malloc failed for shmbuf arrays");

    for (int i = 0; i < bufcounts; i++) {
        size_t alloc_size = chunk_size + alignment;
        shmbuf_base[i] = shmem_alloc(alloc_size, node_comm, i, root_world_rank);
        if (!shmbuf_base[i]) bcast_abort("shmem_alloc failed");
        shmbuf[i]       = shmbuf_base[i];
        shmbuf_sizes[i] = alloc_size;
    }

    /* --- Phase 1: Create directories (writers only) --- */
    double time_start = MPI_Wtime();

    if (node_rank == 0) {
        for (int i = 0; i < flist.count; i++) {
            if (flist.entries[i].type != 0) continue;
            char dpath[PATH_MAX];
            if (strcmp(flist.entries[i].relpath, ".") == 0)
                snprintf(dpath, sizeof(dpath), "%s", dst_dir);
            else
                snprintf(dpath, sizeof(dpath), "%s/%s",
                         dst_dir, flist.entries[i].relpath);
            mkdirp(dpath);
            chmod(dpath, flist.entries[i].mode);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Phase 2: Broadcast regular files --- */
    progress_t progress;
    if (rank == 0) {
        progress_init(&progress, flist.total_file_bytes, time_start);
    }

    uint64_t base_bytes = 0;
    int any_error = 0;

    for (int i = 0; i < flist.count; i++) {
        if (flist.entries[i].type != 1) continue;

        char spath[PATH_MAX], dpath[PATH_MAX];
        snprintf(spath, sizeof(spath), "%s/%s", src_dir, flist.entries[i].relpath);
        snprintf(dpath, sizeof(dpath), "%s/%s", dst_dir, flist.entries[i].relpath);

        if (rank == 0) {
            progress_set_file(&progress, flist.entries[i].relpath);
        }

        int rc = broadcast_one_file(
            spath, dpath,
            flist.entries[i].size, flist.entries[i].mode,
            rank,
            node_comm, node_rank, node_size,
            level_comm, level_rank, level_size,
            shmbuf, stripe_size, chunk_size,
            &progress, base_bytes);

        if (rc) {
            any_error = 1;
            if (rank == 0)
                fprintf(stderr, "\nWarning: errors broadcasting '%s'\n",
                        flist.entries[i].relpath);
        }

        base_bytes += flist.entries[i].size;
    }

    if (rank == 0) {
        progress.bytes_done = progress.total_bytes;
        progress_finish(&progress);
    }

    /* --- Phase 3: Recreate symlinks (writers only) --- */
    if (node_rank == 0) {
        for (int i = 0; i < flist.count; i++) {
            if (flist.entries[i].type != 2) continue;
            if (!flist.entries[i].link_target) continue;
            char dpath[PATH_MAX];
            snprintf(dpath, sizeof(dpath), "%s/%s",
                     dst_dir, flist.entries[i].relpath);
            /* Remove existing entry if any, then create symlink */
            unlink(dpath);
            if (symlink(flist.entries[i].link_target, dpath) != 0) {
                if (rank == 0)
                    fprintf(stderr, "Warning: failed to create symlink '%s'\n",
                            dpath);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Report --- */
    if (rank == 0) {
        double elapsed = MPI_Wtime() - time_start;
        double rate = (elapsed > 0.01)
                    ? (double)flist.total_file_bytes / elapsed : 0.0;
        double rv; const char* ru;
        mfu_format_bw(rate, &rv, &ru);
        fprintf(stderr, "Broadcast complete: %d files, %.2f s, %.1f %s\n",
                n_files, elapsed, rv, ru);
    }

    /* --- Cleanup --- */
    for (int i = 0; i < bufcounts; i++) {
        shmem_free(shmbuf_base[i], shmbuf_sizes[i], node_comm);
    }
    free(shmbuf_base);
    free(shmbuf);
    free(shmbuf_sizes);

    file_list_free(&flist);
    mfu_free(&src_dir);
    mfu_free(&dst_dir);
    MPI_Comm_free(&level_comm);
    MPI_Comm_free(&node_comm);

    mfu_finalize();
    MPI_Finalize();
    return any_error ? 1 : 0;
}
