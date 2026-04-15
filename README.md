# dbcast_dir

Broadcast an entire directory tree from a shared filesystem to node-local
storage on every node, using MPI. Based on the
[dbcast](https://github.com/hpc/mpifileutils) single-file broadcast tool from
mpifileutils, extended to handle recursive directories with a progress
indicator.

## Usage

```
srun -N <nodes> --ntasks-per-node=<n> dbcast_dir [options] <SRC_DIR> <DST_DIR>
```

Requires at least 2 tasks per node. All nodes must have the same task count.

### Options

| Flag | Description |
|------|-------------|
| `-s, --size <SIZE>` | Block/stripe size (default 1MB) |
| `-h, --help` | Print usage |

### Example

```bash
salloc -N 4 --ntasks-per-node=8 -C nvme -t 10
srun dbcast_dir /lustre/project/my_env /mnt/bb/$USER/my_env
```

## Building

Requires an MPI compiler and [mpifileutils](https://github.com/hpc/mpifileutils).

```bash
module load mpifileutils   # or however your site provides it
mkdir build && cd build
cmake ..
make
```

CMake finds `mfu.h` and `libmfu` automatically via `CMAKE_PREFIX_PATH`, which
most module systems set. To point to a custom install:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/mpifileutils ..
```

## How it works

1. Rank 0 walks the source directory and broadcasts the file list to all ranks.
2. Writer ranks (one per node) create the directory skeleton on local storage.
3. Each file is broadcast using the ring-based shared-memory algorithm from
   dbcast: reader ranks read from the shared filesystem and propagate data
   through an MPI ring, while the writer rank on each node writes to local
   disk via POSIX shared memory.
4. Symlinks are recreated on each node.

Progress is reported on stderr: a single-line updating bar in interactive
terminals, one line per percent otherwise.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

This project is based on [mpifileutils](https://github.com/hpc/mpifileutils),
which is distributed under a BSD license by Lawrence Livermore National
Laboratory.
