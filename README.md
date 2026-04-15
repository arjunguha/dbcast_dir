# dbcast_dir

Broadcast an entire directory tree from a shared filesystem to node-local
storage on every node, using MPI. Based on
[dbcast](https://github.com/hpc/mpifileutils) from mpifileutils, extended to
handle recursive directories with a progress indicator.

## Building

Requires an MPI compiler and [mpifileutils](https://github.com/hpc/mpifileutils).

```bash
module load mpifileutils
mkdir build && cd build && cmake .. && make
```

## Usage

```
srun -N <nodes> --ntasks-per-node=<n> dbcast_dir [options] <SRC_DIR> <DST_DIR>
```

Requires at least 2 tasks per node. All nodes must have the same task count.

### Options

| Flag | Description |
|------|-------------|
| `-s, --size <SIZE>` | Block/stripe size (default 1MB) |
| `--bar` | Show a single-line progress bar |
| `--silent` | Suppress all progress output |
| `-h, --help` | Print usage |

### Example

```bash
srun -N 4 --ntasks-per-node=8 dbcast_dir --bar /lustre/project/my_env /mnt/bb/$USER/my_env
```

## Benchmark

148 GiB model (52 files) broadcast from Lustre to node-local NVMe on two
Frontier nodes (8 tasks per node):

```
$ srun -N2 --ntasks-per-node=8 dbcast_dir --bar qwen3_coder_next /mnt/bb/arjunguha/y
Broadcasting 52 files, 1 dirs, 0 symlinks (148.4 GiB) from /lustre/orion/lrn089/scratch/arjunguha/models/qwen3_coder_next to /mnt/bb/arjunguha/y
100% |################################| 148.4 GiB / 148.4 GiB  1.0 GiB/s  elapsed 2m 26s
Broadcast complete: 52 files, 146.48 s, 1.0 GiB/s
```
