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
