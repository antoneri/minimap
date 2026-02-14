# minimap

`minimap` is a performance-first, minimal Rust implementation of Infomap for **standard networks** (weighted/unweighted, directed/undirected), focused on:

1. Fast execution
2. Low memory footprint
3. Infomap-compatible output files (`.tree`, `.clu`, `.ftree`)

## Original project

This project is based on ideas and output conventions from **Infomap**.

- Original project: [Infomap](https://github.com/mapequation/infomap)
- Documentation: [mapequation.org/infomap](https://www.mapequation.org/infomap)
- Authors: **M. Rosvall** and **D. Edler**

## Current scope (v1)

- Two-level optimization only
- Standard network input only (link-list and Pajek `*Vertices`, `*Edges`, `*Arcs`, `*Links`)
- Output files: `.tree`, `.clu`, `.ftree`
- Deterministic seeded trials
- Parallel trial execution (`--threads`)
- Unknown/unsupported flags are ignored

Out of scope in v1:

- State networks
- Multilayer networks
- Multiplex networks

## Build

```bash
cargo build --release
```

Binary:

```bash
target/release/minimap
```

## Usage

```bash
minimap network_file out_directory [options]
```

Supported options (v1):

- `--directed`
- `--two-level`
- `--seed <u32>`
- `--num-trials <u32>`
- `--threads <usize>` or `--threads=<usize>`
- `--parity-rng` (use MT19937-compatible RNG for parity-focused runs)
- `--tree`
- `--clu`
- `--ftree`
- `-o, --output tree,clu,ftree`
- `--out-name <name>`
- `--silent`
- `--clu-level <i32>`

Examples:

```bash
# Undirected, write only .tree
minimap modular_w.net . --two-level --tree --seed 123 --num-trials 8 --threads 8 --silent
```

```bash
# Directed, write all supported outputs
minimap modular_wd.net . --directed --two-level --output tree,clu,ftree --seed 123 --num-trials 8 --threads 8 --silent
```

## Implementation overview

`minimap` is implemented with cache-friendly array layouts rather than linked lists.

- Parser:
  - Reads link-list and Pajek sections
  - Deduplicates edges by `(source, target)` and aggregates weights
- Graph storage:
  - CSR-style arrays for outgoing edges (`out_offsets`, `edge_source`, `edge_target`, `edge_weight`)
  - Reverse index for incoming traversal (`in_offsets`, `in_edge_idx`)
  - Edge flow stored in-place (`edge_flow`)
- Flow engine:
  - Undirected flow model
  - Directed flow with deterministic power iteration and teleportation behavior matched to Infomap expectations
- Optimizer:
  - Two-level-only map equation optimization
  - Array-backed active network/module state
  - Rust-native seeded RNG by default (`SmallRng`)
  - MT19937-compatible seeded RNG when `--parity-rng` is set
  - Parallel trial evaluation with Rayon thread pool (`--threads`)
- Writers:
  - Streaming `.tree`, `.clu`, `.ftree` output
  - Deterministic ordering for stable file generation
  - Header format aligned with Infomap-compatible output style

## Benchmark examples

Example numbers from local runs on **2026-02-15** (`100k` synthetic edge list, one run per point, Infomap built with OpenMP and run with `--inner-parallelization`):

| `--num-trials` | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Codelength |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `infomap` (`OMP_NUM_THREADS=1`) | 1 | 4.71 s | 42,800 | 2138 | 4.93197 |
| 1 | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 1.52 s | 70,880 | 2136 | 4.93218 |
| 1 | `minimap` | 1 | 0.06 s | 29,984 | 2167 | 4.93144 |
| 1 | `minimap` | 1 | 0.05 s | 29,280 | 2167 | 4.93144 |
| 8 | `infomap` (`OMP_NUM_THREADS=1`) | 1 | 35.14 s | 141,456 | 2135 | 4.93177 |
| 8 | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 11.20 s | 168,608 | 2133 | 4.93187 |
| 8 | `minimap --threads 1` | 1 | 0.26 s | 35,456 | 2167 | 4.93144 |
| 8 | `minimap --threads 8` | 8 | 0.12 s | 152,128 | 2167 | 4.93144 |

Notes:

- Peak RSS was collected with `/usr/bin/time -lp` and converted from bytes to KiB (`bytes / 1024`).
- These are environment-specific example results, not universal guarantees.

## Author

Anton Holmgren
