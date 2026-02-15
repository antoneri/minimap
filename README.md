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

- Multilevel hierarchy mode by default
- Optional two-level mode (`--two-level`)
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
- `--multilevel`
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

```bash
# Multilevel hierarchy output
minimap ninetriangles.net . --multilevel --output tree,clu,ftree --seed 123 --num-trials 1 --silent
```

## Implementation overview

`minimap` is implemented with cache-friendly array layouts rather than linked lists.

- Parser:
  - Reads link-list and Pajek sections
  - Deduplicates edges by `(source, target)` and aggregates weights
- Graph storage:
  - Hot/cold node split:
    - Hot flow fields in a contiguous `node_data: Vec<FlowData>`
    - Cold metadata split into `node_ids`, `node_names`, and `node_input_weight`
  - CSR-style arrays for outgoing edges (`out_offsets`, `edge_source`, `edge_target`, `edge_weight`)
  - Reverse index for incoming traversal (`in_offsets`, `in_edge_idx`)
  - Edge flow stored in-place (`edge_flow`)
- Flow engine:
  - Undirected flow model
  - Directed flow with deterministic power iteration and teleportation behavior matched to Infomap expectations
- Optimizer:
  - Two-level map equation optimization with optional multilevel hierarchy retention (`--multilevel`)
  - Array-backed active network/module state with dual edge storage:
    - Borrowed graph CSR at level 0 (no duplicated adjacency copy)
    - Owned compact arrays after consolidation levels
  - Storage-specialized hot kernels (borrowed vs owned) selected once per phase, avoiding per-edge storage branching
  - Flat module membership storage (`member_leaf` + per-module spans), replacing per-node member vectors
  - Per-trial reusable workspace for hot buffers (node order, candidates, redirects, consolidation scratch)
  - Candidate move scoring reuses precomputed old-module delta context to reduce repeated `plogp` work
  - Consolidation uses array-native stamp/touched-list aggregation (no hash map in the hot path)
  - Rust-native seeded RNG by default (`SmallRng`)
  - MT19937-compatible seeded RNG when `--parity-rng` is set
  - Parallel trial evaluation with Rayon thread pool (`--threads`)
- Writers:
  - Streaming `.tree`, `.clu`, `.ftree` output
  - Multilevel path/link serialization when hierarchy mode is enabled
  - Deterministic ordering for stable file generation
  - Header format aligned with Infomap-compatible output style

## Benchmark examples

Example numbers from local runs on **2026-02-15** (`100k` synthetic edge list, one run per point, Infomap built with OpenMP and run with `--inner-parallelization`):

| `--num-trials` | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Codelength |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `infomap` (`OMP_NUM_THREADS=1`) | 1 | 11.37 s | 55,296 | 1630 | 13.4330 |
| 1 | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 2.43 s | 90,128 | 1639 | 13.4342 |
| 1 | `minimap` | 1 | 0.17 s | 31,216 | 1598 | 13.4361 |
| 8 | `infomap` (`OMP_NUM_THREADS=1`) | 1 | 86.38 s | 126,864 | 1630 | 13.4330 |
| 8 | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 17.48 s | 173,824 | 1630 | 13.4353 |
| 8 | `minimap --threads 1` | 1 | 1.15 s | 39,120 | 1602 | 13.4339 |
| 8 | `minimap --threads 8` | 8 | 0.21 s | 113,888 | 1602 | 13.4339 |

Notes:

- Peak RSS was collected with `/usr/bin/time -lp` and converted from bytes to KiB (`bytes / 1024`).
- These are environment-specific example results, not universal guarantees.

## Author

Anton Holmgren
