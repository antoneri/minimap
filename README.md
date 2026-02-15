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

Benchmark harness policy: Infomap runs use `OMP_NUM_THREADS=12`.

Example numbers from local runs on **2026-02-15** using:

- `python3 scripts/run_benchmark_matrix.py --out-dir /tmp/minimap_bench_matrix_latest2`
- 4 modes (`undirected|directed` x `two-level|multilevel`)
- trials `{1, 8}`
- datasets `{100k, 1M}` synthetic edge lists

### 100k (`--num-trials 1`)

| Mode | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Depth | Codelength |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| undirected/multilevel | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 3.43 s | 82240 | 1320 | 2 | 13.8494 |
| undirected/multilevel | `minimap --threads 1` | 1 | 0.35 s | 31792 | 1371 | 2 | 13.8570 |
| undirected/two-level | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 3.30 s | 89408 | 1303 | 2 | 13.8510 |
| undirected/two-level | `minimap --threads 1` | 1 | 0.26 s | 31472 | 1390 | 2 | 13.8630 |
| directed/multilevel | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 24.49 s | 82064 | 54 | 5 | 10.0246 |
| directed/multilevel | `minimap --threads 1` | 1 | 0.33 s | 34128 | 1993 | 5 | 10.1281 |
| directed/two-level | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 4.37 s | 83488 | 1989 | 2 | 10.1635 |
| directed/two-level | `minimap --threads 1` | 1 | 0.23 s | 31408 | 1997 | 2 | 10.1635 |

### 100k (`--num-trials 8`)

| Mode | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Depth | Codelength |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| undirected/multilevel | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 27.20 s | 159312 | 1316 | 2 | 13.8425 |
| undirected/multilevel | `minimap --threads 8` | 8 | 0.75 s | 135936 | 1385 | 2 | 13.8119 |
| undirected/two-level | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 24.51 s | 157792 | 1291 | 2 | 13.8444 |
| undirected/two-level | `minimap --threads 8` | 8 | 0.51 s | 119696 | 1405 | 2 | 13.8473 |
| directed/multilevel | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 181.59 s | 166160 | 54 | 5 | 10.0267 |
| directed/multilevel | `minimap --threads 8` | 8 | 0.88 s | 153424 | 6 | 7 | 10.1076 |
| directed/two-level | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 41.48 s | 152256 | 1985 | 2 | 10.1597 |
| directed/two-level | `minimap --threads 8` | 8 | 0.31 s | 118704 | 2011 | 2 | 10.1548 |

### 1M (`--num-trials 1`)

| Mode | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Depth | Codelength |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| undirected/multilevel | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 390.57 s | 471792 | 16219 | 2 | 16.7107 |
| undirected/multilevel | `minimap --threads 1` | 1 | 5.40 s | 258368 | 13801 | 2 | 16.7578 |
| undirected/two-level | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 405.15 s | 487120 | 16224 | 2 | 16.7104 |
| undirected/two-level | `minimap --threads 1` | 1 | 3.90 s | 289040 | 13801 | 2 | 16.7636 |
| directed/multilevel | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 18.74 s | 557840 | 17177 | 5 | 12.0636 |
| directed/multilevel | `minimap --threads 1` | 1 | 4.70 s | 251264 | 16431 | 5 | 12.1188 |
| directed/two-level | `infomap` (`OMP_NUM_THREADS=12`) | 12 | 15.16 s | 549104 | 17177 | 2 | 12.1218 |
| directed/two-level | `minimap --threads 1` | 1 | 3.26 s | 251408 | 16384 | 2 | 12.1756 |

### 1M (`--num-trials 8`)

| Mode | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Depth | Codelength |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| undirected/multilevel | `minimap --threads 8` | 8 | 7.47 s | 1064416 | 13961 | 2 | 16.7561 |
| undirected/two-level | `minimap --threads 8` | 8 | 6.02 s | 923152 | 13784 | 2 | 16.7621 |
| directed/multilevel | `minimap --threads 8` | 8 | 7.10 s | 1155584 | 16445 | 5 | 12.1155 |
| directed/two-level | `minimap --threads 8` | 8 | 5.67 s | 935456 | 16463 | 2 | 12.1694 |

Notes:

- Peak RSS was collected with `/usr/bin/time -lp` and converted from bytes to KiB (`bytes / 1024`).
- Infomap rows for `1M` + `--num-trials 8` are intentionally skipped by default in the matrix harness (`--skip-infomap-1m-trials8`) due very high runtime.
- The matrix harness includes fallback retry for Infomap runs that fail with `--inner-parallelization`; retries keep `OMP_NUM_THREADS=12` and remove only `--inner-parallelization` for that case.
- These are environment-specific example results, not universal guarantees.

## Author

Anton Holmgren
