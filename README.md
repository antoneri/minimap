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
  - Deterministic MT19937-compatible seeded trials
  - Parallel trial evaluation with Rayon thread pool (`--threads`)
- Writers:
  - Streaming `.tree`, `.clu`, `.ftree` output
  - Deterministic ordering for stable file generation
  - Header format aligned with Infomap-compatible output style

## Benchmark examples

Example numbers from local runs on **2026-02-14** (release build, 1M-edge synthetic network, `--num-trials 8`, median of 3 runs):

| Tool/config | Wall time | Peak RSS |
| --- | ---: | ---: |
| Infomap (no OpenMP build) | 21.16 s | 1254.9 MiB |
| minimap `--threads 1` | 4.95 s | 265.4 MiB |
| minimap `--threads 8` | 1.35 s | 1253.9 MiB |

Derived ratios from the above run:

- minimap `--threads 1` vs Infomap: `4.27x` faster, much lower RSS
- minimap `--threads 8` vs Infomap: `15.67x` faster, similar RSS
- minimap `--threads 8` vs minimap `--threads 1`: `3.67x` faster, higher RSS

Notes:

- Peak RSS was collected with `/usr/bin/time -lp`.
- These are example results, not universal guarantees.
