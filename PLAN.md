# Minimap v1 Plan Update: Performance-First (Speed #1, Memory #2)

## Summary
Implement `minimap` as a **deterministic, cache-friendly, two-level-only** Rust engine optimized for:
1. **Execution speed**: at least **1.5x faster** than Infomap on agreed benchmarks.
2. **Memory**: peak RSS **<= Infomap** on the same benchmarks.
3. **Parity**: exact output parity after normalizing only:
   - `# started at ...`
   - `# completed in ... s`

## Public APIs / Interfaces / Types
1. Public interface remains CLI-only:
   - `minimap network_file out_directory [options]`
2. Supported v1 options:
   - `--directed`, `--two-level`, `--seed`, `--num-trials`, `--tree`, `--clu`, `--ftree`, `-o/--output`, `--out-name`, `--silent`, `--clu-level`
3. Unknown/unsupported flags are ignored.
4. Internal core types:
   - Node/module IDs: `u32`
   - Weights/flows/codelength: `f64`
   - No linked-list node structures.

## Implementation Design (Performance + Memory)

### 1. Data Layout (replace linked lists)
1. Use contiguous arrays and index-based references only.
2. Graph storage:
   - `out_offsets: Vec<u32>` (CSR)
   - `edge_target: Vec<u32>`
   - `edge_source: Vec<u32>`
   - `edge_value: Vec<f64>` (weight, then overwritten by flow after flow calc)
   - `in_offsets: Vec<u32>`
   - `in_edge_idx: Vec<u32>` (points into edge arrays, no duplicated flow)
3. Node flow arrays (SoA):
   - `node_flow`, `node_enter`, `node_exit`, `node_teleflow`, `node_teleweight`
4. Module arrays (SoA):
   - `node_module`, `module_size`, `module_flow_data`
5. All hot-path arrays preallocated and reused.

### 2. Parser (standard networks only, low-overhead)
1. Parse link-list + Pajek `*Vertices`, `*Edges`, `*Arcs`, `*Links`.
2. Deduplicate edges during parse via hash map keyed by `(source,target)`.
3. Freeze into sorted contiguous arrays, then free parse-time maps.
4. Keep name table compact:
   - store names only when present
   - fallback to numeric ID at output time.

### 3. Flow Engine (fast, deterministic)
1. Implement only required flow models:
   - undirected
   - directed (default teleport behavior matching Infomap)
2. Directed flow:
   - dense node-index mapping
   - two-buffer power iteration
   - deterministic loop order
3. After flow writeback, discard temporary buffers immediately.

### 4. Two-Level Optimizer (array-native)
1. Port two-level logic (`findTopModulesRepeatedly`, fine/coarse tune, core loops) with array-backed state.
2. Node move candidate evaluation uses reusable scratch buffers:
   - touched module list
   - stamped redirect array
   - delta enter/exit arrays
3. No per-node heap allocations in the optimization loop.
4. Preserve deterministic randomization with MT19937-compatible generator.

### 5. Output Writers (`.tree`, `.clu`, `.ftree`)
1. Stream writes directly to file, no large output buffers.
2. Preserve Infomap formatting/precision/order semantics.
3. For `.ftree` links, aggregate with deterministic ordered maps (`BTreeMap`) for stable byte output.
4. Force two-level path structure while keeping Infomap-compatible file schemas.

### 6. Build/Profile Tuning
1. Release profile tuned for throughput:
   - LTO
   - `codegen-units=1`
   - `panic=abort`
2. Single-thread deterministic default (no parallel mode in v1).

## Performance and Memory Acceptance Gates
1. Benchmark scale target: up to **10M edges**.
2. Runtime gate:
   - median wall-clock <= `0.67 * Infomap` (>=1.5x faster)
3. Memory gate:
   - peak RSS <= Infomap peak RSS
4. Parity gate:
   - normalized golden-file equality for `.tree`, `.clu`, `.ftree`

## Test Cases and Scenarios
1. Parser correctness:
   - link-list + Pajek variants
   - weighted/unweighted, directed/undirected headings
   - duplicate edge aggregation
2. Flow correctness:
   - deterministic repeated runs with same seed
   - invariants on node/link flow sums
3. Optimizer correctness:
   - fixed-seed module assignment stability
   - two-level result consistency across trials
4. Output parity:
   - golden comparisons vs Infomap oracle for:
     - `modular_w.net`
     - `modular_wd.net`
     - `--clu-level` variants
     - `--out-name` and `--output` combinations
5. Resource benchmarks:
   - automated speed and peak RSS comparison harness.

## Assumptions and Defaults
1. Two-level optimization is always used in v1.
2. Strict normalized parity is mandatory.
3. Unknown flags are ignored.
4. Single-thread deterministic execution is the only v1 mode.
5. Scope excludes state/multilayer/multiplex models.
