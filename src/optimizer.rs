use crate::graph::{FlowData, Graph};
use crate::objective::{DeltaFlow, MapEquationObjective, plogp};
use crate::rng_compat::{Mt19937, RustRng, TrialRng};
use rayon::prelude::*;

const CORE_LOOP_LIMIT: usize = 10;
const MIN_CODELENGTH_IMPROVEMENT: f64 = 1e-10;
const MIN_SINGLE_NODE_IMPROVEMENT: f64 = 1e-16;
const MIN_RELATIVE_TUNE_ITERATION_IMPROVEMENT: f64 = 1e-5;

#[inline]
fn trace_super_enabled() -> bool {
    std::env::var_os("MINIMAP_TRACE_SUPER").is_some()
}

#[derive(Debug, Clone, Copy, Default)]
struct EdgeSpan {
    start: u32,
    end: u32,
}

#[derive(Debug, Clone)]
struct ActiveNode {
    data: FlowData,
    out_span: EdgeSpan,
    in_span: EdgeSpan,
    member_span: EdgeSpan,
}

#[derive(Debug, Clone)]
enum EdgeStorage<'g> {
    // Unlike Infomap's linked structures, level 0 borrows graph CSR directly.
    Borrowed {
        edge_source: &'g [u32],
        edge_target: &'g [u32],
        edge_flow: &'g [f64],
        in_edge_idx: &'g [u32],
    },
    // After consolidation, we keep compact owned arrays for cache-friendly traversal.
    Owned {
        out_neighbor: Vec<u32>,
        out_flow: Vec<f64>,
        in_neighbor: Vec<u32>,
        in_flow: Vec<f64>,
    },
}

trait AdjAccess {
    fn out_neighbor(&self, edge_idx: usize) -> u32;
    fn out_flow(&self, edge_idx: usize) -> f64;
    fn in_neighbor(&self, edge_idx: usize) -> u32;
    fn in_flow(&self, edge_idx: usize) -> f64;
}

struct BorrowedAdj<'a> {
    edge_source: &'a [u32],
    edge_target: &'a [u32],
    edge_flow: &'a [f64],
    in_edge_idx: &'a [u32],
}

impl AdjAccess for BorrowedAdj<'_> {
    #[inline]
    fn out_neighbor(&self, edge_idx: usize) -> u32 {
        self.edge_target[edge_idx]
    }

    #[inline]
    fn out_flow(&self, edge_idx: usize) -> f64 {
        self.edge_flow[edge_idx]
    }

    #[inline]
    fn in_neighbor(&self, edge_idx: usize) -> u32 {
        self.edge_source[self.in_edge_idx[edge_idx] as usize]
    }

    #[inline]
    fn in_flow(&self, edge_idx: usize) -> f64 {
        self.edge_flow[self.in_edge_idx[edge_idx] as usize]
    }
}

struct OwnedAdj<'a> {
    out_neighbor: &'a [u32],
    out_flow: &'a [f64],
    in_neighbor: &'a [u32],
    in_flow: &'a [f64],
}

impl AdjAccess for OwnedAdj<'_> {
    #[inline]
    fn out_neighbor(&self, edge_idx: usize) -> u32 {
        self.out_neighbor[edge_idx]
    }

    #[inline]
    fn out_flow(&self, edge_idx: usize) -> f64 {
        self.out_flow[edge_idx]
    }

    #[inline]
    fn in_neighbor(&self, edge_idx: usize) -> u32 {
        self.in_neighbor[edge_idx]
    }

    #[inline]
    fn in_flow(&self, edge_idx: usize) -> f64 {
        self.in_flow[edge_idx]
    }
}

#[derive(Debug, Clone)]
struct ActiveNetwork<'g> {
    nodes: Vec<ActiveNode>,
    // Flat member pool with per-module spans replaces per-node Vec<Vec<...>> style storage.
    member_leaf: Vec<u32>,
    edge_storage: EdgeStorage<'g>,
}

#[derive(Debug, Clone)]
struct OptimizeLevelResult {
    node_module: Vec<u32>,
    module_data: Vec<FlowData>,
    codelength: f64,
    effective_loops: u32,
}

#[derive(Debug, Default)]
struct OptimizeWorkspace {
    // Reused across loops/trials to avoid allocator churn in the hot optimizer path.
    node_order: Vec<u32>,
    redirect: Vec<u32>,
    touched_modules: Vec<u32>,
    cand_modules: Vec<u32>,
    cand_delta_exit: Vec<f64>,
    cand_delta_enter: Vec<f64>,
    module_order: Vec<u32>,
    module_indices: Vec<u32>,
    flow_data: Vec<FlowData>,
    consolidate_module_counts: Vec<u32>,
    consolidate_module_offsets: Vec<u32>,
    consolidate_module_fill: Vec<u32>,
    consolidate_module_nodes: Vec<u32>,
    consolidate_member_counts: Vec<u32>,
    consolidate_member_offsets: Vec<u32>,
    consolidate_member_fill: Vec<u32>,
    consolidate_dst_stamp: Vec<u32>,
    consolidate_dst_flow: Vec<f64>,
    consolidate_touched_dsts: Vec<u32>,
    consolidate_edge_src: Vec<u32>,
    consolidate_edge_dst: Vec<u32>,
    consolidate_edge_flow: Vec<f64>,
    consolidate_out_counts: Vec<u32>,
    consolidate_in_counts: Vec<u32>,
    consolidate_stamp_gen: u32,
    // Reused by multilevel recursive partitioning.
    recurse_queue: Vec<u32>,
    recurse_next_queue: Vec<u32>,
    recurse_module_children: Vec<u32>,
    recurse_layers: Vec<Vec<u32>>,
    recurse_next_level_leaf_modules: Vec<u32>,
    recurse_local_module_data: Vec<FlowData>,
    recurse_sub_global_to_local: Vec<u32>,
    recurse_sub_touched_global: Vec<u32>,
    recurse_sub_edge_src: Vec<u32>,
    recurse_sub_edge_dst: Vec<u32>,
    recurse_sub_edge_flow: Vec<f64>,
    recurse_sub_out_counts: Vec<u32>,
    recurse_sub_in_counts: Vec<u32>,
    recurse_sub_in_fill: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub node_to_module: Vec<u32>,
    pub num_modules: u32,
    pub levels: u32,
    pub codelength: f64,
    pub one_level_codelength: f64,
    pub module_data: Vec<FlowData>,
    pub hierarchy: Option<HierarchyResult>,
}

#[derive(Debug, Clone)]
pub struct HierarchyResult {
    // Per-leaf module chain from top to deepest module containing the leaf.
    pub leaf_paths: Vec<Vec<u32>>,
    // Top module indices (unordered; writer applies deterministic ordering).
    pub top_modules: Vec<u32>,
    pub levels: u32,
    // Parent module index or u32::MAX for top modules.
    pub module_parent: Vec<u32>,
    // Encoded child ids per module:
    // leaves are [0, leaf_count), modules are [leaf_count, leaf_count + module_count).
    pub module_children_offsets: Vec<u32>,
    pub module_children: Vec<u32>,
    // Aggregated flow data for every module in this hierarchy.
    pub module_data: Vec<FlowData>,
}

#[derive(Debug, Clone)]
struct DynamicHierarchy {
    leaf_count: usize,
    module_parent: Vec<u32>,
    module_child_start: Vec<u32>,
    module_child_end: Vec<u32>,
    // Flat pooled children storage for all modules.
    children: Vec<u32>,
    top_modules: Vec<u32>,
}

impl DynamicHierarchy {
    #[inline]
    fn leaf_base(&self) -> u32 {
        self.leaf_count as u32
    }

    fn from_result(hier: &HierarchyResult, leaf_count: usize) -> Self {
        let module_count = hier.module_parent.len();
        let mut module_child_start = Vec::with_capacity(module_count);
        let mut module_child_end = Vec::with_capacity(module_count);
        for m in 0..module_count {
            let start = hier.module_children_offsets[m] as usize;
            let end = hier.module_children_offsets[m + 1] as usize;
            module_child_start.push(start as u32);
            module_child_end.push(end as u32);
        }
        Self {
            leaf_count,
            module_parent: hier.module_parent.clone(),
            module_child_start,
            module_child_end,
            children: hier.module_children.clone(),
            top_modules: hier.top_modules.clone(),
        }
    }

    #[inline]
    fn module_count(&self) -> usize {
        self.module_parent.len()
    }

    #[inline]
    fn child_range(&self, module_idx: u32) -> std::ops::Range<usize> {
        let i = module_idx as usize;
        self.module_child_start[i] as usize..self.module_child_end[i] as usize
    }

    #[inline]
    fn children_slice(&self, module_idx: u32) -> &[u32] {
        let range = self.child_range(module_idx);
        &self.children[range]
    }

    #[inline]
    fn child_count(&self, module_idx: u32) -> usize {
        let i = module_idx as usize;
        (self.module_child_end[i] - self.module_child_start[i]) as usize
    }

    #[inline]
    fn add_module_with_children(&mut self, parent: u32, children: &[u32]) -> u32 {
        let module_idx = self.module_parent.len() as u32;
        self.module_parent.push(parent);
        let start = self.children.len() as u32;
        self.children.extend_from_slice(children);
        let end = self.children.len() as u32;
        self.module_child_start.push(start);
        self.module_child_end.push(end);
        module_idx
    }

    #[inline]
    fn set_module_children(&mut self, module_idx: u32, children: &[u32]) {
        let start = self.children.len() as u32;
        self.children.extend_from_slice(children);
        let end = self.children.len() as u32;
        let i = module_idx as usize;
        self.module_child_start[i] = start;
        self.module_child_end[i] = end;
    }
}

impl<'g> ActiveNetwork<'g> {
    #[inline]
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    fn out_range(&self, node_idx: usize) -> std::ops::Range<usize> {
        let span = self.nodes[node_idx].out_span;
        span.start as usize..span.end as usize
    }

    #[inline]
    fn in_range(&self, node_idx: usize) -> std::ops::Range<usize> {
        let span = self.nodes[node_idx].in_span;
        span.start as usize..span.end as usize
    }

    #[inline]
    fn member_range(&self, node_idx: usize) -> std::ops::Range<usize> {
        let span = self.nodes[node_idx].member_span;
        span.start as usize..span.end as usize
    }

    fn prefix_offsets(counts: &[u32]) -> Vec<u32> {
        let mut offsets = vec![0u32; counts.len() + 1];
        for i in 0..counts.len() {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        offsets
    }

    fn from_graph(graph: &'g Graph) -> Self {
        let n = graph.node_count();
        let mut nodes = Vec::with_capacity(n);
        let mut member_leaf = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(ActiveNode {
                data: graph.node_data[i],
                out_span: EdgeSpan {
                    start: graph.out_offsets[i],
                    end: graph.out_offsets[i + 1],
                },
                in_span: EdgeSpan {
                    start: graph.in_offsets[i],
                    end: graph.in_offsets[i + 1],
                },
                member_span: EdgeSpan {
                    start: i as u32,
                    end: (i + 1) as u32,
                },
            });
            member_leaf.push(i as u32);
        }

        Self {
            nodes,
            member_leaf,
            // Level 0 keeps zero-copy adjacency references into the parsed graph.
            edge_storage: EdgeStorage::Borrowed {
                edge_source: &graph.edge_source,
                edge_target: &graph.edge_target,
                edge_flow: &graph.edge_flow,
                in_edge_idx: &graph.in_edge_idx,
            },
        }
    }

    fn consolidate(
        &self,
        node_module: &[u32],
        module_data: &[FlowData],
        directed: bool,
        workspace: &mut OptimizeWorkspace,
    ) -> Self {
        match &self.edge_storage {
            EdgeStorage::Borrowed {
                edge_source,
                edge_target,
                edge_flow,
                in_edge_idx,
            } => {
                let adj = BorrowedAdj {
                    edge_source,
                    edge_target,
                    edge_flow,
                    in_edge_idx,
                };
                self.consolidate_with_adj(node_module, module_data, directed, workspace, &adj)
            }
            EdgeStorage::Owned {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            } => {
                let adj = OwnedAdj {
                    out_neighbor,
                    out_flow,
                    in_neighbor,
                    in_flow,
                };
                self.consolidate_with_adj(node_module, module_data, directed, workspace, &adj)
            }
        }
    }

    fn consolidate_with_adj<A: AdjAccess>(
        &self,
        node_module: &[u32],
        module_data: &[FlowData],
        directed: bool,
        workspace: &mut OptimizeWorkspace,
        adj: &A,
    ) -> Self {
        let n = self.node_count();
        let mut old_to_new = vec![u32::MAX; module_data.len()];
        let mut ordered_old_modules = Vec::<u32>::new();

        for &m in node_module {
            let m_idx = m as usize;
            if old_to_new[m_idx] == u32::MAX {
                old_to_new[m_idx] = ordered_old_modules.len() as u32;
                ordered_old_modules.push(m);
            }
        }

        let mut new_nodes: Vec<ActiveNode> = (0..ordered_old_modules.len())
            .map(|_| ActiveNode {
                data: FlowData::default(),
                out_span: EdgeSpan::default(),
                in_span: EdgeSpan::default(),
                member_span: EdgeSpan::default(),
            })
            .collect();

        for (new_m, &old_m) in ordered_old_modules.iter().enumerate() {
            new_nodes[new_m].data = module_data[old_m as usize];
        }

        let new_n = new_nodes.len();
        let OptimizeWorkspace {
            consolidate_module_counts,
            consolidate_module_offsets,
            consolidate_module_fill,
            consolidate_module_nodes,
            consolidate_member_counts,
            consolidate_member_offsets,
            consolidate_member_fill,
            consolidate_dst_stamp,
            consolidate_dst_flow,
            consolidate_touched_dsts,
            consolidate_edge_src,
            consolidate_edge_dst,
            consolidate_edge_flow,
            consolidate_out_counts,
            consolidate_in_counts,
            consolidate_stamp_gen,
            ..
        } = workspace;

        // Rebuild module membership as contiguous ranges to keep hierarchy data cache-local.
        consolidate_member_counts.resize(new_n, 0);
        consolidate_member_counts.fill(0);
        for i in 0..n {
            let new_m = old_to_new[node_module[i] as usize] as usize;
            let span = self.nodes[i].member_span;
            consolidate_member_counts[new_m] += span.end - span.start;
        }

        consolidate_member_offsets.resize(new_n + 1, 0);
        consolidate_member_offsets[0] = 0;
        for i in 0..new_n {
            consolidate_member_offsets[i + 1] =
                consolidate_member_offsets[i] + consolidate_member_counts[i];
        }

        consolidate_member_fill.resize(new_n, 0);
        consolidate_member_fill.fill(0);
        let total_members = consolidate_member_offsets[new_n] as usize;
        let mut member_leaf = vec![0u32; total_members];
        for i in 0..n {
            let new_m = old_to_new[node_module[i] as usize] as usize;
            let src_range = self.member_range(i);
            let len = src_range.len() as u32;
            let dst_start = consolidate_member_offsets[new_m] + consolidate_member_fill[new_m];
            let dst_end = dst_start + len;
            member_leaf[dst_start as usize..dst_end as usize]
                .copy_from_slice(&self.member_leaf[src_range]);
            consolidate_member_fill[new_m] += len;
        }
        for i in 0..new_n {
            new_nodes[i].member_span = EdgeSpan {
                start: consolidate_member_offsets[i],
                end: consolidate_member_offsets[i + 1],
            };
        }

        consolidate_module_counts.resize(new_n, 0);
        consolidate_module_counts.fill(0);
        for i in 0..n {
            let src_m = old_to_new[node_module[i] as usize] as usize;
            consolidate_module_counts[src_m] += 1;
        }

        consolidate_module_offsets.resize(new_n + 1, 0);
        consolidate_module_offsets[0] = 0;
        for i in 0..new_n {
            consolidate_module_offsets[i + 1] =
                consolidate_module_offsets[i] + consolidate_module_counts[i];
        }

        consolidate_module_fill.resize(new_n, 0);
        consolidate_module_fill.fill(0);
        consolidate_module_nodes.resize(n, 0);
        for i in 0..n {
            let src_m = old_to_new[node_module[i] as usize] as usize;
            let pos = consolidate_module_offsets[src_m] + consolidate_module_fill[src_m];
            consolidate_module_nodes[pos as usize] = i as u32;
            consolidate_module_fill[src_m] += 1;
        }

        consolidate_dst_stamp.resize(new_n, 0);
        consolidate_dst_flow.resize(new_n, 0.0);
        consolidate_touched_dsts.clear();
        consolidate_edge_src.clear();
        consolidate_edge_dst.clear();
        consolidate_edge_flow.clear();

        if directed {
            let mut stamp_gen = consolidate_stamp_gen.wrapping_add(1);
            if stamp_gen == 0 {
                consolidate_dst_stamp.fill(0);
                stamp_gen = 1;
            }

            for src_m in 0..new_n {
                consolidate_touched_dsts.clear();

                let start = consolidate_module_offsets[src_m] as usize;
                let end = consolidate_module_offsets[src_m + 1] as usize;
                for p in start..end {
                    let node_idx = consolidate_module_nodes[p] as usize;
                    for e in self.out_range(node_idx) {
                        let target_node = adj.out_neighbor(e) as usize;
                        let dst_m = old_to_new[node_module[target_node] as usize] as usize;
                        if src_m == dst_m {
                            continue;
                        }

                        let flow = adj.out_flow(e);
                        // Sparse stamp accumulator: same semantics as map aggregation, lower overhead.
                        if consolidate_dst_stamp[dst_m] != stamp_gen {
                            consolidate_dst_stamp[dst_m] = stamp_gen;
                            consolidate_dst_flow[dst_m] = flow;
                            consolidate_touched_dsts.push(dst_m as u32);
                        } else {
                            consolidate_dst_flow[dst_m] += flow;
                        }
                    }
                }

                consolidate_touched_dsts.sort_unstable();
                for &dst_m in consolidate_touched_dsts.iter() {
                    let dst_idx = dst_m as usize;
                    consolidate_edge_src.push(src_m as u32);
                    consolidate_edge_dst.push(dst_m);
                    consolidate_edge_flow.push(consolidate_dst_flow[dst_idx]);
                }

                stamp_gen = stamp_gen.wrapping_add(1);
                if stamp_gen == 0 {
                    consolidate_dst_stamp.fill(0);
                    stamp_gen = 1;
                }
            }
            *consolidate_stamp_gen = stamp_gen;
        } else {
            // Infomap undirected consolidation canonicalizes module pairs (min, max).
            let mut packed = Vec::<(u64, f64)>::new();
            for src_m in 0..new_n {
                let start = consolidate_module_offsets[src_m] as usize;
                let end = consolidate_module_offsets[src_m + 1] as usize;
                for p in start..end {
                    let node_idx = consolidate_module_nodes[p] as usize;
                    for e in self.out_range(node_idx) {
                        let target_node = adj.out_neighbor(e) as usize;
                        let dst_m = old_to_new[node_module[target_node] as usize] as usize;
                        if src_m == dst_m {
                            continue;
                        }

                        let mut a = src_m as u32;
                        let mut b = dst_m as u32;
                        if a > b {
                            std::mem::swap(&mut a, &mut b);
                        }
                        let key = ((a as u64) << 32) | (b as u64);
                        packed.push((key, adj.out_flow(e)));
                    }
                }
            }

            packed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            let mut i = 0usize;
            while i < packed.len() {
                let key = packed[i].0;
                let mut sum = packed[i].1;
                i += 1;
                while i < packed.len() && packed[i].0 == key {
                    sum += packed[i].1;
                    i += 1;
                }
                consolidate_edge_src.push((key >> 32) as u32);
                consolidate_edge_dst.push(key as u32);
                consolidate_edge_flow.push(sum);
            }
        }

        consolidate_out_counts.resize(new_n, 0);
        consolidate_in_counts.resize(new_n, 0);
        consolidate_out_counts.fill(0);
        consolidate_in_counts.fill(0);
        for idx in 0..consolidate_edge_src.len() {
            let src_m = consolidate_edge_src[idx] as usize;
            let dst_m = consolidate_edge_dst[idx] as usize;
            consolidate_out_counts[src_m] += 1;
            consolidate_in_counts[dst_m] += 1;
        }

        let out_offsets = Self::prefix_offsets(consolidate_out_counts);
        let in_offsets = Self::prefix_offsets(consolidate_in_counts);
        let total_edges = out_offsets[new_n] as usize;

        let mut out_neighbor = vec![0u32; total_edges];
        let mut out_flow = vec![0.0f64; total_edges];
        let mut in_neighbor = vec![0u32; total_edges];
        let mut in_flow = vec![0.0f64; total_edges];
        let mut in_fill = vec![0u32; new_n];

        for idx in 0..total_edges {
            out_neighbor[idx] = consolidate_edge_dst[idx];
            out_flow[idx] = consolidate_edge_flow[idx];
        }

        // In-edges are rebuilt from out-edge order so traversal stays deterministic.
        for idx in 0..total_edges {
            let src_m = consolidate_edge_src[idx];
            let dst_m = consolidate_edge_dst[idx];
            let flow = consolidate_edge_flow[idx];
            let dst_idx = dst_m as usize;

            let in_pos = in_offsets[dst_idx] + in_fill[dst_idx];
            in_neighbor[in_pos as usize] = src_m;
            in_flow[in_pos as usize] = flow;
            in_fill[dst_idx] += 1;
        }

        for i in 0..new_n {
            new_nodes[i].out_span = EdgeSpan {
                start: out_offsets[i],
                end: out_offsets[i + 1],
            };
            new_nodes[i].in_span = EdgeSpan {
                start: in_offsets[i],
                end: in_offsets[i + 1],
            };
        }

        Self {
            nodes: new_nodes,
            member_leaf,
            edge_storage: EdgeStorage::Owned {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            },
        }
    }

    fn assignment_to_leaves(&self, leaf_count: usize) -> Vec<u32> {
        let mut out = vec![0u32; leaf_count];
        for (module_idx, node) in self.nodes.iter().enumerate() {
            for p in node.member_span.start as usize..node.member_span.end as usize {
                let leaf = self.member_leaf[p];
                out[leaf as usize] = module_idx as u32;
            }
        }
        out
    }

    fn with_compact_members(&self) -> Self {
        let n = self.nodes.len();
        let mut out = self.clone();
        out.member_leaf = (0..n as u32).collect();
        for i in 0..n {
            out.nodes[i].member_span = EdgeSpan {
                start: i as u32,
                end: (i + 1) as u32,
            };
        }
        out
    }
}

#[inline]
fn add_candidate(
    module: u32,
    delta_exit: f64,
    delta_enter: f64,
    redirect: &mut [u32],
    touched_modules: &mut Vec<u32>,
    cand_modules: &mut Vec<u32>,
    cand_delta_exit: &mut Vec<f64>,
    cand_delta_enter: &mut Vec<f64>,
) {
    let m = module as usize;
    if redirect[m] == u32::MAX {
        redirect[m] = cand_modules.len() as u32;
        touched_modules.push(module);
        cand_modules.push(module);
        cand_delta_exit.push(delta_exit);
        cand_delta_enter.push(delta_enter);
    } else {
        let idx = redirect[m] as usize;
        cand_delta_exit[idx] += delta_exit;
        cand_delta_enter[idx] += delta_enter;
    }
}

#[inline]
fn degree_with_adj<A: AdjAccess>(active: &ActiveNetwork<'_>, adj: &A, node_idx: usize) -> usize {
    let mut degree = 0usize;
    for e in active.out_range(node_idx) {
        if adj.out_neighbor(e) as usize != node_idx {
            degree += 1;
        }
    }
    for e in active.in_range(node_idx) {
        if adj.in_neighbor(e) as usize != node_idx {
            degree += 1;
        }
    }
    degree
}

fn move_node_to_predefined_module<A: AdjAccess>(
    active: &ActiveNetwork<'_>,
    adj: &A,
    node_idx: usize,
    new_module: u32,
    objective: &mut MapEquationObjective,
    node_module: &mut [u32],
    module_data: &mut [FlowData],
    module_members: &mut [u32],
    empty_modules: &mut Vec<u32>,
) -> bool {
    let old_module = node_module[node_idx];
    if old_module == new_module {
        return false;
    }

    let mut old_delta = DeltaFlow {
        module: old_module,
        delta_exit: 0.0,
        delta_enter: 0.0,
    };
    let mut new_delta = DeltaFlow {
        module: new_module,
        delta_exit: 0.0,
        delta_enter: 0.0,
    };

    for e in active.out_range(node_idx) {
        let nbr_idx = adj.out_neighbor(e) as usize;
        if nbr_idx == node_idx {
            continue;
        }
        let flow = adj.out_flow(e);
        let other_module = node_module[nbr_idx];
        if other_module == old_module {
            old_delta.delta_exit += flow;
        } else if other_module == new_module {
            new_delta.delta_exit += flow;
        }
    }
    for e in active.in_range(node_idx) {
        let nbr_idx = adj.in_neighbor(e) as usize;
        if nbr_idx == node_idx {
            continue;
        }
        let flow = adj.in_flow(e);
        let other_module = node_module[nbr_idx];
        if other_module == old_module {
            old_delta.delta_enter += flow;
        } else if other_module == new_module {
            new_delta.delta_enter += flow;
        }
    }

    // Keep the same stack semantics as Infomap (candidate empty module is always back()).
    if module_members[new_module as usize] == 0 {
        let _ = empty_modules.pop();
    }
    if module_members[old_module as usize] == 1 {
        empty_modules.push(old_module);
    }

    objective.update_on_move(
        &active.nodes[node_idx].data,
        &old_delta,
        &new_delta,
        module_data,
    );

    module_members[old_module as usize] -= 1;
    module_members[new_module as usize] += 1;
    node_module[node_idx] = new_module;

    true
}

fn try_move_each_node_into_best_module<A: AdjAccess>(
    active: &ActiveNetwork<'_>,
    adj: &A,
    rng: &mut impl TrialRng,
    objective: &mut MapEquationObjective,
    node_module: &mut [u32],
    module_data: &mut [FlowData],
    module_members: &mut [u32],
    dirty: &mut [bool],
    empty_modules: &mut Vec<u32>,
    lock_multi_module_nodes: bool,
    workspace: &mut OptimizeWorkspace,
) -> u32 {
    let n = active.nodes.len();

    let OptimizeWorkspace {
        node_order,
        redirect,
        touched_modules,
        cand_modules,
        cand_delta_exit,
        cand_delta_enter,
        module_order,
        ..
    } = workspace;

    node_order.resize(n, 0);
    rng.randomized_index_vector(node_order);

    if redirect.len() != n {
        redirect.resize(n, u32::MAX);
    } else {
        redirect.fill(u32::MAX);
    }

    touched_modules.clear();
    cand_modules.clear();
    cand_delta_exit.clear();
    cand_delta_enter.clear();
    module_order.clear();

    let mut moved = 0u32;

    for &node_u32 in node_order.iter() {
        let node_idx = node_u32 as usize;

        if !dirty[node_idx] {
            continue;
        }

        let current_module = node_module[node_idx];

        // Match Infomap's "isFirstLoop" lock: don't move away from modules that
        // already have merged nodes on the first full-network tune iteration.
        if lock_multi_module_nodes && module_members[current_module as usize] > 1 {
            continue;
        }

        cand_modules.clear();
        cand_delta_exit.clear();
        cand_delta_enter.clear();
        touched_modules.clear();

        for e in active.out_range(node_idx) {
            let nbr_idx = adj.out_neighbor(e) as usize;
            if nbr_idx == node_idx {
                continue;
            }
            let flow = adj.out_flow(e);
            let m = node_module[nbr_idx];
            add_candidate(
                m,
                flow,
                0.0,
                redirect,
                touched_modules,
                cand_modules,
                cand_delta_exit,
                cand_delta_enter,
            );
        }

        for e in active.in_range(node_idx) {
            let nbr_idx = adj.in_neighbor(e) as usize;
            if nbr_idx == node_idx {
                continue;
            }
            let flow = adj.in_flow(e);
            let m = node_module[nbr_idx];
            add_candidate(
                m,
                0.0,
                flow,
                redirect,
                touched_modules,
                cand_modules,
                cand_delta_exit,
                cand_delta_enter,
            );
        }

        add_candidate(
            current_module,
            0.0,
            0.0,
            redirect,
            touched_modules,
            cand_modules,
            cand_delta_exit,
            cand_delta_enter,
        );

        if module_members[current_module as usize] > 1 {
            if let Some(&empty_module) = empty_modules.last() {
                add_candidate(
                    empty_module,
                    0.0,
                    0.0,
                    redirect,
                    touched_modules,
                    cand_modules,
                    cand_delta_exit,
                    cand_delta_enter,
                );
            }
        }

        let old_idx = redirect[current_module as usize] as usize;
        let old_delta = DeltaFlow {
            module: current_module,
            delta_exit: cand_delta_exit[old_idx],
            delta_enter: cand_delta_enter[old_idx],
        };
        // Precompute old-module terms once; candidate loop only varies the destination module.
        let move_context =
            objective.prepare_move_context(&active.nodes[node_idx].data, &old_delta, module_data);

        module_order.clear();
        module_order.resize(cand_modules.len(), 0);
        rng.randomized_index_vector(module_order);

        let mut best_module = current_module;
        let mut best_delta = old_delta;
        let mut best_delta_codelength = 0.0f64;

        let mut strongest_module = current_module;
        let mut strongest_delta = old_delta;
        let mut strongest_delta_codelength = 0.0f64;

        for &enum_idx in module_order.iter() {
            let cidx = enum_idx as usize;
            let other_module = cand_modules[cidx];
            if other_module == current_module {
                continue;
            }

            let cand_delta = DeltaFlow {
                module: other_module,
                delta_exit: cand_delta_exit[cidx],
                delta_enter: cand_delta_enter[cidx],
            };

            let delta =
                objective.get_delta_on_move_with_context(&move_context, &cand_delta, module_data);

            if delta < best_delta_codelength - MIN_SINGLE_NODE_IMPROVEMENT {
                best_module = other_module;
                best_delta = cand_delta;
                best_delta_codelength = delta;
            }

            if cand_delta.delta_exit > strongest_delta.delta_exit {
                strongest_module = other_module;
                strongest_delta = cand_delta;
                strongest_delta_codelength = delta;
            }
        }

        if strongest_module != best_module
            && strongest_delta_codelength <= best_delta_codelength + MIN_SINGLE_NODE_IMPROVEMENT
        {
            best_module = strongest_module;
            best_delta = strongest_delta;
        }

        if best_module != current_module {
            if module_members[best_module as usize] == 0 {
                let _ = empty_modules.pop();
            }
            if module_members[current_module as usize] == 1 {
                empty_modules.push(current_module);
            }

            objective.update_on_move(
                &active.nodes[node_idx].data,
                &old_delta,
                &best_delta,
                module_data,
            );

            module_members[current_module as usize] -= 1;
            module_members[best_module as usize] += 1;

            let old_module = current_module;
            node_module[node_idx] = best_module;
            moved += 1;

            let mut node_in_old_module = node_idx as u32;
            let mut num_linked_nodes_in_old_module = 0u32;

            for e in active.out_range(node_idx) {
                let nbr = adj.out_neighbor(e);
                let nbr_idx = nbr as usize;
                if nbr_idx == node_idx {
                    continue;
                }
                dirty[nbr_idx] = true;
                if node_module[nbr_idx] == old_module {
                    node_in_old_module = nbr;
                    num_linked_nodes_in_old_module += 1;
                }
            }
            for e in active.in_range(node_idx) {
                let nbr = adj.in_neighbor(e);
                let nbr_idx = nbr as usize;
                if nbr_idx == node_idx {
                    continue;
                }
                dirty[nbr_idx] = true;
                if node_module[nbr_idx] == old_module {
                    node_in_old_module = nbr;
                    num_linked_nodes_in_old_module += 1;
                }
            }

            // Move single connected node to same module.
            if num_linked_nodes_in_old_module == 1 && module_members[old_module as usize] == 1 {
                let companion_idx = node_in_old_module as usize;
                if move_node_to_predefined_module(
                    active,
                    adj,
                    companion_idx,
                    best_module,
                    objective,
                    node_module,
                    module_data,
                    module_members,
                    empty_modules,
                ) {
                    moved += 1;

                    if degree_with_adj(active, adj, companion_idx) > 1 {
                        for e in active.out_range(companion_idx) {
                            let nbr = adj.out_neighbor(e);
                            if nbr as usize != companion_idx {
                                dirty[nbr as usize] = true;
                            }
                        }
                        for e in active.in_range(companion_idx) {
                            let nbr = adj.in_neighbor(e);
                            if nbr as usize != companion_idx {
                                dirty[nbr as usize] = true;
                            }
                        }
                    }
                }
            }
        } else {
            dirty[node_idx] = false;
        }

        for &m in touched_modules.iter() {
            redirect[m as usize] = u32::MAX;
        }
        touched_modules.clear();
    }

    moved
}

fn optimize_active_network(
    active: &ActiveNetwork<'_>,
    rng: &mut impl TrialRng,
    objective: &mut MapEquationObjective,
    predefined_modules: Option<&[u32]>,
    lock_multi_module_nodes: bool,
    loop_limit: usize,
    workspace: &mut OptimizeWorkspace,
) -> OptimizeLevelResult {
    match &active.edge_storage {
        EdgeStorage::Borrowed {
            edge_source,
            edge_target,
            edge_flow,
            in_edge_idx,
        } => {
            let adj = BorrowedAdj {
                edge_source,
                edge_target,
                edge_flow,
                in_edge_idx,
            };
            optimize_active_network_with_adj(
                active,
                &adj,
                rng,
                objective,
                predefined_modules,
                lock_multi_module_nodes,
                loop_limit,
                workspace,
            )
        }
        EdgeStorage::Owned {
            out_neighbor,
            out_flow,
            in_neighbor,
            in_flow,
        } => {
            let adj = OwnedAdj {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            };
            optimize_active_network_with_adj(
                active,
                &adj,
                rng,
                objective,
                predefined_modules,
                lock_multi_module_nodes,
                loop_limit,
                workspace,
            )
        }
    }
}

fn optimize_active_network_with_adj<A: AdjAccess>(
    active: &ActiveNetwork<'_>,
    adj: &A,
    rng: &mut impl TrialRng,
    objective: &mut MapEquationObjective,
    predefined_modules: Option<&[u32]>,
    lock_multi_module_nodes: bool,
    loop_limit: usize,
    workspace: &mut OptimizeWorkspace,
) -> OptimizeLevelResult {
    let n = active.nodes.len();

    let mut node_module: Vec<u32> = (0..n as u32).collect();
    let mut module_data: Vec<FlowData> = active.nodes.iter().map(|n| n.data).collect();
    let mut module_members = vec![1u32; n];
    let mut dirty = vec![true; n];
    let mut empty_modules: Vec<u32> = Vec::with_capacity(n);

    workspace.module_indices.resize(n, 0);
    for i in 0..n {
        workspace.module_indices[i] = i as u32;
    }
    objective.init_partition(&module_data, &workspace.module_indices[..n]);

    if let Some(modules) = predefined_modules {
        if modules.len() != n {
            panic!(
                "predefined module length {} != active node count {}",
                modules.len(),
                n
            );
        }
        for i in 0..n {
            let new_m = modules[i];
            let _ = move_node_to_predefined_module(
                active,
                adj,
                i,
                new_m,
                objective,
                &mut node_module,
                &mut module_data,
                &mut module_members,
                &mut empty_modules,
            );
        }
    }

    let mut core_loop_count = 0usize;
    let mut effective_loops = 0u32;
    let mut old_codelength = objective.codelength;

    loop {
        core_loop_count += 1;

        let moved = try_move_each_node_into_best_module(
            active,
            adj,
            rng,
            objective,
            &mut node_module,
            &mut module_data,
            &mut module_members,
            &mut dirty,
            &mut empty_modules,
            lock_multi_module_nodes,
            workspace,
        );

        if moved == 0 || objective.codelength >= old_codelength - MIN_CODELENGTH_IMPROVEMENT {
            break;
        }

        effective_loops += 1;
        old_codelength = objective.codelength;

        if core_loop_count == loop_limit {
            break;
        }
    }

    OptimizeLevelResult {
        node_module,
        module_data,
        codelength: objective.codelength,
        effective_loops,
    }
}

fn assignment_from_top_network(top_network: &ActiveNetwork<'_>, leaf_count: usize) -> Vec<u32> {
    top_network.assignment_to_leaves(leaf_count)
}

fn partition_terms(
    objective: &mut MapEquationObjective,
    modules: &[FlowData],
    module_indices: &mut Vec<u32>,
) -> (f64, f64) {
    module_indices.resize(modules.len(), 0);
    for i in 0..modules.len() {
        module_indices[i] = i as u32;
    }
    objective.init_partition(modules, &module_indices[..modules.len()]);
    (objective.codelength, objective.index_codelength)
}

fn active_modules_codelength(
    active: &ActiveNetwork<'_>,
    objective: &mut MapEquationObjective,
    workspace: &mut OptimizeWorkspace,
) -> f64 {
    active_modules_terms(active, objective, workspace).0
}

fn active_modules_terms(
    active: &ActiveNetwork<'_>,
    objective: &mut MapEquationObjective,
    workspace: &mut OptimizeWorkspace,
) -> (f64, f64) {
    workspace.flow_data.clear();
    workspace.flow_data.reserve(active.nodes.len());
    for node in &active.nodes {
        workspace.flow_data.push(node.data);
    }
    partition_terms(
        objective,
        &workspace.flow_data,
        &mut workspace.module_indices,
    )
}

#[inline]
fn assignment_module_count(assignment: &[u32]) -> usize {
    let mut max = 0u32;
    for &m in assignment {
        max = max.max(m);
    }
    let mut seen = vec![false; max as usize + 1];
    let mut count = 0usize;
    for &m in assignment {
        let idx = m as usize;
        if !seen[idx] {
            seen[idx] = true;
            count += 1;
        }
    }
    count
}

fn find_top_modules_repeatedly_from_leaf<'g>(
    leaf_network: &ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    directed: bool,
    lock_first_loop: bool,
    allow_non_improving_hierarchy_levels: bool,
    workspace: &mut OptimizeWorkspace,
) -> ActiveNetwork<'g> {
    let mut have_modules = false;
    let mut active = leaf_network.clone();
    let mut consolidated_codelength = f64::INFINITY;
    let mut aggregation_level = 0usize;

    loop {
        if active.nodes.len() <= 1 {
            break;
        }

        let loop_limit = if aggregation_level > 0 {
            20
        } else {
            CORE_LOOP_LIMIT
        };
        let lock = lock_first_loop && aggregation_level == 0;
        let level =
            optimize_active_network(&active, rng, objective, None, lock, loop_limit, workspace);

        let next_module_count = assignment_module_count(&level.node_module);
        let allow_non_improving_level =
            allow_non_improving_hierarchy_levels && next_module_count < active.nodes.len();
        if have_modules
            && level.codelength >= consolidated_codelength - MIN_SINGLE_NODE_IMPROVEMENT
            && !allow_non_improving_level
        {
            break;
        }

        let next = active.consolidate(&level.node_module, &level.module_data, directed, workspace);
        consolidated_codelength = level.codelength;
        have_modules = true;
        aggregation_level += 1;

        if next.nodes.len() <= 1 {
            active = next;
            break;
        }

        active = next;
    }

    active
}

fn find_top_modules_repeatedly_from_modules<'g>(
    active_top: &ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    directed: bool,
    allow_non_improving_hierarchy_levels: bool,
    workspace: &mut OptimizeWorkspace,
) -> ActiveNetwork<'g> {
    let mut active = active_top.clone();
    let mut consolidated_codelength = active_modules_codelength(&active, objective, workspace);
    let mut aggregation_level = 0usize;

    loop {
        if active.nodes.len() <= 1 {
            break;
        }

        let loop_limit = if aggregation_level > 0 {
            20
        } else {
            CORE_LOOP_LIMIT
        };
        let level =
            optimize_active_network(&active, rng, objective, None, false, loop_limit, workspace);

        let next_module_count = assignment_module_count(&level.node_module);
        let allow_non_improving_level =
            allow_non_improving_hierarchy_levels && next_module_count < active.nodes.len();
        if level.codelength >= consolidated_codelength - MIN_SINGLE_NODE_IMPROVEMENT
            && !allow_non_improving_level
        {
            break;
        }

        let next = active.consolidate(&level.node_module, &level.module_data, directed, workspace);
        consolidated_codelength = level.codelength;
        aggregation_level += 1;

        if next.nodes.len() <= 1 {
            active = next;
            break;
        }

        active = next;
    }

    active
}

fn transform_node_flow_to_enter_flow(active: &mut ActiveNetwork<'_>) {
    for node in &mut active.nodes {
        node.data.flow = node.data.enter_flow;
    }
}

fn refresh_active_data_from_leaf_network(
    leaf_network: &ActiveNetwork<'_>,
    active: &mut ActiveNetwork<'_>,
    directed: bool,
) {
    if active.nodes.is_empty() {
        return;
    }
    let assignment = assignment_from_top_network(active, leaf_network.node_count());
    let module_data = compute_module_data_active(
        leaf_network,
        &assignment,
        active.nodes.len() as u32,
        directed,
    );
    for (node, data) in active.nodes.iter_mut().zip(module_data.into_iter()) {
        node.data = data;
    }
}

fn compute_module_data_active(
    active: &ActiveNetwork<'_>,
    node_to_module: &[u32],
    num_modules: u32,
    directed: bool,
) -> Vec<FlowData> {
    match &active.edge_storage {
        EdgeStorage::Borrowed {
            edge_source,
            edge_target,
            edge_flow,
            in_edge_idx,
        } => {
            let adj = BorrowedAdj {
                edge_source,
                edge_target,
                edge_flow,
                in_edge_idx,
            };
            compute_module_data_active_with_adj(active, &adj, node_to_module, num_modules, directed)
        }
        EdgeStorage::Owned {
            out_neighbor,
            out_flow,
            in_neighbor,
            in_flow,
        } => {
            let adj = OwnedAdj {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            };
            compute_module_data_active_with_adj(active, &adj, node_to_module, num_modules, directed)
        }
    }
}

fn compute_module_data_active_with_adj<A: AdjAccess>(
    active: &ActiveNetwork<'_>,
    adj: &A,
    node_to_module: &[u32],
    num_modules: u32,
    directed: bool,
) -> Vec<FlowData> {
    let mut modules = vec![FlowData::default(); num_modules as usize];

    for (i, node) in active.nodes.iter().enumerate() {
        let m = node_to_module[i] as usize;
        modules[m].flow += node.data.flow;
    }

    for s in 0..active.node_count() {
        let ms = node_to_module[s] as usize;
        for e in active.out_range(s) {
            let t = adj.out_neighbor(e) as usize;
            let mt = node_to_module[t] as usize;
            if ms == mt {
                continue;
            }
            let f = adj.out_flow(e);
            if directed {
                modules[ms].exit_flow += f;
                modules[mt].enter_flow += f;
            } else {
                let h = f / 2.0;
                modules[ms].exit_flow += h;
                modules[ms].enter_flow += h;
                modules[mt].exit_flow += h;
                modules[mt].enter_flow += h;
            }
        }
    }

    modules
}

fn induced_subnetwork<'g>(
    active: &ActiveNetwork<'g>,
    members: &[u32],
    workspace: &mut OptimizeWorkspace,
) -> ActiveNetwork<'g> {
    match &active.edge_storage {
        EdgeStorage::Borrowed {
            edge_source,
            edge_target,
            edge_flow,
            in_edge_idx,
        } => {
            let adj = BorrowedAdj {
                edge_source,
                edge_target,
                edge_flow,
                in_edge_idx,
            };
            induced_subnetwork_with_adj(active, members, workspace, &adj)
        }
        EdgeStorage::Owned {
            out_neighbor,
            out_flow,
            in_neighbor,
            in_flow,
        } => {
            let adj = OwnedAdj {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            };
            induced_subnetwork_with_adj(active, members, workspace, &adj)
        }
    }
}

fn induced_subnetwork_with_adj<'g, A: AdjAccess>(
    active: &ActiveNetwork<'g>,
    members: &[u32],
    workspace: &mut OptimizeWorkspace,
    adj: &A,
) -> ActiveNetwork<'g> {
    let k = members.len();
    let OptimizeWorkspace {
        recurse_sub_global_to_local,
        recurse_sub_touched_global,
        recurse_sub_edge_src,
        recurse_sub_edge_dst,
        recurse_sub_edge_flow,
        recurse_sub_out_counts,
        recurse_sub_in_counts,
        recurse_sub_in_fill,
        ..
    } = workspace;

    if recurse_sub_global_to_local.len() != active.node_count() {
        recurse_sub_global_to_local.resize(active.node_count(), u32::MAX);
    }
    recurse_sub_touched_global.clear();

    let mut nodes = Vec::with_capacity(k);
    let mut member_leaf = Vec::with_capacity(k);

    for (local_idx, &global_idx_u32) in members.iter().enumerate() {
        let global_idx = global_idx_u32 as usize;
        recurse_sub_global_to_local[global_idx] = local_idx as u32;
        recurse_sub_touched_global.push(global_idx_u32);
        nodes.push(ActiveNode {
            data: active.nodes[global_idx].data,
            out_span: EdgeSpan::default(),
            in_span: EdgeSpan::default(),
            member_span: EdgeSpan {
                start: local_idx as u32,
                end: local_idx as u32 + 1,
            },
        });
        member_leaf.push(local_idx as u32);
    }

    recurse_sub_edge_src.clear();
    recurse_sub_edge_dst.clear();
    recurse_sub_edge_flow.clear();
    recurse_sub_out_counts.resize(k, 0);
    recurse_sub_in_counts.resize(k, 0);
    recurse_sub_out_counts.fill(0);
    recurse_sub_in_counts.fill(0);

    for (local_src, &global_src_u32) in members.iter().enumerate() {
        let global_src = global_src_u32 as usize;
        for e in active.out_range(global_src) {
            let global_dst = adj.out_neighbor(e) as usize;
            let local_dst = recurse_sub_global_to_local[global_dst];
            if local_dst == u32::MAX {
                continue;
            }
            recurse_sub_edge_src.push(local_src as u32);
            recurse_sub_edge_dst.push(local_dst);
            recurse_sub_edge_flow.push(adj.out_flow(e));
            recurse_sub_out_counts[local_src] += 1;
            recurse_sub_in_counts[local_dst as usize] += 1;
        }
    }

    let out_offsets = ActiveNetwork::prefix_offsets(recurse_sub_out_counts);
    let in_offsets = ActiveNetwork::prefix_offsets(recurse_sub_in_counts);
    let m = recurse_sub_edge_src.len();

    let mut out_neighbor = vec![0u32; m];
    let mut out_flow = vec![0.0f64; m];
    for i in 0..m {
        out_neighbor[i] = recurse_sub_edge_dst[i];
        out_flow[i] = recurse_sub_edge_flow[i];
    }

    let mut in_neighbor = vec![0u32; m];
    let mut in_flow = vec![0.0f64; m];
    recurse_sub_in_fill.resize(k, 0);
    recurse_sub_in_fill.fill(0);
    for i in 0..m {
        let dst = recurse_sub_edge_dst[i] as usize;
        let pos = in_offsets[dst] + recurse_sub_in_fill[dst];
        in_neighbor[pos as usize] = recurse_sub_edge_src[i];
        in_flow[pos as usize] = recurse_sub_edge_flow[i];
        recurse_sub_in_fill[dst] += 1;
    }

    for i in 0..k {
        nodes[i].out_span = EdgeSpan {
            start: out_offsets[i],
            end: out_offsets[i + 1],
        };
        nodes[i].in_span = EdgeSpan {
            start: in_offsets[i],
            end: in_offsets[i + 1],
        };
    }

    for &global_idx in recurse_sub_touched_global.iter() {
        recurse_sub_global_to_local[global_idx as usize] = u32::MAX;
    }
    recurse_sub_touched_global.clear();

    ActiveNetwork {
        nodes,
        member_leaf,
        edge_storage: EdgeStorage::Owned {
            out_neighbor,
            out_flow,
            in_neighbor,
            in_flow,
        },
    }
}

fn coarse_tune<'g>(
    leaf_network: &ActiveNetwork<'g>,
    top_network: &mut ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    directed: bool,
    allow_non_improving_hierarchy_levels: bool,
    workspace: &mut OptimizeWorkspace,
) -> u32 {
    if top_network.nodes.len() <= 1 {
        return 0;
    }

    let leaf_count = leaf_network.nodes.len();
    let old_assignment = assignment_from_top_network(top_network, leaf_count);
    let mut sub_assignment = vec![u32::MAX; leaf_count];
    let mut submodule_to_old_module = Vec::<u32>::new();
    let mut module_index_offset = 0u32;

    for old_module in 0..top_network.nodes.len() {
        let members_range = top_network.member_range(old_module);
        let members = &top_network.member_leaf[members_range];

        if members.len() < 2 {
            for &leaf in members {
                sub_assignment[leaf as usize] = module_index_offset;
            }
            submodule_to_old_module.push(old_module as u32);
            module_index_offset += 1;
            continue;
        }

        let sub_active = induced_subnetwork(leaf_network, members, workspace);
        let mut sub_node_data = Vec::with_capacity(sub_active.nodes.len());
        for node in &sub_active.nodes {
            sub_node_data.push(node.data);
        }
        let mut sub_objective = MapEquationObjective::new(&sub_node_data);
        let sub_top = find_top_modules_repeatedly_from_leaf(
            &sub_active,
            &mut sub_objective,
            rng,
            directed,
            false,
            allow_non_improving_hierarchy_levels,
            workspace,
        );
        let sub_local_assignment = assignment_from_top_network(&sub_top, sub_active.node_count());
        let num_submodules = assignment_module_count(&sub_local_assignment) as u32;

        for (local_idx, &leaf) in members.iter().enumerate() {
            sub_assignment[leaf as usize] = module_index_offset + sub_local_assignment[local_idx];
        }
        for _ in 0..num_submodules {
            submodule_to_old_module.push(old_module as u32);
        }
        module_index_offset += num_submodules;
    }

    debug_assert_eq!(module_index_offset as usize, submodule_to_old_module.len());
    debug_assert!(sub_assignment.iter().all(|&m| m != u32::MAX));

    let submodule_data =
        compute_module_data_active(leaf_network, &sub_assignment, module_index_offset, directed);
    let submodule_network =
        leaf_network.consolidate(&sub_assignment, &submodule_data, directed, workspace);

    let level = optimize_active_network(
        &submodule_network,
        rng,
        objective,
        Some(&submodule_to_old_module),
        false,
        CORE_LOOP_LIMIT,
        workspace,
    );

    *top_network =
        submodule_network.consolidate(&level.node_module, &level.module_data, directed, workspace);

    // Keep objective consistent with the current top-level partition for subsequent iterations.
    let refreshed_assignment = assignment_from_top_network(top_network, leaf_count);
    let refreshed_data = compute_module_data_active(
        leaf_network,
        &refreshed_assignment,
        top_network.nodes.len() as u32,
        directed,
    );
    workspace.module_indices.resize(top_network.nodes.len(), 0);
    for i in 0..top_network.nodes.len() {
        workspace.module_indices[i] = i as u32;
    }
    objective.init_partition(
        &refreshed_data,
        &workspace.module_indices[..top_network.nodes.len()],
    );

    let changed = old_assignment != refreshed_assignment;
    if changed {
        level.effective_loops.max(1)
    } else {
        0
    }
}

fn tune_top_modules<'g>(
    leaf_network: &ActiveNetwork<'g>,
    top_network: &mut ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    directed: bool,
    one_level_codelength: f64,
    allow_non_improving_hierarchy_levels: bool,
    workspace: &mut OptimizeWorkspace,
) {
    let mut old_codelength = active_modules_codelength(top_network, objective, workspace);
    let mut do_fine_tune = true;
    let mut coarse_tuned = false;

    while top_network.nodes.len() > 1 {
        if do_fine_tune {
            let num_effective_loops = fine_tune(
                leaf_network,
                top_network,
                objective,
                rng,
                directed,
                workspace,
            );
            if num_effective_loops > 0 {
                *top_network = find_top_modules_repeatedly_from_modules(
                    top_network,
                    objective,
                    rng,
                    directed,
                    allow_non_improving_hierarchy_levels,
                    workspace,
                );
            }
        } else {
            coarse_tuned = true;
            let num_effective_loops = coarse_tune(
                leaf_network,
                top_network,
                objective,
                rng,
                directed,
                allow_non_improving_hierarchy_levels,
                workspace,
            );
            if num_effective_loops > 0 {
                *top_network = find_top_modules_repeatedly_from_modules(
                    top_network,
                    objective,
                    rng,
                    directed,
                    allow_non_improving_hierarchy_levels,
                    workspace,
                );
            }
        }

        let new_codelength = active_modules_codelength(top_network, objective, workspace);
        let is_improvement = new_codelength <= old_codelength - MIN_CODELENGTH_IMPROVEMENT
            && new_codelength
                < old_codelength - one_level_codelength * MIN_RELATIVE_TUNE_ITERATION_IMPROVEMENT;

        if !is_improvement {
            if coarse_tuned {
                break;
            }
        } else {
            old_codelength = new_codelength;
        }

        do_fine_tune = !do_fine_tune;
    }
}

fn optimize_two_level_from_leaf<'g>(
    leaf_network: &ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    directed: bool,
    one_level_codelength: f64,
    lock_first_loop: bool,
    allow_non_improving_hierarchy_levels: bool,
    workspace: &mut OptimizeWorkspace,
) -> ActiveNetwork<'g> {
    let mut top_network = find_top_modules_repeatedly_from_leaf(
        leaf_network,
        objective,
        rng,
        directed,
        lock_first_loop,
        allow_non_improving_hierarchy_levels,
        workspace,
    );
    tune_top_modules(
        leaf_network,
        &mut top_network,
        objective,
        rng,
        directed,
        one_level_codelength,
        allow_non_improving_hierarchy_levels,
        workspace,
    );
    top_network
}

struct SuperHierarchySearch<'g> {
    active_top: ActiveNetwork<'g>,
    // Per accepted level: previous-level module id -> new super-module id.
    super_assignments: Vec<Vec<u32>>,
}

fn find_hierarchical_super_modules_fast<'g>(
    leaf_network: &ActiveNetwork<'g>,
    active_top: &ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    workspace: &mut OptimizeWorkspace,
    directed: bool,
) -> SuperHierarchySearch<'g> {
    let mut active = active_top.clone();
    refresh_active_data_from_leaf_network(leaf_network, &mut active, directed);
    let _ = active_modules_codelength(&active, objective, workspace);
    let mut old_index_length = active_modules_terms(&active, objective, workspace).1;
    let mut num_non_trivial_top_modules = usize::MAX;
    let mut super_assignments = Vec::<Vec<u32>>::new();

    while active.nodes.len() > 1 && num_non_trivial_top_modules > 1 {
        let mut super_active = active.clone();
        transform_node_flow_to_enter_flow(&mut super_active);
        let super_leaf_network = super_active.with_compact_members();
        let mut super_node_data = Vec::with_capacity(super_active.nodes.len());
        for node in &super_leaf_network.nodes {
            super_node_data.push(node.data);
        }
        let mut super_objective = MapEquationObjective::new(&super_node_data);

        let super_one_level_codelength =
            active_modules_codelength(&super_leaf_network, &mut super_objective, workspace);
        let super_top = optimize_two_level_from_leaf(
            &super_leaf_network,
            &mut super_objective,
            rng,
            directed,
            super_one_level_codelength,
            false,
            true,
            workspace,
        );

        let super_assignment =
            assignment_from_top_network(&super_top, super_leaf_network.node_count());
        let num_super_modules = assignment_module_count(&super_assignment);
        let trivial_solution =
            num_super_modules == 1 || num_super_modules == super_leaf_network.nodes.len();
        let (super_codelength, super_index_codelength) =
            active_modules_terms(&super_top, &mut super_objective, workspace);

        if trace_super_enabled() {
            eprintln!(
                "[super] candidates={} super_modules={} codelength={} old_index={} index={}",
                super_active.nodes.len(),
                num_super_modules,
                super_codelength,
                old_index_length,
                super_index_codelength
            );
        }

        if trivial_solution {
            if trace_super_enabled() {
                eprintln!("[super] reject=trivial");
            }
            break;
        }

        // Infomap super-level acceptance compares full two-level super codelength
        // against the previous level's index codelength.
        if super_codelength >= old_index_length - MIN_CODELENGTH_IMPROVEMENT {
            if trace_super_enabled() {
                eprintln!("[super] reject=no_super_improvement");
            }
            break;
        }

        let mut super_module_members = vec![0u32; super_top.nodes.len()];
        for &m in &super_assignment {
            super_module_members[m as usize] += 1;
        }
        num_non_trivial_top_modules = super_module_members.iter().filter(|&&n| n > 1).count();

        let mut super_module_data = Vec::with_capacity(super_top.nodes.len());
        for node in &super_top.nodes {
            super_module_data.push(node.data);
        }
        super_assignments.push(super_assignment.clone());
        active =
            super_active.consolidate(&super_assignment, &super_module_data, directed, workspace);
        refresh_active_data_from_leaf_network(leaf_network, &mut active, directed);
        old_index_length = super_index_codelength;
    }

    SuperHierarchySearch {
        active_top: active,
        super_assignments,
    }
}

fn fine_tune<'g>(
    leaf_network: &ActiveNetwork<'g>,
    top_network: &mut ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
    directed: bool,
    workspace: &mut OptimizeWorkspace,
) -> u32 {
    if top_network.nodes.len() <= 1 {
        return 0;
    }

    let leaf_count = leaf_network.nodes.len();
    let predefined_modules = assignment_from_top_network(top_network, leaf_count);
    let level = optimize_active_network(
        leaf_network,
        rng,
        objective,
        Some(&predefined_modules),
        false,
        CORE_LOOP_LIMIT,
        workspace,
    );

    if level.effective_loops == 0 {
        return 0;
    }

    *top_network =
        leaf_network.consolidate(&level.node_module, &level.module_data, directed, workspace);
    level.effective_loops
}

fn flatten_active_to_assignment(active: &ActiveNetwork<'_>) -> (Vec<u32>, u32) {
    let mut max_member = 0usize;
    for n in &active.nodes {
        for p in n.member_span.start as usize..n.member_span.end as usize {
            max_member = max_member.max(active.member_leaf[p] as usize);
        }
    }
    let mut out = vec![0u32; max_member + 1];
    for (module_idx, n) in active.nodes.iter().enumerate() {
        for p in n.member_span.start as usize..n.member_span.end as usize {
            out[active.member_leaf[p] as usize] = module_idx as u32;
        }
    }
    (out, active.nodes.len() as u32)
}

#[inline]
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    let len = a.len().min(b.len());
    let mut i = 0usize;
    while i < len && a[i] == b[i] {
        i += 1;
    }
    i
}

fn compute_hierarchy_module_data(
    graph: &Graph,
    leaf_paths: &[Vec<u32>],
    module_count: usize,
    directed: bool,
) -> Vec<FlowData> {
    let mut module_data = vec![FlowData::default(); module_count];

    for (leaf, path) in leaf_paths.iter().enumerate() {
        let flow = graph.node_data[leaf].flow;
        for &m in path {
            module_data[m as usize].flow += flow;
        }
    }

    for e in 0..graph.edge_count() {
        let s = graph.edge_source[e] as usize;
        let t = graph.edge_target[e] as usize;
        if s == t {
            continue;
        }
        let ps = &leaf_paths[s];
        let pt = &leaf_paths[t];
        let lcp = common_prefix_len(ps, pt);
        let f = graph.edge_flow[e];
        if directed {
            for &m in &ps[lcp..] {
                module_data[m as usize].exit_flow += f;
            }
            for &m in &pt[lcp..] {
                module_data[m as usize].enter_flow += f;
            }
        } else {
            let h = f / 2.0;
            for &m in &ps[lcp..] {
                let md = &mut module_data[m as usize];
                md.exit_flow += h;
                md.enter_flow += h;
            }
            for &m in &pt[lcp..] {
                let md = &mut module_data[m as usize];
                md.exit_flow += h;
                md.enter_flow += h;
            }
        }
    }

    module_data
}

fn compute_hierarchy_module_data_active(
    active: &ActiveNetwork<'_>,
    leaf_paths: &[Vec<u32>],
    module_count: usize,
    directed: bool,
) -> Vec<FlowData> {
    match &active.edge_storage {
        EdgeStorage::Borrowed {
            edge_source,
            edge_target,
            edge_flow,
            in_edge_idx,
        } => {
            let adj = BorrowedAdj {
                edge_source,
                edge_target,
                edge_flow,
                in_edge_idx,
            };
            compute_hierarchy_module_data_active_with_adj(
                active,
                &adj,
                leaf_paths,
                module_count,
                directed,
            )
        }
        EdgeStorage::Owned {
            out_neighbor,
            out_flow,
            in_neighbor,
            in_flow,
        } => {
            let adj = OwnedAdj {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            };
            compute_hierarchy_module_data_active_with_adj(
                active,
                &adj,
                leaf_paths,
                module_count,
                directed,
            )
        }
    }
}

fn compute_hierarchy_module_data_active_with_adj<A: AdjAccess>(
    active: &ActiveNetwork<'_>,
    adj: &A,
    leaf_paths: &[Vec<u32>],
    module_count: usize,
    directed: bool,
) -> Vec<FlowData> {
    let mut module_data = vec![FlowData::default(); module_count];

    for (leaf, path) in leaf_paths.iter().enumerate() {
        let flow = active.nodes[leaf].data.flow;
        for &m in path {
            module_data[m as usize].flow += flow;
        }
    }

    for s in 0..active.node_count() {
        for e in active.out_range(s) {
            let t = adj.out_neighbor(e) as usize;
            if s == t {
                continue;
            }
            let ps = &leaf_paths[s];
            let pt = &leaf_paths[t];
            let lcp = common_prefix_len(ps, pt);
            let f = adj.out_flow(e);
            if directed {
                for &m in &ps[lcp..] {
                    module_data[m as usize].exit_flow += f;
                }
                for &m in &pt[lcp..] {
                    module_data[m as usize].enter_flow += f;
                }
            } else {
                let h = f / 2.0;
                for &m in &ps[lcp..] {
                    let md = &mut module_data[m as usize];
                    md.exit_flow += h;
                    md.enter_flow += h;
                }
                for &m in &pt[lcp..] {
                    let md = &mut module_data[m as usize];
                    md.exit_flow += h;
                    md.enter_flow += h;
                }
            }
        }
    }

    module_data
}

fn fill_leaf_paths_from_dynamic(
    hierarchy: &DynamicHierarchy,
    module_idx: u32,
    path: &mut Vec<u32>,
    leaf_paths: &mut [Vec<u32>],
) {
    path.push(module_idx);
    let leaf_base = hierarchy.leaf_base();
    for &child in hierarchy.children_slice(module_idx).iter() {
        if child < leaf_base {
            leaf_paths[child as usize] = path.clone();
        } else {
            let child_module = child - leaf_base;
            fill_leaf_paths_from_dynamic(hierarchy, child_module, path, leaf_paths);
        }
    }
    let _ = path.pop();
}

fn dynamic_hierarchy_to_result_ref(
    graph: &Graph,
    hierarchy: &DynamicHierarchy,
    directed: bool,
) -> HierarchyResult {
    let module_count = hierarchy.module_count();
    let mut module_children_offsets = Vec::with_capacity(module_count + 1);
    module_children_offsets.push(0);
    let mut module_children = Vec::<u32>::new();
    for m in 0..module_count {
        let range = hierarchy.child_range(m as u32);
        module_children.extend_from_slice(&hierarchy.children[range]);
        module_children_offsets.push(module_children.len() as u32);
    }

    let mut leaf_paths = vec![Vec::<u32>::new(); hierarchy.leaf_count];
    let mut path = Vec::<u32>::new();
    for &top in &hierarchy.top_modules {
        fill_leaf_paths_from_dynamic(hierarchy, top, &mut path, &mut leaf_paths);
    }

    let mut max_depth = 0usize;
    for p in &leaf_paths {
        max_depth = max_depth.max(p.len());
    }
    let levels = (max_depth as u32).saturating_add(1);
    let module_data = compute_hierarchy_module_data(graph, &leaf_paths, module_count, directed);

    HierarchyResult {
        leaf_paths,
        top_modules: hierarchy.top_modules.clone(),
        levels,
        module_parent: hierarchy.module_parent.clone(),
        module_children_offsets,
        module_children,
        module_data,
    }
}

fn dynamic_hierarchy_to_result(
    graph: &Graph,
    hierarchy: DynamicHierarchy,
    directed: bool,
) -> HierarchyResult {
    dynamic_hierarchy_to_result_ref(graph, &hierarchy, directed)
}

fn hierarchy_codelength(graph: &Graph, hier: &HierarchyResult) -> f64 {
    let leaf_base = graph.node_count() as u32;
    let mut root_total = 0.0f64;
    let mut root_child_log = 0.0f64;

    for &top in &hier.top_modules {
        let w = hier.module_data[top as usize].enter_flow;
        root_total += w;
        root_child_log += plogp(w);
    }
    for leaf in 0..graph.node_count() {
        if hier.leaf_paths[leaf].is_empty() {
            let w = graph.node_data[leaf].flow;
            root_total += w;
            root_child_log += plogp(w);
        }
    }

    let mut codelength = plogp(root_total) - root_child_log;

    for m in 0..hier.module_parent.len() {
        let mut child_total = 0.0f64;
        let mut child_log = 0.0f64;
        let range =
            hier.module_children_offsets[m] as usize..hier.module_children_offsets[m + 1] as usize;
        for &child in hier.module_children[range].iter() {
            let w = if child < leaf_base {
                graph.node_data[child as usize].flow
            } else {
                hier.module_data[(child - leaf_base) as usize].enter_flow
            };
            child_total += w;
            child_log += plogp(w);
        }
        let exit = hier.module_data[m].exit_flow;
        codelength += plogp(exit + child_total) - plogp(exit) - child_log;
    }

    codelength
}

fn module_term_codelength_graph(graph: &Graph, hier: &HierarchyResult, module_idx: usize) -> f64 {
    let leaf_base = graph.node_count() as u32;
    let mut child_total = 0.0f64;
    let mut child_log = 0.0f64;
    let range = hier.module_children_offsets[module_idx] as usize
        ..hier.module_children_offsets[module_idx + 1] as usize;
    for &child in hier.module_children[range].iter() {
        let w = if child < leaf_base {
            graph.node_data[child as usize].flow
        } else {
            hier.module_data[(child - leaf_base) as usize].enter_flow
        };
        child_total += w;
        child_log += plogp(w);
    }
    let exit = hier.module_data[module_idx].exit_flow;
    plogp(exit + child_total) - plogp(exit) - child_log
}

#[inline]
fn split_term_validation_limit() -> usize {
    if std::env::var_os("MINIMAP_VALIDATE_SPLIT_TERM").is_none() {
        return 0;
    }
    if let Some(v) = std::env::var_os("MINIMAP_VALIDATE_SPLIT_TERM_N") {
        if let Ok(s) = v.into_string() {
            if let Ok(n) = s.parse::<usize>() {
                return n.max(1);
            }
        }
    }
    16
}

fn local_module_term_codelength_active(
    leaf_network: &ActiveNetwork<'_>,
    parent_members: &[u32],
    local_hierarchy: &HierarchyResult,
    local_module_data: &[FlowData],
    local_module_idx: usize,
) -> f64 {
    let local_leaf_base = parent_members.len() as u32;
    let mut child_total = 0.0f64;
    let mut child_log = 0.0f64;
    let range = local_hierarchy.module_children_offsets[local_module_idx] as usize
        ..local_hierarchy.module_children_offsets[local_module_idx + 1] as usize;
    for &child in local_hierarchy.module_children[range].iter() {
        let w = if child < local_leaf_base {
            let global_leaf = parent_members[child as usize] as usize;
            leaf_network.nodes[global_leaf].data.flow
        } else {
            local_module_data[(child - local_leaf_base) as usize].enter_flow
        };
        child_total += w;
        child_log += plogp(w);
    }
    let exit = local_module_data[local_module_idx].exit_flow;
    plogp(exit + child_total) - plogp(exit) - child_log
}

fn candidate_split_local_codelength(
    leaf_network: &ActiveNetwork<'_>,
    parent_members: &[u32],
    sub_hierarchy: &HierarchyResult,
    parent_exit_flow: f64,
    directed: bool,
    workspace: &mut OptimizeWorkspace,
) -> f64 {
    match &leaf_network.edge_storage {
        EdgeStorage::Borrowed {
            edge_source,
            edge_target,
            edge_flow,
            in_edge_idx,
        } => {
            let adj = BorrowedAdj {
                edge_source,
                edge_target,
                edge_flow,
                in_edge_idx,
            };
            candidate_split_local_codelength_with_adj(
                leaf_network,
                parent_members,
                sub_hierarchy,
                parent_exit_flow,
                directed,
                workspace,
                &adj,
            )
        }
        EdgeStorage::Owned {
            out_neighbor,
            out_flow,
            in_neighbor,
            in_flow,
        } => {
            let adj = OwnedAdj {
                out_neighbor,
                out_flow,
                in_neighbor,
                in_flow,
            };
            candidate_split_local_codelength_with_adj(
                leaf_network,
                parent_members,
                sub_hierarchy,
                parent_exit_flow,
                directed,
                workspace,
                &adj,
            )
        }
    }
}

fn candidate_split_local_codelength_with_adj<A: AdjAccess>(
    leaf_network: &ActiveNetwork<'_>,
    parent_members: &[u32],
    sub_hierarchy: &HierarchyResult,
    parent_exit_flow: f64,
    directed: bool,
    workspace: &mut OptimizeWorkspace,
    adj: &A,
) -> f64 {
    let local_module_count = sub_hierarchy.module_parent.len();
    if local_module_count == 0 {
        return f64::INFINITY;
    }

    let OptimizeWorkspace {
        recurse_sub_global_to_local,
        recurse_sub_touched_global,
        recurse_local_module_data,
        ..
    } = workspace;

    if recurse_sub_global_to_local.len() != leaf_network.node_count() {
        recurse_sub_global_to_local.resize(leaf_network.node_count(), u32::MAX);
    }
    recurse_sub_touched_global.clear();
    for (local_leaf, &global_leaf_u32) in parent_members.iter().enumerate() {
        recurse_sub_global_to_local[global_leaf_u32 as usize] = local_leaf as u32;
        recurse_sub_touched_global.push(global_leaf_u32);
    }

    recurse_local_module_data.clear();
    recurse_local_module_data.extend_from_slice(&sub_hierarchy.module_data);
    for d in recurse_local_module_data.iter_mut() {
        d.enter_flow = 0.0;
        d.exit_flow = 0.0;
    }

    for (local_s, &global_s_u32) in parent_members.iter().enumerate() {
        let ps = &sub_hierarchy.leaf_paths[local_s];
        debug_assert!(
            !ps.is_empty(),
            "leaf path unexpectedly empty for local source leaf {}",
            local_s
        );

        let source = global_s_u32 as usize;
        for e in leaf_network.out_range(source) {
            let target = adj.out_neighbor(e) as usize;
            let local_t = recurse_sub_global_to_local[target];
            let f = adj.out_flow(e);

            if local_t != u32::MAX {
                let pt = &sub_hierarchy.leaf_paths[local_t as usize];
                let lcp = common_prefix_len(ps, pt);
                if directed {
                    for &m in &ps[lcp..] {
                        recurse_local_module_data[m as usize].exit_flow += f;
                    }
                    for &m in &pt[lcp..] {
                        recurse_local_module_data[m as usize].enter_flow += f;
                    }
                } else {
                    let h = f / 2.0;
                    for &m in &ps[lcp..] {
                        let md = &mut recurse_local_module_data[m as usize];
                        md.exit_flow += h;
                        md.enter_flow += h;
                    }
                    for &m in &pt[lcp..] {
                        let md = &mut recurse_local_module_data[m as usize];
                        md.exit_flow += h;
                        md.enter_flow += h;
                    }
                }
            } else if directed {
                for &m in ps {
                    recurse_local_module_data[m as usize].exit_flow += f;
                }
            } else {
                let h = f / 2.0;
                for &m in ps {
                    let md = &mut recurse_local_module_data[m as usize];
                    md.exit_flow += h;
                    md.enter_flow += h;
                }
            }
        }
    }

    for (local_t, &global_t_u32) in parent_members.iter().enumerate() {
        let pt = &sub_hierarchy.leaf_paths[local_t];
        debug_assert!(
            !pt.is_empty(),
            "leaf path unexpectedly empty for local target leaf {}",
            local_t
        );

        let target = global_t_u32 as usize;
        for e in leaf_network.in_range(target) {
            let source = adj.in_neighbor(e) as usize;
            if recurse_sub_global_to_local[source] != u32::MAX {
                continue;
            }
            let f = adj.in_flow(e);
            if directed {
                for &m in pt {
                    recurse_local_module_data[m as usize].enter_flow += f;
                }
            } else {
                let h = f / 2.0;
                for &m in pt {
                    let md = &mut recurse_local_module_data[m as usize];
                    md.exit_flow += h;
                    md.enter_flow += h;
                }
            }
        }
    }

    for &global_leaf_u32 in recurse_sub_touched_global.iter() {
        recurse_sub_global_to_local[global_leaf_u32 as usize] = u32::MAX;
    }
    recurse_sub_touched_global.clear();

    let mut parent_child_total = 0.0f64;
    let mut parent_child_log = 0.0f64;
    for &top in sub_hierarchy.top_modules.iter() {
        let w = recurse_local_module_data[top as usize].enter_flow;
        parent_child_total += w;
        parent_child_log += plogp(w);
    }
    let parent_term =
        plogp(parent_exit_flow + parent_child_total) - plogp(parent_exit_flow) - parent_child_log;

    let mut inserted_terms = 0.0f64;
    for local_module_idx in 0..local_module_count {
        inserted_terms += local_module_term_codelength_active(
            leaf_network,
            parent_members,
            sub_hierarchy,
            recurse_local_module_data,
            local_module_idx,
        );
    }

    parent_term + inserted_terms
}

fn validate_fast_split_local_codelength(
    graph: &Graph,
    hierarchy: &DynamicHierarchy,
    module_idx: u32,
    sub_hierarchy: &HierarchyResult,
    parent_members: &[u32],
    directed: bool,
    fast_local_codelength: f64,
) {
    let old_module_count = hierarchy.module_count();
    let mut candidate_hierarchy = hierarchy.clone();
    let mut next_level_leaf_modules = Vec::<u32>::new();
    graft_local_hierarchy_under_module(
        &mut candidate_hierarchy,
        module_idx,
        sub_hierarchy,
        parent_members,
        &mut next_level_leaf_modules,
    );
    let candidate_result = dynamic_hierarchy_to_result_ref(graph, &candidate_hierarchy, directed);
    if module_idx as usize >= candidate_result.module_parent.len() {
        panic!(
            "split term validator failed: module {} missing after graft",
            module_idx
        );
    }

    let mut slow_local_codelength =
        module_term_codelength_graph(graph, &candidate_result, module_idx as usize);
    for added_module_idx in old_module_count..candidate_hierarchy.module_count() {
        slow_local_codelength +=
            module_term_codelength_graph(graph, &candidate_result, added_module_idx);
    }

    let diff = (slow_local_codelength - fast_local_codelength).abs();
    if diff > 1e-9 {
        panic!(
            "split local codelength validator mismatch for module {}: fast={} slow={} diff={}",
            module_idx, fast_local_codelength, slow_local_codelength, diff
        );
    }
}

fn build_hierarchy_from_layers(
    graph: &Graph,
    layers: &[Vec<u32>],
    directed: bool,
) -> HierarchyResult {
    let leaf_count = graph.node_count();
    if layers.is_empty() {
        return HierarchyResult {
            leaf_paths: vec![Vec::new(); leaf_count],
            top_modules: Vec::new(),
            levels: 1,
            module_parent: Vec::new(),
            module_children_offsets: vec![0],
            module_children: Vec::new(),
            module_data: Vec::new(),
        };
    }

    let depth_count = layers.len();
    debug_assert_eq!(layers[0].len(), leaf_count);

    let mut module_counts_bottom = Vec::<usize>::with_capacity(depth_count);
    for (k, layer) in layers.iter().enumerate() {
        let mut max_id = 0u32;
        for &m in layer {
            max_id = max_id.max(m);
        }
        let count = max_id as usize + 1;
        if k > 0 {
            debug_assert_eq!(layers[k].len(), module_counts_bottom[k - 1]);
        }
        module_counts_bottom.push(count);
    }

    // Map (bottom depth, module id) -> local module id in a top-down contiguous numbering.
    let mut local_of_bottom = vec![Vec::<u32>::new(); depth_count];
    let mut next_local = 0u32;
    for top_depth in 0..depth_count {
        let bottom_k = depth_count - 1 - top_depth;
        let count = module_counts_bottom[bottom_k];
        let mut map = vec![u32::MAX; count];
        for id in 0..count {
            map[id] = next_local;
            next_local += 1;
        }
        local_of_bottom[bottom_k] = map;
    }

    let module_count = next_local as usize;
    let leaf_base = leaf_count as u32;
    let mut module_parent = vec![u32::MAX; module_count];
    let mut module_children_raw: Vec<Vec<u32>> = vec![Vec::new(); module_count];

    // Parent links from bottom (nearest leaves) up to top.
    for bottom_k in 0..depth_count.saturating_sub(1) {
        let count = module_counts_bottom[bottom_k];
        for id in 0..count {
            let local = local_of_bottom[bottom_k][id] as usize;
            let parent_id = layers[bottom_k + 1][id] as usize;
            module_parent[local] = local_of_bottom[bottom_k + 1][parent_id];
        }
    }

    // Children for modules nearest leaves.
    for leaf in 0..leaf_count {
        let bottom_id = layers[0][leaf] as usize;
        let local = local_of_bottom[0][bottom_id] as usize;
        module_children_raw[local].push(leaf as u32);
    }

    // Children for all higher module levels.
    for bottom_k in 1..depth_count {
        let lower_count = module_counts_bottom[bottom_k - 1];
        for child_id in 0..lower_count {
            let parent_id = layers[bottom_k][child_id] as usize;
            let child_local = local_of_bottom[bottom_k - 1][child_id];
            let parent_local = local_of_bottom[bottom_k][parent_id] as usize;
            module_children_raw[parent_local].push(leaf_base + child_local);
        }
    }

    let top_k = depth_count - 1;
    let mut top_modules = Vec::<u32>::with_capacity(module_counts_bottom[top_k]);
    for id in 0..module_counts_bottom[top_k] {
        top_modules.push(local_of_bottom[top_k][id]);
    }

    let mut module_children_offsets = Vec::with_capacity(module_count + 1);
    module_children_offsets.push(0);
    let mut module_children = Vec::<u32>::new();
    for children in &module_children_raw {
        module_children.extend_from_slice(children);
        module_children_offsets.push(module_children.len() as u32);
    }

    let mut leaf_paths = vec![Vec::<u32>::new(); leaf_count];
    for leaf in 0..leaf_count {
        let mut bottom_ids = vec![0u32; depth_count];
        bottom_ids[0] = layers[0][leaf];
        for k in 1..depth_count {
            bottom_ids[k] = layers[k][bottom_ids[k - 1] as usize];
        }

        let mut path = Vec::<u32>::with_capacity(depth_count);
        for top_depth in 0..depth_count {
            let bottom_k = depth_count - 1 - top_depth;
            let local = local_of_bottom[bottom_k][bottom_ids[bottom_k] as usize];
            path.push(local);
        }
        leaf_paths[leaf] = path;
    }

    let levels = depth_count as u32 + 1;
    let mut module_data = vec![FlowData::default(); module_count];

    for (leaf, path) in leaf_paths.iter().enumerate() {
        let flow = graph.node_data[leaf].flow;
        for &m in path {
            module_data[m as usize].flow += flow;
        }
    }

    for e in 0..graph.edge_count() {
        let s = graph.edge_source[e] as usize;
        let t = graph.edge_target[e] as usize;
        if s == t {
            continue;
        }
        let ps = &leaf_paths[s];
        let pt = &leaf_paths[t];
        let lcp = common_prefix_len(ps, pt);
        let f = graph.edge_flow[e];
        if directed {
            for &m in &ps[lcp..] {
                module_data[m as usize].exit_flow += f;
            }
            for &m in &pt[lcp..] {
                module_data[m as usize].enter_flow += f;
            }
        } else {
            let h = f / 2.0;
            for &m in &ps[lcp..] {
                let md = &mut module_data[m as usize];
                md.exit_flow += h;
                md.enter_flow += h;
            }
            for &m in &pt[lcp..] {
                let md = &mut module_data[m as usize];
                md.exit_flow += h;
                md.enter_flow += h;
            }
        }
    }

    HierarchyResult {
        leaf_paths,
        top_modules,
        levels,
        module_parent,
        module_children_offsets,
        module_children,
        module_data,
    }
}

fn build_hierarchy_from_layers_active(
    active: &ActiveNetwork<'_>,
    layers: &[Vec<u32>],
    directed: bool,
) -> HierarchyResult {
    let leaf_count = active.node_count();
    if layers.is_empty() {
        return HierarchyResult {
            leaf_paths: vec![Vec::new(); leaf_count],
            top_modules: Vec::new(),
            levels: 1,
            module_parent: Vec::new(),
            module_children_offsets: vec![0],
            module_children: Vec::new(),
            module_data: Vec::new(),
        };
    }

    let depth_count = layers.len();
    debug_assert_eq!(layers[0].len(), leaf_count);

    let mut module_counts_bottom = Vec::<usize>::with_capacity(depth_count);
    for (k, layer) in layers.iter().enumerate() {
        let mut max_id = 0u32;
        for &m in layer {
            max_id = max_id.max(m);
        }
        let count = max_id as usize + 1;
        if k > 0 {
            debug_assert_eq!(layers[k].len(), module_counts_bottom[k - 1]);
        }
        module_counts_bottom.push(count);
    }

    let mut local_of_bottom = vec![Vec::<u32>::new(); depth_count];
    let mut next_local = 0u32;
    for top_depth in 0..depth_count {
        let bottom_k = depth_count - 1 - top_depth;
        let count = module_counts_bottom[bottom_k];
        let mut map = vec![u32::MAX; count];
        for id in 0..count {
            map[id] = next_local;
            next_local += 1;
        }
        local_of_bottom[bottom_k] = map;
    }

    let module_count = next_local as usize;
    let leaf_base = leaf_count as u32;
    let mut module_parent = vec![u32::MAX; module_count];
    let mut module_children_raw: Vec<Vec<u32>> = vec![Vec::new(); module_count];

    for bottom_k in 0..depth_count.saturating_sub(1) {
        let count = module_counts_bottom[bottom_k];
        for id in 0..count {
            let local = local_of_bottom[bottom_k][id] as usize;
            let parent_id = layers[bottom_k + 1][id] as usize;
            module_parent[local] = local_of_bottom[bottom_k + 1][parent_id];
        }
    }

    for leaf in 0..leaf_count {
        let bottom_id = layers[0][leaf] as usize;
        let local = local_of_bottom[0][bottom_id] as usize;
        module_children_raw[local].push(leaf as u32);
    }

    for bottom_k in 1..depth_count {
        let lower_count = module_counts_bottom[bottom_k - 1];
        for child_id in 0..lower_count {
            let parent_id = layers[bottom_k][child_id] as usize;
            let child_local = local_of_bottom[bottom_k - 1][child_id];
            let parent_local = local_of_bottom[bottom_k][parent_id] as usize;
            module_children_raw[parent_local].push(leaf_base + child_local);
        }
    }

    let top_k = depth_count - 1;
    let mut top_modules = Vec::<u32>::with_capacity(module_counts_bottom[top_k]);
    for id in 0..module_counts_bottom[top_k] {
        top_modules.push(local_of_bottom[top_k][id]);
    }

    let mut module_children_offsets = Vec::with_capacity(module_count + 1);
    module_children_offsets.push(0);
    let mut module_children = Vec::<u32>::new();
    for children in &module_children_raw {
        module_children.extend_from_slice(children);
        module_children_offsets.push(module_children.len() as u32);
    }

    let mut leaf_paths = vec![Vec::<u32>::new(); leaf_count];
    for leaf in 0..leaf_count {
        let mut bottom_ids = vec![0u32; depth_count];
        bottom_ids[0] = layers[0][leaf];
        for k in 1..depth_count {
            bottom_ids[k] = layers[k][bottom_ids[k - 1] as usize];
        }

        let mut path = Vec::<u32>::with_capacity(depth_count);
        for top_depth in 0..depth_count {
            let bottom_k = depth_count - 1 - top_depth;
            let local = local_of_bottom[bottom_k][bottom_ids[bottom_k] as usize];
            path.push(local);
        }
        leaf_paths[leaf] = path;
    }

    let levels = depth_count as u32 + 1;
    let module_data =
        compute_hierarchy_module_data_active(active, &leaf_paths, module_count, directed);

    HierarchyResult {
        leaf_paths,
        top_modules,
        levels,
        module_parent,
        module_children_offsets,
        module_children,
        module_data,
    }
}

fn compute_module_data(
    graph: &Graph,
    node_to_module: &[u32],
    num_modules: u32,
    directed: bool,
) -> Vec<FlowData> {
    let mut modules = vec![FlowData::default(); num_modules as usize];

    for (i, node) in graph.node_data.iter().enumerate() {
        let m = node_to_module[i] as usize;
        modules[m].flow += node.flow;
    }

    for e in 0..graph.edge_count() {
        let s = graph.edge_source[e] as usize;
        let t = graph.edge_target[e] as usize;
        let ms = node_to_module[s] as usize;
        let mt = node_to_module[t] as usize;
        if ms == mt {
            continue;
        }
        let f = graph.edge_flow[e];
        if directed {
            modules[ms].exit_flow += f;
            modules[mt].enter_flow += f;
        } else {
            let h = f / 2.0;
            modules[ms].exit_flow += h;
            modules[ms].enter_flow += h;
            modules[mt].exit_flow += h;
            modules[mt].enter_flow += h;
        }
    }

    modules
}

fn one_level_codelength(graph: &Graph) -> f64 {
    let mut sum = 0.0;
    for node in &graph.node_data {
        sum -= plogp(node.flow);
    }
    sum
}

fn active_one_level_codelength(active: &ActiveNetwork<'_>) -> f64 {
    let mut sum = 0.0;
    for node in &active.nodes {
        sum -= plogp(node.data.flow);
    }
    sum
}

#[inline]
fn is_leaf_module_dynamic(hierarchy: &DynamicHierarchy, module_idx: u32) -> bool {
    let leaf_base = hierarchy.leaf_base();
    hierarchy
        .children_slice(module_idx)
        .iter()
        .all(|&child| child < leaf_base)
}

fn copy_local_module_into_dynamic(
    hierarchy: &mut DynamicHierarchy,
    parent_global: u32,
    local_module_idx: u32,
    local_hier: &HierarchyResult,
    local_leaf_to_global_leaf: &[u32],
    next_level_leaf_modules: &mut Vec<u32>,
) -> u32 {
    let global_module = hierarchy.add_module_with_children(parent_global, &[]);
    let local_leaf_base = local_leaf_to_global_leaf.len() as u32;
    let range = local_hier.module_children_offsets[local_module_idx as usize] as usize
        ..local_hier.module_children_offsets[local_module_idx as usize + 1] as usize;

    let mut children = Vec::<u32>::with_capacity(range.len());
    let mut all_children_are_leaves = true;
    for &child in local_hier.module_children[range].iter() {
        if child < local_leaf_base {
            children.push(local_leaf_to_global_leaf[child as usize]);
        } else {
            all_children_are_leaves = false;
            let local_child_module = child - local_leaf_base;
            let global_child_module = copy_local_module_into_dynamic(
                hierarchy,
                global_module,
                local_child_module,
                local_hier,
                local_leaf_to_global_leaf,
                next_level_leaf_modules,
            );
            children.push(hierarchy.leaf_base() + global_child_module);
        }
    }

    hierarchy.set_module_children(global_module, &children);
    if all_children_are_leaves && children.len() > 1 {
        next_level_leaf_modules.push(global_module);
    }

    global_module
}

fn graft_local_hierarchy_under_module(
    hierarchy: &mut DynamicHierarchy,
    parent_module: u32,
    local_hier: &HierarchyResult,
    local_leaf_to_global_leaf: &[u32],
    next_level_leaf_modules: &mut Vec<u32>,
) {
    next_level_leaf_modules.clear();
    let mut new_children = Vec::<u32>::with_capacity(local_hier.top_modules.len());
    let leaf_base = hierarchy.leaf_base();

    for &local_top in local_hier.top_modules.iter() {
        let global_top_module = copy_local_module_into_dynamic(
            hierarchy,
            parent_module,
            local_top,
            local_hier,
            local_leaf_to_global_leaf,
            next_level_leaf_modules,
        );
        new_children.push(leaf_base + global_top_module);
    }

    hierarchy.set_module_children(parent_module, &new_children);
}

fn recursive_partition_bottom_modules<'g>(
    graph: &Graph,
    leaf_network: &ActiveNetwork<'g>,
    hierarchy: &mut DynamicHierarchy,
    rng: &mut impl TrialRng,
    directed: bool,
    workspace: &mut OptimizeWorkspace,
) {
    workspace.recurse_queue.clear();
    for module_idx in 0..hierarchy.module_count() as u32 {
        if is_leaf_module_dynamic(hierarchy, module_idx) && hierarchy.child_count(module_idx) > 1 {
            workspace.recurse_queue.push(module_idx);
        }
    }

    let mut validate_remaining = split_term_validation_limit();

    while !workspace.recurse_queue.is_empty() {
        // Keep one stable snapshot per queue level, matching Infomap's queue processing semantics.
        let snapshot = dynamic_hierarchy_to_result_ref(graph, hierarchy, directed);
        workspace.recurse_next_queue.clear();
        let queue_len = workspace.recurse_queue.len();

        for i in 0..queue_len {
            let module_idx = workspace.recurse_queue[i];
            if module_idx as usize >= hierarchy.module_count() {
                continue;
            }
            if !is_leaf_module_dynamic(hierarchy, module_idx) {
                continue;
            }

            workspace.recurse_module_children.clear();
            workspace
                .recurse_module_children
                .extend_from_slice(hierarchy.children_slice(module_idx));
            if workspace.recurse_module_children.len() <= 2 {
                continue;
            }
            let parent_members = workspace.recurse_module_children.clone();

            if module_idx as usize >= snapshot.module_parent.len() {
                continue;
            }
            let old_module_codelength =
                module_term_codelength_graph(graph, &snapshot, module_idx as usize);

            let sub_active = induced_subnetwork(leaf_network, &parent_members, workspace);
            if sub_active.node_count() <= 2 {
                continue;
            }

            let mut sub_node_data = Vec::with_capacity(sub_active.nodes.len());
            for node in &sub_active.nodes {
                sub_node_data.push(node.data);
            }
            let mut sub_objective = MapEquationObjective::new(&sub_node_data);
            let sub_one_level = active_one_level_codelength(&sub_active);
            let sub_top = optimize_two_level_from_leaf(
                &sub_active,
                &mut sub_objective,
                rng,
                directed,
                sub_one_level,
                true,
                true,
                workspace,
            );
            let base_assignment = assignment_from_top_network(&sub_top, sub_active.node_count());

            let super_result = find_hierarchical_super_modules_fast(
                &sub_active,
                &sub_top,
                &mut sub_objective,
                rng,
                workspace,
                directed,
            );
            let super_assignments = super_result.super_assignments;

            workspace.recurse_layers.clear();
            workspace.recurse_layers.push(base_assignment);
            workspace.recurse_layers.extend(super_assignments);

            let sub_hierarchy = build_hierarchy_from_layers_active(
                &sub_active,
                &workspace.recurse_layers,
                directed,
            );
            let num_submodules = sub_hierarchy.top_modules.len();
            let trivial_sub_partition =
                num_submodules <= 1 || num_submodules >= parent_members.len();
            if trivial_sub_partition {
                continue;
            }

            // Exact local split scoring in full graph context without cloning/materializing hierarchy.
            let parent_exit_flow = snapshot.module_data[module_idx as usize].exit_flow;
            let new_local_codelength = candidate_split_local_codelength(
                leaf_network,
                &parent_members,
                &sub_hierarchy,
                parent_exit_flow,
                directed,
                workspace,
            );
            if validate_remaining > 0 {
                validate_fast_split_local_codelength(
                    graph,
                    hierarchy,
                    module_idx,
                    &sub_hierarchy,
                    &parent_members,
                    directed,
                    new_local_codelength,
                );
                validate_remaining -= 1;
            }
            if new_local_codelength >= old_module_codelength - MIN_CODELENGTH_IMPROVEMENT {
                continue;
            }

            workspace.recurse_next_level_leaf_modules.clear();
            graft_local_hierarchy_under_module(
                hierarchy,
                module_idx,
                &sub_hierarchy,
                &parent_members,
                &mut workspace.recurse_next_level_leaf_modules,
            );
            workspace
                .recurse_next_queue
                .extend(workspace.recurse_next_level_leaf_modules.iter().copied());
        }

        std::mem::swap(
            &mut workspace.recurse_queue,
            &mut workspace.recurse_next_queue,
        );
    }
}

fn single_trial(
    graph: &Graph,
    rng: &mut impl TrialRng,
    directed: bool,
    multilevel: bool,
) -> TrialResult {
    let node_data = graph.node_data.clone();
    let mut objective = MapEquationObjective::new(&node_data);
    let mut workspace = OptimizeWorkspace::default();

    let leaf_network = ActiveNetwork::from_graph(graph);
    let one_level = one_level_codelength(graph);
    let mut top_network = optimize_two_level_from_leaf(
        &leaf_network,
        &mut objective,
        rng,
        directed,
        one_level,
        true,
        multilevel,
        &mut workspace,
    );

    let base_assignment = if multilevel {
        Some(assignment_from_top_network(
            &top_network,
            graph.node_count(),
        ))
    } else {
        None
    };
    let mut super_assignments: Vec<Vec<u32>> = Vec::new();

    if multilevel {
        let super_result = find_hierarchical_super_modules_fast(
            &leaf_network,
            &top_network,
            &mut objective,
            rng,
            &mut workspace,
            directed,
        );
        super_assignments = super_result.super_assignments;
        top_network = super_result.active_top;
    }

    let (node_to_module, num_modules) = flatten_active_to_assignment(&top_network);
    let module_data = compute_module_data(graph, &node_to_module, num_modules, directed);
    let hierarchy = if multilevel {
        let mut layers = Vec::<Vec<u32>>::new();
        if let Some(base) = base_assignment {
            layers.push(base);
            layers.extend(super_assignments);
        }
        let base_hierarchy = build_hierarchy_from_layers(graph, &layers, directed);
        let base_codelength = hierarchy_codelength(graph, &base_hierarchy);
        let mut dynamic_hierarchy =
            DynamicHierarchy::from_result(&base_hierarchy, graph.node_count());
        recursive_partition_bottom_modules(
            graph,
            &leaf_network,
            &mut dynamic_hierarchy,
            rng,
            directed,
            &mut workspace,
        );
        let refined_hierarchy = dynamic_hierarchy_to_result(graph, dynamic_hierarchy, directed);
        let refined_codelength = hierarchy_codelength(graph, &refined_hierarchy);
        if refined_codelength < base_codelength - MIN_CODELENGTH_IMPROVEMENT {
            Some(refined_hierarchy)
        } else {
            Some(base_hierarchy)
        }
    } else {
        None
    };
    let levels = hierarchy.as_ref().map_or(2, |h| h.levels.max(2));

    let codelength = if let Some(hier) = hierarchy.as_ref() {
        hierarchy_codelength(graph, hier)
    } else {
        let mut final_obj = MapEquationObjective::new(&node_data);
        let module_indices: Vec<u32> = (0..num_modules).collect();
        final_obj.init_partition(&module_data, &module_indices);
        final_obj.codelength
    };

    TrialResult {
        node_to_module,
        num_modules,
        levels,
        codelength,
        one_level_codelength: one_level,
        module_data,
        hierarchy,
    }
}

#[inline]
fn seed_for_trial(base_seed: u32, trial_index: u32) -> u32 {
    if trial_index == 0 {
        return base_seed;
    }

    // SplitMix64-style mixing for deterministic independent trial seeds.
    let mut z = (base_seed as u64) ^ ((trial_index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

#[inline]
fn trace_threads_enabled() -> bool {
    std::env::var_os("MINIMAP_TRACE_THREADS").is_some()
}

#[inline]
fn env_trial_threads() -> Option<usize> {
    if let Some(v) = std::env::var_os("MINIMAP_TRIAL_THREADS") {
        if let Ok(s) = v.into_string() {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }
    }

    if let Some(v) = std::env::var_os("RAYON_NUM_THREADS") {
        if let Ok(s) = v.into_string() {
            if let Ok(n) = s.parse::<usize>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }
    }

    None
}

#[inline]
fn resolve_trial_threads(trials: u32, requested_threads: Option<usize>) -> usize {
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut threads = requested_threads
        .or_else(env_trial_threads)
        .unwrap_or(default_threads);
    if threads == 0 {
        threads = 1;
    }
    threads.min(trials as usize).max(1)
}

fn collect_trials_with_rng<R: TrialRng + Send>(
    graph: &Graph,
    seed: u32,
    trials: u32,
    directed: bool,
    multilevel: bool,
    worker_threads: usize,
    make_rng: fn(u32) -> R,
) -> Vec<(u32, TrialResult)> {
    // Minimap parallelizes outer trials; each trial run remains single-thread deterministic.
    if worker_threads == 1 {
        let mut out = Vec::with_capacity(trials as usize);
        for trial_index in 0..trials {
            let mut rng = make_rng(seed_for_trial(seed, trial_index));
            let trial = single_trial(graph, &mut rng, directed, multilevel);
            out.push((trial_index, trial));
        }
        return out;
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_threads)
        .build()
        .expect("failed to build trial thread pool");

    if trace_threads_enabled() {
        eprintln!(
            "[trial-par] requested_threads={} pool_threads={} trials={}",
            worker_threads,
            pool.current_num_threads(),
            trials
        );
        let with_threads: Vec<(u32, TrialResult, String)> = pool.install(|| {
            (0..trials)
                .into_par_iter()
                .map(|trial_index| {
                    let mut rng = make_rng(seed_for_trial(seed, trial_index));
                    let trial = single_trial(graph, &mut rng, directed, multilevel);
                    let tid = format!("{:?}", std::thread::current().id());
                    (trial_index, trial, tid)
                })
                .collect()
        });

        let mut unique_threads = std::collections::BTreeSet::new();
        let mut out = Vec::with_capacity(with_threads.len());
        for (trial_index, trial, tid) in with_threads {
            unique_threads.insert(tid);
            out.push((trial_index, trial));
        }
        eprintln!("[trial-par] used_worker_threads={}", unique_threads.len());
        out
    } else {
        pool.install(|| {
            (0..trials)
                .into_par_iter()
                .map(|trial_index| {
                    let mut rng = make_rng(seed_for_trial(seed, trial_index));
                    let trial = single_trial(graph, &mut rng, directed, multilevel);
                    (trial_index, trial)
                })
                .collect()
        })
    }
}

pub fn run_trials(
    graph: &Graph,
    seed: u32,
    num_trials: u32,
    directed: bool,
    multilevel: bool,
    requested_threads: Option<usize>,
    parity_rng: bool,
) -> TrialResult {
    let trials = num_trials.max(1);

    if trials == 1 {
        if parity_rng {
            let mut rng = Mt19937::new(seed);
            return single_trial(graph, &mut rng, directed, multilevel);
        }
        let mut rng = RustRng::new(seed);
        return single_trial(graph, &mut rng, directed, multilevel);
    }

    let worker_threads = resolve_trial_threads(trials, requested_threads);

    let mut trial_results: Vec<(u32, TrialResult)> = if parity_rng {
        collect_trials_with_rng(
            graph,
            seed,
            trials,
            directed,
            multilevel,
            worker_threads,
            Mt19937::new,
        )
    } else {
        collect_trials_with_rng(
            graph,
            seed,
            trials,
            directed,
            multilevel,
            worker_threads,
            RustRng::new,
        )
    };

    // Deterministic best-trial selection independent of worker scheduling.
    trial_results.sort_unstable_by_key(|(trial_index, _)| *trial_index);

    let mut best = trial_results
        .drain(..1)
        .next()
        .expect("at least one trial")
        .1;

    for (_, trial) in trial_results {
        if trial.codelength < best.codelength - MIN_CODELENGTH_IMPROVEMENT {
            best = trial;
        }
    }

    best
}
