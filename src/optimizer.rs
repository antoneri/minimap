use crate::graph::{FlowData, Graph};
use crate::objective::{DeltaFlow, MapEquationObjective, plogp};
use crate::rng_compat::{Mt19937, RustRng, TrialRng};
use rayon::prelude::*;

const CORE_LOOP_LIMIT: usize = 10;
const MIN_CODELENGTH_IMPROVEMENT: f64 = 1e-10;
const MIN_SINGLE_NODE_IMPROVEMENT: f64 = 1e-16;
const MIN_RELATIVE_TUNE_ITERATION_IMPROVEMENT: f64 = 1e-5;

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
}

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub node_to_module: Vec<u32>,
    pub num_modules: u32,
    pub codelength: f64,
    pub one_level_codelength: f64,
    pub module_data: Vec<FlowData>,
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
                self.consolidate_with_adj(node_module, module_data, workspace, &adj)
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
                self.consolidate_with_adj(node_module, module_data, workspace, &adj)
            }
        }
    }

    fn consolidate_with_adj<A: AdjAccess>(
        &self,
        node_module: &[u32],
        module_data: &[FlowData],
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

fn partition_codelength(
    objective: &mut MapEquationObjective,
    modules: &[FlowData],
    module_indices: &mut Vec<u32>,
) -> f64 {
    module_indices.resize(modules.len(), 0);
    for i in 0..modules.len() {
        module_indices[i] = i as u32;
    }
    objective.init_partition(modules, &module_indices[..modules.len()]);
    objective.codelength
}

fn active_modules_codelength(
    active: &ActiveNetwork<'_>,
    objective: &mut MapEquationObjective,
    workspace: &mut OptimizeWorkspace,
) -> f64 {
    workspace.flow_data.clear();
    workspace.flow_data.reserve(active.nodes.len());
    for node in &active.nodes {
        workspace.flow_data.push(node.data);
    }
    partition_codelength(
        objective,
        &workspace.flow_data,
        &mut workspace.module_indices,
    )
}

fn find_top_modules_repeatedly_from_leaf<'g>(
    leaf_network: &ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
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
        let lock = aggregation_level == 0;
        let level =
            optimize_active_network(&active, rng, objective, None, lock, loop_limit, workspace);

        if have_modules && level.codelength >= consolidated_codelength - MIN_SINGLE_NODE_IMPROVEMENT
        {
            break;
        }

        let next = active.consolidate(&level.node_module, &level.module_data, workspace);
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

        if level.codelength >= consolidated_codelength - MIN_SINGLE_NODE_IMPROVEMENT {
            break;
        }

        let next = active.consolidate(&level.node_module, &level.module_data, workspace);
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

fn fine_tune<'g>(
    leaf_network: &ActiveNetwork<'g>,
    top_network: &mut ActiveNetwork<'g>,
    objective: &mut MapEquationObjective,
    rng: &mut impl TrialRng,
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

    *top_network = leaf_network.consolidate(&level.node_module, &level.module_data, workspace);
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

fn single_trial(graph: &Graph, rng: &mut impl TrialRng, directed: bool) -> TrialResult {
    let node_data = graph.node_data.clone();
    let mut objective = MapEquationObjective::new(&node_data);
    let mut workspace = OptimizeWorkspace::default();

    let leaf_network = ActiveNetwork::from_graph(graph);
    let mut top_network =
        find_top_modules_repeatedly_from_leaf(&leaf_network, &mut objective, rng, &mut workspace);

    let one_level = one_level_codelength(graph);
    let mut old_codelength =
        active_modules_codelength(&top_network, &mut objective, &mut workspace);

    let mut do_fine_tune = true;
    let mut coarse_tuned = false;

    while top_network.nodes.len() > 1 {
        if do_fine_tune {
            let num_effective_loops = fine_tune(
                &leaf_network,
                &mut top_network,
                &mut objective,
                rng,
                &mut workspace,
            );
            if num_effective_loops > 0 {
                top_network = find_top_modules_repeatedly_from_modules(
                    &top_network,
                    &mut objective,
                    rng,
                    &mut workspace,
                );
            }
        } else {
            coarse_tuned = true;
        }

        let new_codelength =
            active_modules_codelength(&top_network, &mut objective, &mut workspace);

        let is_improvement = new_codelength <= old_codelength - MIN_CODELENGTH_IMPROVEMENT
            && new_codelength
                < old_codelength - one_level * MIN_RELATIVE_TUNE_ITERATION_IMPROVEMENT;

        if !is_improvement {
            if coarse_tuned {
                break;
            }
        } else {
            old_codelength = new_codelength;
        }

        do_fine_tune = !do_fine_tune;
    }

    let (node_to_module, num_modules) = flatten_active_to_assignment(&top_network);
    let module_data = compute_module_data(graph, &node_to_module, num_modules, directed);

    let mut final_obj = MapEquationObjective::new(&node_data);
    let module_indices: Vec<u32> = (0..num_modules).collect();
    final_obj.init_partition(&module_data, &module_indices);

    TrialResult {
        node_to_module,
        num_modules,
        codelength: final_obj.codelength,
        one_level_codelength: one_level,
        module_data,
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
    worker_threads: usize,
    make_rng: fn(u32) -> R,
) -> Vec<(u32, TrialResult)> {
    // Minimap parallelizes outer trials; each trial run remains single-thread deterministic.
    if worker_threads == 1 {
        let mut out = Vec::with_capacity(trials as usize);
        for trial_index in 0..trials {
            let mut rng = make_rng(seed_for_trial(seed, trial_index));
            let trial = single_trial(graph, &mut rng, directed);
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
                    let trial = single_trial(graph, &mut rng, directed);
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
                    let trial = single_trial(graph, &mut rng, directed);
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
    requested_threads: Option<usize>,
    parity_rng: bool,
) -> TrialResult {
    let trials = num_trials.max(1);

    if trials == 1 {
        if parity_rng {
            let mut rng = Mt19937::new(seed);
            return single_trial(graph, &mut rng, directed);
        }
        let mut rng = RustRng::new(seed);
        return single_trial(graph, &mut rng, directed);
    }

    let worker_threads = resolve_trial_threads(trials, requested_threads);

    let mut trial_results: Vec<(u32, TrialResult)> = if parity_rng {
        collect_trials_with_rng(graph, seed, trials, directed, worker_threads, Mt19937::new)
    } else {
        collect_trials_with_rng(graph, seed, trials, directed, worker_threads, RustRng::new)
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
