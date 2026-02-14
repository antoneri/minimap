use crate::graph::{FlowData, Graph};
use crate::objective::{DeltaFlow, MapEquationObjective, plogp};
use crate::rng_compat::Mt19937;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

const CORE_LOOP_LIMIT: usize = 10;
const MIN_CODELENGTH_IMPROVEMENT: f64 = 1e-10;
const MIN_SINGLE_NODE_IMPROVEMENT: f64 = 1e-16;
const MIN_RELATIVE_TUNE_ITERATION_IMPROVEMENT: f64 = 1e-5;

#[derive(Debug, Clone)]
struct ActiveNode {
    data: FlowData,
    out_edges: Vec<(u32, f64)>,
    in_edges: Vec<(u32, f64)>,
    members: Vec<u32>,
}

impl ActiveNode {
    #[inline]
    fn degree(&self) -> usize {
        self.out_edges.len() + self.in_edges.len()
    }
}

#[derive(Debug, Clone)]
struct ActiveNetwork {
    nodes: Vec<ActiveNode>,
}

#[derive(Debug, Clone)]
struct OptimizeLevelResult {
    node_module: Vec<u32>,
    module_data: Vec<FlowData>,
    codelength: f64,
    effective_loops: u32,
}

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub node_to_module: Vec<u32>,
    pub num_modules: u32,
    pub codelength: f64,
    pub one_level_codelength: f64,
    pub module_data: Vec<FlowData>,
}

impl ActiveNetwork {
    fn from_graph(graph: &Graph) -> Self {
        let n = graph.node_count();
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(ActiveNode {
                data: graph.nodes[i].data,
                out_edges: Vec::new(),
                in_edges: Vec::new(),
                members: vec![i as u32],
            });
        }

        for s in 0..n {
            for e in graph.out_range(s) {
                let t = graph.edge_target[e] as usize;
                if s == t {
                    continue;
                }
                let f = graph.edge_flow[e];
                nodes[s].out_edges.push((t as u32, f));
                nodes[t].in_edges.push((s as u32, f));
            }
        }

        Self { nodes }
    }

    fn consolidate(&self, node_module: &[u32], module_data: &[FlowData]) -> Self {
        let n = self.nodes.len();
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
                out_edges: Vec::new(),
                in_edges: Vec::new(),
                members: Vec::new(),
            })
            .collect();

        for (new_m, &old_m) in ordered_old_modules.iter().enumerate() {
            new_nodes[new_m].data = module_data[old_m as usize];
        }

        for i in 0..n {
            let old_m = node_module[i] as usize;
            let new_m = old_to_new[old_m] as usize;
            new_nodes[new_m]
                .members
                .extend_from_slice(&self.nodes[i].members);
        }

        let mut edge_map: FxHashMap<u64, f64> = FxHashMap::default();
        edge_map.reserve(self.nodes.iter().map(|n| n.out_edges.len()).sum());

        for i in 0..n {
            let src_m = old_to_new[node_module[i] as usize];
            for &(t, f) in &self.nodes[i].out_edges {
                let dst_m = old_to_new[node_module[t as usize] as usize];
                if src_m == dst_m {
                    continue;
                }
                let key = ((src_m as u64) << 32) | (dst_m as u64);
                *edge_map.entry(key).or_insert(0.0) += f;
            }
        }

        let mut edges: Vec<(u32, u32, f64)> = edge_map
            .into_iter()
            .map(|(k, v)| (((k >> 32) as u32), (k as u32), v))
            .collect();
        edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        for (s, t, f) in edges {
            new_nodes[s as usize].out_edges.push((t, f));
            new_nodes[t as usize].in_edges.push((s, f));
        }

        Self { nodes: new_nodes }
    }

    fn assignment_to_leaves(&self, leaf_count: usize) -> Vec<u32> {
        let mut out = vec![0u32; leaf_count];
        for (module_idx, node) in self.nodes.iter().enumerate() {
            for &leaf in &node.members {
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

fn move_node_to_predefined_module(
    active: &ActiveNetwork,
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

    for &(nbr, flow) in &active.nodes[node_idx].out_edges {
        let other_module = node_module[nbr as usize];
        if other_module == old_module {
            old_delta.delta_exit += flow;
        } else if other_module == new_module {
            new_delta.delta_exit += flow;
        }
    }
    for &(nbr, flow) in &active.nodes[node_idx].in_edges {
        let other_module = node_module[nbr as usize];
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

fn try_move_each_node_into_best_module(
    active: &ActiveNetwork,
    rng: &mut Mt19937,
    objective: &mut MapEquationObjective,
    node_module: &mut [u32],
    module_data: &mut [FlowData],
    module_members: &mut [u32],
    dirty: &mut [bool],
    empty_modules: &mut Vec<u32>,
    lock_multi_module_nodes: bool,
) -> u32 {
    let n = active.nodes.len();

    let mut node_order = vec![0u32; n];
    rng.randomized_index_vector(&mut node_order);

    let mut moved = 0u32;

    let mut redirect = vec![u32::MAX; n];
    let mut touched_modules: Vec<u32> = Vec::with_capacity(64);
    let mut cand_modules: Vec<u32> = Vec::with_capacity(64);
    let mut cand_delta_exit: Vec<f64> = Vec::with_capacity(64);
    let mut cand_delta_enter: Vec<f64> = Vec::with_capacity(64);
    let mut module_order: Vec<u32> = Vec::with_capacity(64);

    for &node_u32 in &node_order {
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

        for &(nbr, flow) in &active.nodes[node_idx].out_edges {
            let m = node_module[nbr as usize];
            add_candidate(
                m,
                flow,
                0.0,
                &mut redirect,
                &mut touched_modules,
                &mut cand_modules,
                &mut cand_delta_exit,
                &mut cand_delta_enter,
            );
        }

        for &(nbr, flow) in &active.nodes[node_idx].in_edges {
            let m = node_module[nbr as usize];
            add_candidate(
                m,
                0.0,
                flow,
                &mut redirect,
                &mut touched_modules,
                &mut cand_modules,
                &mut cand_delta_exit,
                &mut cand_delta_enter,
            );
        }

        add_candidate(
            current_module,
            0.0,
            0.0,
            &mut redirect,
            &mut touched_modules,
            &mut cand_modules,
            &mut cand_delta_exit,
            &mut cand_delta_enter,
        );

        if module_members[current_module as usize] > 1 {
            if let Some(&empty_module) = empty_modules.last() {
                add_candidate(
                    empty_module,
                    0.0,
                    0.0,
                    &mut redirect,
                    &mut touched_modules,
                    &mut cand_modules,
                    &mut cand_delta_exit,
                    &mut cand_delta_enter,
                );
            }
        }

        let old_idx = redirect[current_module as usize] as usize;
        let old_delta = DeltaFlow {
            module: current_module,
            delta_exit: cand_delta_exit[old_idx],
            delta_enter: cand_delta_enter[old_idx],
        };

        module_order.clear();
        module_order.resize(cand_modules.len(), 0);
        rng.randomized_index_vector(&mut module_order);

        let mut best_module = current_module;
        let mut best_delta = old_delta;
        let mut best_delta_codelength = 0.0f64;

        let mut strongest_module = current_module;
        let mut strongest_delta = old_delta;
        let mut strongest_delta_codelength = 0.0f64;

        for &enum_idx in &module_order {
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

            let delta = objective.get_delta_on_move(
                &active.nodes[node_idx].data,
                &old_delta,
                &cand_delta,
                module_data,
            );

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

            for &(nbr, _) in &active.nodes[node_idx].out_edges {
                let nbr_idx = nbr as usize;
                dirty[nbr_idx] = true;
                if node_module[nbr_idx] == old_module {
                    node_in_old_module = nbr;
                    num_linked_nodes_in_old_module += 1;
                }
            }
            for &(nbr, _) in &active.nodes[node_idx].in_edges {
                let nbr_idx = nbr as usize;
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
                    companion_idx,
                    best_module,
                    objective,
                    node_module,
                    module_data,
                    module_members,
                    empty_modules,
                ) {
                    moved += 1;

                    if active.nodes[companion_idx].degree() > 1 {
                        for &(nbr, _) in &active.nodes[companion_idx].out_edges {
                            dirty[nbr as usize] = true;
                        }
                        for &(nbr, _) in &active.nodes[companion_idx].in_edges {
                            dirty[nbr as usize] = true;
                        }
                    }
                }
            }
        } else {
            dirty[node_idx] = false;
        }

        for &m in &touched_modules {
            redirect[m as usize] = u32::MAX;
        }
    }

    moved
}

fn optimize_active_network(
    active: &ActiveNetwork,
    rng: &mut Mt19937,
    objective: &mut MapEquationObjective,
    predefined_modules: Option<&[u32]>,
    lock_multi_module_nodes: bool,
    loop_limit: usize,
) -> OptimizeLevelResult {
    let n = active.nodes.len();

    let mut node_module: Vec<u32> = (0..n as u32).collect();
    let mut module_data: Vec<FlowData> = active.nodes.iter().map(|n| n.data).collect();
    let mut module_members = vec![1u32; n];
    let mut dirty = vec![true; n];
    let mut empty_modules: Vec<u32> = Vec::with_capacity(n);

    let module_indices: Vec<u32> = (0..n as u32).collect();
    objective.init_partition(&module_data, &module_indices);

    if let Some(modules) = predefined_modules {
        if modules.len() != n {
            panic!("predefined module length {} != active node count {}", modules.len(), n);
        }
        for i in 0..n {
            let new_m = modules[i];
            let _ = move_node_to_predefined_module(
                active,
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
            rng,
            objective,
            &mut node_module,
            &mut module_data,
            &mut module_members,
            &mut dirty,
            &mut empty_modules,
            lock_multi_module_nodes,
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

fn assignment_from_top_network(top_network: &ActiveNetwork, leaf_count: usize) -> Vec<u32> {
    top_network.assignment_to_leaves(leaf_count)
}

fn partition_codelength(node_data: &[FlowData], objective: &mut MapEquationObjective, modules: &[FlowData]) -> f64 {
    let mut module_indices = Vec::with_capacity(modules.len());
    for i in 0..modules.len() {
        module_indices.push(i as u32);
    }
    objective.init_partition(modules, &module_indices);

    // Keep objective bound to the same constant node data.
    let _ = node_data;

    objective.codelength
}

fn find_top_modules_repeatedly_from_leaf(
    leaf_network: &ActiveNetwork,
    objective: &mut MapEquationObjective,
    rng: &mut Mt19937,
) -> ActiveNetwork {
    let mut have_modules = false;
    let mut active = leaf_network.clone();
    let mut consolidated_codelength = f64::INFINITY;
    let mut aggregation_level = 0usize;

    loop {
        if active.nodes.len() <= 1 {
            break;
        }

        let loop_limit = if aggregation_level > 0 { 20 } else { CORE_LOOP_LIMIT };
        let lock = aggregation_level == 0;
        let level = optimize_active_network(&active, rng, objective, None, lock, loop_limit);

        if have_modules
            && level.codelength >= consolidated_codelength - MIN_SINGLE_NODE_IMPROVEMENT
        {
            break;
        }

        let next = active.consolidate(&level.node_module, &level.module_data);
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

fn find_top_modules_repeatedly_from_modules(
    active_top: &ActiveNetwork,
    objective: &mut MapEquationObjective,
    rng: &mut Mt19937,
) -> ActiveNetwork {
    let mut active = active_top.clone();
    let mut consolidated_codelength = partition_codelength(
        &[],
        objective,
        &active.nodes.iter().map(|n| n.data).collect::<Vec<_>>(),
    );
    let mut aggregation_level = 0usize;

    loop {
        if active.nodes.len() <= 1 {
            break;
        }

        let loop_limit = if aggregation_level > 0 { 20 } else { CORE_LOOP_LIMIT };
        let level = optimize_active_network(&active, rng, objective, None, false, loop_limit);

        if level.codelength >= consolidated_codelength - MIN_SINGLE_NODE_IMPROVEMENT {
            break;
        }

        let next = active.consolidate(&level.node_module, &level.module_data);
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

fn fine_tune(
    leaf_network: &ActiveNetwork,
    top_network: &mut ActiveNetwork,
    objective: &mut MapEquationObjective,
    rng: &mut Mt19937,
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
    );

    if level.effective_loops == 0 {
        return 0;
    }

    *top_network = leaf_network.consolidate(&level.node_module, &level.module_data);
    level.effective_loops
}

fn flatten_active_to_assignment(active: &ActiveNetwork) -> (Vec<u32>, u32) {
    let mut max_member = 0usize;
    for n in &active.nodes {
        for &m in &n.members {
            max_member = max_member.max(m as usize);
        }
    }
    let mut out = vec![0u32; max_member + 1];
    for (module_idx, n) in active.nodes.iter().enumerate() {
        for &m in &n.members {
            out[m as usize] = module_idx as u32;
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

    for (i, node) in graph.nodes.iter().enumerate() {
        let m = node_to_module[i] as usize;
        modules[m].flow += node.data.flow;
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
    for node in &graph.nodes {
        sum -= plogp(node.data.flow);
    }
    sum
}

fn single_trial(graph: &Graph, rng: &mut Mt19937, directed: bool) -> TrialResult {
    let node_data: Vec<FlowData> = graph.nodes.iter().map(|n| n.data).collect();
    let mut objective = MapEquationObjective::new(&node_data);

    let leaf_network = ActiveNetwork::from_graph(graph);
    let mut top_network = find_top_modules_repeatedly_from_leaf(&leaf_network, &mut objective, rng);

    let one_level = one_level_codelength(graph);
    let mut old_codelength = partition_codelength(
        &node_data,
        &mut objective,
        &top_network.nodes.iter().map(|n| n.data).collect::<Vec<_>>(),
    );

    let mut do_fine_tune = true;
    let mut coarse_tuned = false;

    while top_network.nodes.len() > 1 {
        if do_fine_tune {
            let num_effective_loops = fine_tune(&leaf_network, &mut top_network, &mut objective, rng);
            if num_effective_loops > 0 {
                top_network =
                    find_top_modules_repeatedly_from_modules(&top_network, &mut objective, rng);
            }
        } else {
            coarse_tuned = true;
        }

        let new_codelength = partition_codelength(
            &node_data,
            &mut objective,
            &top_network.nodes.iter().map(|n| n.data).collect::<Vec<_>>(),
        );

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

pub fn run_trials(
    graph: &Graph,
    seed: u32,
    num_trials: u32,
    directed: bool,
    requested_threads: Option<usize>,
) -> TrialResult {
    let trials = num_trials.max(1);

    if trials == 1 {
        let mut rng = Mt19937::new(seed);
        return single_trial(graph, &mut rng, directed);
    }

    let worker_threads = resolve_trial_threads(trials, requested_threads);

    let mut trial_results: Vec<(u32, TrialResult)> = if worker_threads == 1 {
        let mut out = Vec::with_capacity(trials as usize);
        for trial_index in 0..trials {
            let mut rng = Mt19937::new(seed_for_trial(seed, trial_index));
            let trial = single_trial(graph, &mut rng, directed);
            out.push((trial_index, trial));
        }
        out
    } else {
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
                        let mut rng = Mt19937::new(seed_for_trial(seed, trial_index));
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
            eprintln!(
                "[trial-par] used_worker_threads={}",
                unique_threads.len()
            );
            out
        } else {
            pool.install(|| {
                (0..trials)
                    .into_par_iter()
                    .map(|trial_index| {
                        let mut rng = Mt19937::new(seed_for_trial(seed, trial_index));
                        let trial = single_trial(graph, &mut rng, directed);
                        (trial_index, trial)
                    })
                    .collect()
            })
        }
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
