use crate::config::Config;
use crate::graph::Graph;
use crate::optimizer::{HierarchyResult, TrialResult};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

const VERSION: &str = "2.8.1";
const ROOT_PARENT: u32 = u32::MAX;

fn fmt_sig(v: f64, sig: usize) -> String {
    if !v.is_finite() {
        return v.to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }

    let abs = v.abs();
    let digits_before = if abs >= 1.0 {
        abs.log10().floor() as i32 + 1
    } else {
        abs.log10().floor() as i32 + 1
    };

    let decimals = if digits_before >= sig as i32 {
        0usize
    } else {
        (sig as i32 - digits_before).max(0) as usize
    };

    let mut s = format!("{:.*}", decimals, v);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

fn format_started_at(start_time: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = start_time.into();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
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

#[inline]
fn pair_key(parent: u32, child: u32) -> u64 {
    ((parent as u64) << 32) | (child as u64)
}

fn path_to_string(path: &[u32]) -> String {
    let mut s = String::new();
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            s.push(':');
        }
        s.push_str(&p.to_string());
    }
    s
}

#[derive(Debug)]
struct OrderedModules {
    old_to_new: Vec<u32>,
    new_to_old: Vec<u32>,
    module_nodes: Vec<Vec<usize>>,
    node_child_pos: Vec<u32>,
}

fn build_ordered_modules(graph: &Graph, trial: &TrialResult) -> OrderedModules {
    let k = trial.num_modules as usize;
    let mut modules: Vec<u32> = (0..trial.num_modules).collect();
    modules.sort_unstable_by(|&a, &b| {
        let fa = trial.module_data[a as usize].flow;
        let fb = trial.module_data[b as usize].flow;
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut old_to_new = vec![0u32; k];
    for (new_idx, &old_idx) in modules.iter().enumerate() {
        old_to_new[old_idx as usize] = new_idx as u32;
    }

    let mut module_nodes: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (node_idx, &old_m) in trial.node_to_module.iter().enumerate() {
        let new_m = old_to_new[old_m as usize] as usize;
        module_nodes[new_m].push(node_idx);
    }

    for nodes in &mut module_nodes {
        nodes.sort_unstable_by(|&a, &b| {
            let fa = graph.node_data[a].flow;
            let fb = graph.node_data[b].flow;
            fb.partial_cmp(&fa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(graph.node_ids[a].cmp(&graph.node_ids[b]))
        });
    }

    let mut node_child_pos = vec![0u32; graph.node_count()];
    for nodes in &module_nodes {
        for (pos, &node_idx) in nodes.iter().enumerate() {
            node_child_pos[node_idx] = (pos + 1) as u32;
        }
    }

    OrderedModules {
        old_to_new,
        new_to_old: modules,
        module_nodes,
        node_child_pos,
    }
}

#[derive(Debug)]
struct OrderedHierarchy {
    leaf_base: u32,
    root_children: Vec<u32>,
    module_children_offsets: Vec<u32>,
    child_pos_lookup: FxHashMap<u64, u32>,
    module_label_paths: Vec<Vec<u32>>,
    leaf_label_paths: Vec<Vec<u32>>,
    leaf_order: Vec<usize>,
    top_module_count: u32,
}

impl OrderedHierarchy {
    fn module_child_range(&self, module: usize) -> std::ops::Range<usize> {
        self.module_children_offsets[module] as usize
            ..self.module_children_offsets[module + 1] as usize
    }

    fn lookup_child_pos(&self, parent: u32, child: u32) -> Option<u32> {
        self.child_pos_lookup.get(&pair_key(parent, child)).copied()
    }
}

fn build_ordered_hierarchy(graph: &Graph, hier: &HierarchyResult) -> OrderedHierarchy {
    let leaf_base = graph.node_count() as u32;
    let module_count = hier.module_parent.len();

    let mut module_min_id = vec![u32::MAX; module_count];
    for start in 0..module_count {
        if module_min_id[start] != u32::MAX {
            continue;
        }

        let mut stack: Vec<(usize, bool)> = vec![(start, false)];
        while let Some((m, done)) = stack.pop() {
            if done {
                if module_min_id[m] != u32::MAX {
                    continue;
                }
                let range = hier.module_children_offsets[m] as usize
                    ..hier.module_children_offsets[m + 1] as usize;
                let mut min_id = u32::MAX;
                for &child in &hier.module_children[range] {
                    if child < leaf_base {
                        min_id = min_id.min(graph.node_ids[child as usize]);
                    } else {
                        min_id = min_id.min(module_min_id[(child - leaf_base) as usize]);
                    }
                }
                module_min_id[m] = min_id;
                continue;
            }

            if module_min_id[m] != u32::MAX {
                continue;
            }

            stack.push((m, true));
            let range = hier.module_children_offsets[m] as usize
                ..hier.module_children_offsets[m + 1] as usize;
            for &child in &hier.module_children[range] {
                if child >= leaf_base {
                    let cm = (child - leaf_base) as usize;
                    if module_min_id[cm] == u32::MAX {
                        stack.push((cm, false));
                    }
                }
            }
        }
    }

    let child_sort = |a: &u32, b: &u32| {
        let fa = if *a < leaf_base {
            graph.node_data[*a as usize].flow
        } else {
            hier.module_data[(*a - leaf_base) as usize].flow
        };
        let fb = if *b < leaf_base {
            graph.node_data[*b as usize].flow
        } else {
            hier.module_data[(*b - leaf_base) as usize].flow
        };

        let ia = if *a < leaf_base {
            graph.node_ids[*a as usize]
        } else {
            module_min_id[(*a - leaf_base) as usize]
        };
        let ib = if *b < leaf_base {
            graph.node_ids[*b as usize]
        } else {
            module_min_id[(*b - leaf_base) as usize]
        };

        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(&ib))
            .then(a.cmp(b))
    };

    let mut root_children: Vec<u32> = Vec::new();
    for m in 0..module_count as u32 {
        if hier.module_parent[m as usize] == u32::MAX {
            root_children.push(leaf_base + m);
        }
    }
    for leaf in 0..graph.node_count() {
        if hier.leaf_paths[leaf].is_empty() {
            root_children.push(leaf as u32);
        }
    }
    root_children.sort_unstable_by(child_sort);

    let top_module_count = root_children.iter().filter(|&&c| c >= leaf_base).count() as u32;

    let mut module_children_offsets = Vec::with_capacity(module_count + 1);
    module_children_offsets.push(0);
    let mut module_children = Vec::<u32>::new();

    for m in 0..module_count {
        let range =
            hier.module_children_offsets[m] as usize..hier.module_children_offsets[m + 1] as usize;
        let mut children = hier.module_children[range].to_vec();
        children.sort_unstable_by(child_sort);
        module_children.extend_from_slice(&children);
        module_children_offsets.push(module_children.len() as u32);
    }

    let mut child_pos_lookup: FxHashMap<u64, u32> = FxHashMap::default();
    child_pos_lookup.reserve(root_children.len() + module_children.len());

    for (pos, &child) in root_children.iter().enumerate() {
        child_pos_lookup.insert(pair_key(ROOT_PARENT, child), (pos + 1) as u32);
    }
    for m in 0..module_count {
        let start = module_children_offsets[m] as usize;
        let end = module_children_offsets[m + 1] as usize;
        for (local_pos, &child) in module_children[start..end].iter().enumerate() {
            child_pos_lookup.insert(pair_key(m as u32, child), (local_pos + 1) as u32);
        }
    }

    let mut module_label_paths: Vec<Vec<u32>> = vec![Vec::new(); module_count];
    for m in 0..module_count {
        let mut chain_rev = Vec::<u32>::new();
        let mut cur = m as u32;
        loop {
            chain_rev.push(cur);
            let p = hier.module_parent[cur as usize];
            if p == u32::MAX {
                break;
            }
            cur = p;
        }
        chain_rev.reverse();

        let mut labels = Vec::with_capacity(chain_rev.len());
        let mut parent = ROOT_PARENT;
        for module in chain_rev {
            let child = leaf_base + module;
            let label = child_pos_lookup
                .get(&pair_key(parent, child))
                .copied()
                .unwrap_or(1);
            labels.push(label);
            parent = module;
        }
        module_label_paths[m] = labels;
    }

    let mut leaf_label_paths: Vec<Vec<u32>> = vec![Vec::new(); graph.node_count()];
    for leaf in 0..graph.node_count() {
        let mut labels = Vec::with_capacity(hier.leaf_paths[leaf].len() + 1);
        let mut parent = ROOT_PARENT;

        for &module in &hier.leaf_paths[leaf] {
            let child = leaf_base + module;
            let label = child_pos_lookup
                .get(&pair_key(parent, child))
                .copied()
                .unwrap_or(1);
            labels.push(label);
            parent = module;
        }

        let leaf_label = child_pos_lookup
            .get(&pair_key(parent, leaf as u32))
            .copied()
            .unwrap_or(1);
        labels.push(leaf_label);

        leaf_label_paths[leaf] = labels;
    }

    let mut leaf_order: Vec<usize> = (0..graph.node_count()).collect();
    leaf_order.sort_unstable_by(|&a, &b| {
        leaf_label_paths[a]
            .cmp(&leaf_label_paths[b])
            .then(graph.node_ids[a].cmp(&graph.node_ids[b]))
    });

    OrderedHierarchy {
        leaf_base,
        root_children,
        module_children_offsets,
        child_pos_lookup,
        module_label_paths,
        leaf_label_paths,
        leaf_order,
        top_module_count,
    }
}

fn output_base_path(cfg: &Config) -> PathBuf {
    let mut p = cfg.out_dir.clone();
    p.push(&cfg.out_name);
    p
}

fn write_header(
    w: &mut BufWriter<File>,
    cfg: &Config,
    started: &str,
    elapsed: Duration,
    levels: u32,
    top_modules: u32,
    codelength: f64,
    one_level: f64,
) -> Result<(), String> {
    let rel = if one_level.abs() < 1e-16 {
        0.0
    } else {
        (1.0 - codelength / one_level) * 100.0
    };

    writeln!(w, "# v{}", VERSION).map_err(|e| e.to_string())?;
    writeln!(w, "# minimap {}", cfg.raw_args).map_err(|e| e.to_string())?;
    writeln!(w, "# started at {}", started).map_err(|e| e.to_string())?;
    writeln!(w, "# completed in {} s", elapsed.as_secs_f64()).map_err(|e| e.to_string())?;
    writeln!(
        w,
        "# partitioned into {} levels with {} top modules",
        levels, top_modules
    )
    .map_err(|e| e.to_string())?;
    writeln!(w, "# codelength {} bits", fmt_sig(codelength, 6)).map_err(|e| e.to_string())?;
    writeln!(w, "# relative codelength savings {}%", fmt_sig(rel, 6)).map_err(|e| e.to_string())?;
    writeln!(
        w,
        "# flow model {}",
        if cfg.directed {
            "directed"
        } else {
            "undirected"
        }
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn write_tree_file_two_level(
    path: &Path,
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    ordered: &OrderedModules,
    started: &str,
    elapsed: Duration,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let mut w = BufWriter::new(file);

    write_header(
        &mut w,
        cfg,
        started,
        elapsed,
        trial.levels,
        trial.num_modules,
        trial.codelength,
        trial.one_level_codelength,
    )?;
    writeln!(w, "# path flow name node_id").map_err(|e| e.to_string())?;

    for (module_zero, nodes) in ordered.module_nodes.iter().enumerate() {
        let module_id = module_zero as u32 + 1;
        for (pos, &node_idx) in nodes.iter().enumerate() {
            let path_s = format!("{}:{}", module_id, pos + 1);
            let flow_s = fmt_sig(graph.node_data[node_idx].flow, 6);
            let name_s = graph.node_name_or_id(node_idx);
            writeln!(
                w,
                "{} {} \"{}\" {}",
                path_s, flow_s, name_s, graph.node_ids[node_idx]
            )
            .map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())
}

fn write_clu_file_two_level(
    path: &Path,
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    ordered: &OrderedModules,
    started: &str,
    elapsed: Duration,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let mut w = BufWriter::new(file);

    write_header(
        &mut w,
        cfg,
        started,
        elapsed,
        trial.levels,
        trial.num_modules,
        trial.codelength,
        trial.one_level_codelength,
    )?;
    writeln!(w, "# module level {}", cfg.clu_level).map_err(|e| e.to_string())?;
    writeln!(w, "# node_id module flow").map_err(|e| e.to_string())?;

    for (module_zero, nodes) in ordered.module_nodes.iter().enumerate() {
        let module_id = module_zero as u32 + 1;
        for &node_idx in nodes {
            writeln!(
                w,
                "{} {} {}",
                graph.node_ids[node_idx],
                module_id,
                fmt_sig(graph.node_data[node_idx].flow, 6)
            )
            .map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())
}

fn write_ftree_file_two_level(
    path: &Path,
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    ordered: &OrderedModules,
    started: &str,
    elapsed: Duration,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let mut w = BufWriter::new(file);

    write_header(
        &mut w,
        cfg,
        started,
        elapsed,
        trial.levels,
        trial.num_modules,
        trial.codelength,
        trial.one_level_codelength,
    )?;
    writeln!(w, "# path flow name node_id").map_err(|e| e.to_string())?;

    for (module_zero, nodes) in ordered.module_nodes.iter().enumerate() {
        let module_id = module_zero as u32 + 1;
        for (pos, &node_idx) in nodes.iter().enumerate() {
            let path_s = format!("{}:{}", module_id, pos + 1);
            let flow_s = fmt_sig(graph.node_data[node_idx].flow, 6);
            let name_s = graph.node_name_or_id(node_idx);
            writeln!(
                w,
                "{} {} \"{}\" {}",
                path_s, flow_s, name_s, graph.node_ids[node_idx]
            )
            .map_err(|e| e.to_string())?;
        }
    }

    let mut root_links: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    let mut module_links: Vec<BTreeMap<(u32, u32), f64>> =
        vec![BTreeMap::new(); trial.num_modules as usize];

    for e in 0..graph.edge_count() {
        let s = graph.edge_source[e] as usize;
        let t = graph.edge_target[e] as usize;
        if s == t {
            continue;
        }

        let old_ms = trial.node_to_module[s] as usize;
        let old_mt = trial.node_to_module[t] as usize;
        let new_ms = ordered.old_to_new[old_ms] + 1;
        let new_mt = ordered.old_to_new[old_mt] + 1;
        let f = graph.edge_flow[e];

        if new_ms != new_mt {
            *root_links.entry((new_ms, new_mt)).or_insert(0.0) += f;
        } else {
            let src_pos = ordered.node_child_pos[s];
            let dst_pos = ordered.node_child_pos[t];
            if src_pos != dst_pos {
                *module_links[(new_ms - 1) as usize]
                    .entry((src_pos, dst_pos))
                    .or_insert(0.0) += f;
            }
        }
    }

    writeln!(
        w,
        "*Links {}",
        if cfg.directed {
            "directed"
        } else {
            "undirected"
        }
    )
    .map_err(|e| e.to_string())?;
    writeln!(w, "#*Links path enterFlow exitFlow numEdges numChildren")
        .map_err(|e| e.to_string())?;

    writeln!(
        w,
        "*Links root 0 0 {} {}",
        root_links.len(),
        trial.num_modules
    )
    .map_err(|e| e.to_string())?;

    for ((s, t), f) in &root_links {
        writeln!(w, "{} {} {}", s, t, fmt_sig(*f, 6)).map_err(|e| e.to_string())?;
    }

    for new_idx in 0..trial.num_modules as usize {
        let old_idx = ordered.new_to_old[new_idx] as usize;
        let md = trial.module_data[old_idx];
        let links = &module_links[new_idx];
        let num_children = ordered.module_nodes[new_idx].len();

        writeln!(
            w,
            "*Links {} {} {} {} {}",
            new_idx + 1,
            fmt_sig(md.enter_flow, 6),
            fmt_sig(md.exit_flow, 6),
            links.len(),
            num_children
        )
        .map_err(|e| e.to_string())?;

        for ((s, t), f) in links {
            writeln!(w, "{} {} {}", s, t, fmt_sig(*f, 6)).map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())
}

fn write_tree_file_multilevel(
    path: &Path,
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    ordered: &OrderedHierarchy,
    started: &str,
    elapsed: Duration,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let mut w = BufWriter::new(file);

    write_header(
        &mut w,
        cfg,
        started,
        elapsed,
        trial.levels,
        ordered.top_module_count,
        trial.codelength,
        trial.one_level_codelength,
    )?;
    writeln!(w, "# path flow name node_id").map_err(|e| e.to_string())?;

    for &leaf in &ordered.leaf_order {
        let path_s = path_to_string(&ordered.leaf_label_paths[leaf]);
        let flow_s = fmt_sig(graph.node_data[leaf].flow, 6);
        let name_s = graph.node_name_or_id(leaf);
        writeln!(
            w,
            "{} {} \"{}\" {}",
            path_s, flow_s, name_s, graph.node_ids[leaf]
        )
        .map_err(|e| e.to_string())?;
    }

    w.flush().map_err(|e| e.to_string())
}

fn write_clu_file_multilevel(
    path: &Path,
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    hier: &HierarchyResult,
    ordered: &OrderedHierarchy,
    started: &str,
    elapsed: Duration,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let mut w = BufWriter::new(file);

    write_header(
        &mut w,
        cfg,
        started,
        elapsed,
        trial.levels,
        ordered.top_module_count,
        trial.codelength,
        trial.one_level_codelength,
    )?;
    writeln!(w, "# module level {}", cfg.clu_level).map_err(|e| e.to_string())?;
    writeln!(w, "# node_id module flow").map_err(|e| e.to_string())?;

    let target_level = cfg.clu_level.max(1) as usize;

    let mut key_to_module_id: FxHashMap<u64, u32> = FxHashMap::default();
    key_to_module_id.reserve(graph.node_count());
    let mut module_nodes: Vec<Vec<usize>> = Vec::new();

    for &leaf in &ordered.leaf_order {
        let path = &hier.leaf_paths[leaf];
        let key = if path.is_empty() {
            (1u64 << 63) | leaf as u64
        } else {
            path[(target_level - 1).min(path.len() - 1)] as u64
        };

        let module_id = if let Some(&id) = key_to_module_id.get(&key) {
            id
        } else {
            let id = (module_nodes.len() as u32) + 1;
            key_to_module_id.insert(key, id);
            module_nodes.push(Vec::new());
            id
        };
        module_nodes[(module_id - 1) as usize].push(leaf);
    }

    for (idx, nodes) in module_nodes.iter().enumerate() {
        let module_id = idx as u32 + 1;
        for &leaf in nodes {
            writeln!(
                w,
                "{} {} {}",
                graph.node_ids[leaf],
                module_id,
                fmt_sig(graph.node_data[leaf].flow, 6)
            )
            .map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())
}

fn write_ftree_file_multilevel(
    path: &Path,
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    hier: &HierarchyResult,
    ordered: &OrderedHierarchy,
    started: &str,
    elapsed: Duration,
) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let mut w = BufWriter::new(file);

    write_header(
        &mut w,
        cfg,
        started,
        elapsed,
        trial.levels,
        ordered.top_module_count,
        trial.codelength,
        trial.one_level_codelength,
    )?;
    writeln!(w, "# path flow name node_id").map_err(|e| e.to_string())?;

    for &leaf in &ordered.leaf_order {
        let path_s = path_to_string(&ordered.leaf_label_paths[leaf]);
        let flow_s = fmt_sig(graph.node_data[leaf].flow, 6);
        let name_s = graph.node_name_or_id(leaf);
        writeln!(
            w,
            "{} {} \"{}\" {}",
            path_s, flow_s, name_s, graph.node_ids[leaf]
        )
        .map_err(|e| e.to_string())?;
    }

    let module_count = hier.module_parent.len();
    let mut root_links: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    let mut module_links: Vec<BTreeMap<(u32, u32), f64>> = vec![BTreeMap::new(); module_count];

    for e in 0..graph.edge_count() {
        let s = graph.edge_source[e] as usize;
        let t = graph.edge_target[e] as usize;
        if s == t {
            continue;
        }

        let ps = &hier.leaf_paths[s];
        let pt = &hier.leaf_paths[t];
        let lcp = common_prefix_len(ps, pt);

        let parent = if lcp == 0 { ROOT_PARENT } else { ps[lcp - 1] };
        let src_child = if lcp < ps.len() {
            ordered.leaf_base + ps[lcp]
        } else {
            s as u32
        };
        let dst_child = if lcp < pt.len() {
            ordered.leaf_base + pt[lcp]
        } else {
            t as u32
        };

        if src_child == dst_child {
            continue;
        }

        let src_pos = if let Some(v) = ordered.lookup_child_pos(parent, src_child) {
            v
        } else {
            continue;
        };
        let dst_pos = if let Some(v) = ordered.lookup_child_pos(parent, dst_child) {
            v
        } else {
            continue;
        };

        let f = graph.edge_flow[e];
        if parent == ROOT_PARENT {
            *root_links.entry((src_pos, dst_pos)).or_insert(0.0) += f;
        } else {
            *module_links[parent as usize]
                .entry((src_pos, dst_pos))
                .or_insert(0.0) += f;
        }
    }

    writeln!(
        w,
        "*Links {}",
        if cfg.directed {
            "directed"
        } else {
            "undirected"
        }
    )
    .map_err(|e| e.to_string())?;
    writeln!(w, "#*Links path enterFlow exitFlow numEdges numChildren")
        .map_err(|e| e.to_string())?;

    writeln!(
        w,
        "*Links root 0 0 {} {}",
        root_links.len(),
        ordered.root_children.len()
    )
    .map_err(|e| e.to_string())?;
    for ((s, t), f) in &root_links {
        writeln!(w, "{} {} {}", s, t, fmt_sig(*f, 6)).map_err(|e| e.to_string())?;
    }

    let mut modules_by_path: Vec<usize> = (0..module_count).collect();
    modules_by_path.sort_unstable_by(|&a, &b| {
        ordered.module_label_paths[a].cmp(&ordered.module_label_paths[b])
    });

    for m in modules_by_path {
        let md = hier.module_data[m];
        let links = &module_links[m];
        let num_children = ordered.module_child_range(m).len();
        let path_s = path_to_string(&ordered.module_label_paths[m]);

        writeln!(
            w,
            "*Links {} {} {} {} {}",
            path_s,
            fmt_sig(md.enter_flow, 6),
            fmt_sig(md.exit_flow, 6),
            links.len(),
            num_children
        )
        .map_err(|e| e.to_string())?;

        for ((s, t), f) in links {
            writeln!(w, "{} {} {}", s, t, fmt_sig(*f, 6)).map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())
}

pub fn write_outputs(
    cfg: &Config,
    graph: &Graph,
    trial: &TrialResult,
    start_time: SystemTime,
    elapsed: Duration,
) -> Result<Vec<PathBuf>, String> {
    if !cfg.any_output_enabled() {
        return Ok(Vec::new());
    }

    fs::create_dir_all(&cfg.out_dir).map_err(|e| {
        format!(
            "Can't write to directory '{}': {}",
            cfg.out_dir.display(),
            e
        )
    })?;

    let started = format_started_at(start_time);
    let base = output_base_path(cfg);
    let mut written = Vec::new();

    if let Some(hier) = trial.hierarchy.as_ref() {
        let ordered = build_ordered_hierarchy(graph, hier);

        if cfg.print_tree {
            let p = base.with_extension("tree");
            write_tree_file_multilevel(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
            written.push(p);
        }
        if cfg.print_clu {
            let p = base.with_extension("clu");
            write_clu_file_multilevel(&p, cfg, graph, trial, hier, &ordered, &started, elapsed)?;
            written.push(p);
        }
        if cfg.print_ftree {
            let p = base.with_extension("ftree");
            write_ftree_file_multilevel(&p, cfg, graph, trial, hier, &ordered, &started, elapsed)?;
            written.push(p);
        }
    } else {
        let ordered = build_ordered_modules(graph, trial);

        if cfg.print_tree {
            let p = base.with_extension("tree");
            write_tree_file_two_level(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
            written.push(p);
        }
        if cfg.print_clu {
            let p = base.with_extension("clu");
            write_clu_file_two_level(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
            written.push(p);
        }
        if cfg.print_ftree {
            let p = base.with_extension("ftree");
            write_ftree_file_two_level(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
            written.push(p);
        }
    }

    Ok(written)
}
