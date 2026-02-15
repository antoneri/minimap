use crate::config::Config;
use crate::graph::Graph;
use crate::optimizer::TrialResult;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

const VERSION: &str = "2.8.1";

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
        "# partitioned into 2 levels with {} top modules",
        top_modules
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

fn write_tree_file(
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

fn write_clu_file(
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

fn write_ftree_file(
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

    let ordered = build_ordered_modules(graph, trial);
    let started = format_started_at(start_time);

    let base = output_base_path(cfg);
    let mut written = Vec::new();

    if cfg.print_tree {
        let p = base.with_extension("tree");
        write_tree_file(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
        written.push(p);
    }
    if cfg.print_clu {
        let p = base.with_extension("clu");
        write_clu_file(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
        written.push(p);
    }
    if cfg.print_ftree {
        let p = base.with_extension("ftree");
        write_ftree_file(&p, cfg, graph, trial, &ordered, &started, elapsed)?;
        written.push(p);
    }

    Ok(written)
}
