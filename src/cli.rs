use crate::config::Config;
use std::path::PathBuf;

fn parse_u32(s: &str) -> Option<u32> {
    s.parse::<u32>().ok()
}

fn parse_i32(s: &str) -> Option<i32> {
    s.parse::<i32>().ok()
}

fn parse_usize(s: &str) -> Option<usize> {
    s.parse::<usize>().ok()
}

fn parse_bool_output(list: &str, tree: &mut bool, clu: &mut bool, ftree: &mut bool) {
    for token in list.split(',') {
        match token.trim() {
            "tree" => *tree = true,
            "clu" => *clu = true,
            "ftree" => *ftree = true,
            _ => {}
        }
    }
}

pub fn parse_args(args: &[String]) -> Result<Config, String> {
    let raw_args = args.join(" ");

    let mut network_file: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;

    let mut directed = false;
    let mut seed = 123u32;
    let mut num_trials = 1u32;
    let mut trial_threads: Option<usize> = None;
    let mut parity_rng = false;
    let mut silent = false;
    let mut print_tree = false;
    let mut print_clu = false;
    let mut print_ftree = false;
    let mut out_name: Option<String> = None;
    let mut clu_level = 1i32;

    let mut i = 0usize;
    while i < args.len() {
        let tok = &args[i];

        if let Some(rest) = tok.strip_prefix("--seed=") {
            if let Some(v) = parse_u32(rest) {
                seed = v;
            }
            i += 1;
            continue;
        }
        if let Some(rest) = tok.strip_prefix("--num-trials=") {
            if let Some(v) = parse_u32(rest) {
                num_trials = v.max(1);
            }
            i += 1;
            continue;
        }
        if let Some(rest) = tok.strip_prefix("--threads=") {
            if let Some(v) = parse_usize(rest) {
                if v > 0 {
                    trial_threads = Some(v);
                }
            }
            i += 1;
            continue;
        }
        if let Some(rest) = tok.strip_prefix("--out-name=") {
            out_name = Some(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = tok.strip_prefix("--clu-level=") {
            if let Some(v) = parse_i32(rest) {
                clu_level = v;
            }
            i += 1;
            continue;
        }
        if let Some(rest) = tok.strip_prefix("--output=") {
            parse_bool_output(rest, &mut print_tree, &mut print_clu, &mut print_ftree);
            i += 1;
            continue;
        }

        match tok.as_str() {
            "--directed" => {
                directed = true;
                i += 1;
            }
            "--two-level" => {
                i += 1;
            }
            "--seed" => {
                if let Some(next) = args.get(i + 1) {
                    if let Some(v) = parse_u32(next) {
                        seed = v;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--num-trials" => {
                if let Some(next) = args.get(i + 1) {
                    if let Some(v) = parse_u32(next) {
                        num_trials = v.max(1);
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--threads" => {
                if let Some(next) = args.get(i + 1) {
                    if let Some(v) = parse_usize(next) {
                        if v > 0 {
                            trial_threads = Some(v);
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--parity-rng" => {
                parity_rng = true;
                i += 1;
            }
            "--tree" => {
                print_tree = true;
                i += 1;
            }
            "--clu" => {
                print_clu = true;
                i += 1;
            }
            "--ftree" => {
                print_ftree = true;
                i += 1;
            }
            "--output" => {
                if let Some(next) = args.get(i + 1) {
                    parse_bool_output(next, &mut print_tree, &mut print_clu, &mut print_ftree);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "-o" => {
                if let Some(next) = args.get(i + 1) {
                    parse_bool_output(next, &mut print_tree, &mut print_clu, &mut print_ftree);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--out-name" => {
                if let Some(next) = args.get(i + 1) {
                    out_name = Some(next.clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--silent" => {
                silent = true;
                i += 1;
            }
            "--clu-level" => {
                if let Some(next) = args.get(i + 1) {
                    if let Some(v) = parse_i32(next) {
                        clu_level = v;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ if tok.starts_with('-') => {
                i += 1;
            }
            _ => {
                if network_file.is_none() {
                    network_file = Some(PathBuf::from(tok));
                } else if out_dir.is_none() {
                    out_dir = Some(PathBuf::from(tok));
                }
                i += 1;
            }
        }
    }

    let network_file = network_file
        .ok_or_else(|| "Usage: minimap network_file out_directory [options]".to_string())?;
    let out_dir =
        out_dir.ok_or_else(|| "Usage: minimap network_file out_directory [options]".to_string())?;

    let out_name = match out_name {
        Some(v) => v,
        None => network_file
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("no-name")
            .to_string(),
    };

    if !print_tree && !print_clu && !print_ftree {
        print_tree = true;
    }

    Ok(Config {
        raw_args,
        network_file,
        out_dir,
        out_name,
        directed,
        seed,
        num_trials,
        trial_threads,
        parity_rng,
        silent,
        print_tree,
        print_clu,
        print_ftree,
        clu_level,
    })
}
