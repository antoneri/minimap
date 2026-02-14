use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct VertexInput {
    pub name: Option<String>,
    pub weight: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ParsedNetwork {
    pub vertices: FxHashMap<u32, VertexInput>,
    pub links: FxHashMap<(u32, u32), f64>,
}

enum Section {
    Links,
    Vertices,
    Ignore,
}

fn first_word_lower(line: &str) -> String {
    line.split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn parse_link_line(line: &str) -> Result<(u32, u32, f64), String> {
    let mut parts = line.split_whitespace();
    let s = parts
        .next()
        .ok_or_else(|| format!("Can't parse link data from line '{}'", line))?
        .parse::<u32>()
        .map_err(|_| format!("Can't parse link data from line '{}'", line))?;
    let t = parts
        .next()
        .ok_or_else(|| format!("Can't parse link data from line '{}'", line))?
        .parse::<u32>()
        .map_err(|_| format!("Can't parse link data from line '{}'", line))?;
    let w = match parts.next() {
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| format!("Can't parse link weight from line '{}'", line))?,
        None => 1.0,
    };
    Ok((s, t, w))
}

fn parse_vertex_line(line: &str) -> Result<(u32, Option<String>, f64), String> {
    let mut it = line.split_whitespace();
    let id = it
        .next()
        .ok_or_else(|| format!("Can't parse node id from line '{}'", line))?
        .parse::<u32>()
        .map_err(|_| format!("Can't parse node id from line '{}'", line))?;

    let mut name: Option<String> = None;
    let mut weight = 1.0;

    let quote_start = line.find('"');
    let quote_end = line.rfind('"');

    if let (Some(qs), Some(qe)) = (quote_start, quote_end) {
        if qs < qe {
            name = Some(line[qs + 1..qe].to_string());
            let rest = line[qe + 1..].trim();
            if !rest.is_empty() {
                weight = rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| format!("Can't parse vertex data from line '{}'", line))?
                    .parse::<f64>()
                    .map_err(|_| format!("Can't parse vertex weight from line '{}'", line))?;
            }
            if weight < 0.0 {
                return Err(format!("Negative node weight ({}) from line '{}'", weight, line));
            }
            return Ok((id, name, weight));
        }
    }

    if let Some(n) = it.next() {
        name = Some(n.to_string());
    }
    if let Some(w) = it.next() {
        weight = w
            .parse::<f64>()
            .map_err(|_| format!("Can't parse vertex weight from line '{}'", line))?;
        if weight < 0.0 {
            return Err(format!("Negative node weight ({}) from line '{}'", weight, line));
        }
    }

    Ok((id, name, weight))
}

pub fn parse_network_file(path: &Path) -> Result<ParsedNetwork, String> {
    let file = File::open(path)
        .map_err(|e| format!("Error opening file '{}': {}", path.display(), e))?;
    let reader = BufReader::new(file);

    let mut parsed = ParsedNetwork::default();
    let mut section = Section::Links;

    for line_res in reader.lines() {
        let line = line_res
            .map_err(|e| format!("Error reading file '{}': {}", path.display(), e))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('*') {
            let heading = first_word_lower(line);
            section = match heading.as_str() {
                "*vertices" => Section::Vertices,
                "*edges" | "*arcs" | "*links" => Section::Links,
                _ => Section::Ignore,
            };
            continue;
        }

        match section {
            Section::Ignore => {}
            Section::Vertices => {
                let (id, name, weight) = parse_vertex_line(line)?;
                let entry = parsed.vertices.entry(id).or_default();
                entry.weight = weight;
                if let Some(n) = name {
                    entry.name = Some(n);
                }
            }
            Section::Links => {
                let (s, t, w) = parse_link_line(line)?;
                if w <= 0.0 {
                    continue;
                }
                *parsed.links.entry((s, t)).or_insert(0.0) += w;
                parsed.vertices.entry(s).or_insert_with(|| VertexInput {
                    name: None,
                    weight: 1.0,
                });
                parsed.vertices.entry(t).or_insert_with(|| VertexInput {
                    name: None,
                    weight: 1.0,
                });
            }
        }
    }

    if parsed.vertices.is_empty() {
        return Err("Network is empty".to_string());
    }

    Ok(parsed)
}
