use crate::parser::ParsedNetwork;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, Default)]
pub struct FlowData {
    pub flow: f64,
    pub enter_flow: f64,
    pub exit_flow: f64,
    pub teleport_flow: f64,
    pub teleport_weight: f64,
    pub dangling_flow: f64,
}

impl FlowData {
    pub fn add_assign(&mut self, other: &FlowData) {
        self.flow += other.flow;
        self.enter_flow += other.enter_flow;
        self.exit_flow += other.exit_flow;
        self.teleport_flow += other.teleport_flow;
        self.teleport_weight += other.teleport_weight;
        self.dangling_flow += other.dangling_flow;
    }

    pub fn sub_assign(&mut self, other: &FlowData) {
        self.flow -= other.flow;
        self.enter_flow -= other.enter_flow;
        self.exit_flow -= other.exit_flow;
        self.teleport_flow -= other.teleport_flow;
        self.teleport_weight -= other.teleport_weight;
        self.dangling_flow -= other.dangling_flow;
    }
}

#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: u32,
    pub name: Option<String>,
    pub input_weight: f64,
    pub data: FlowData,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<NodeRecord>,
    pub edge_source: Vec<u32>,
    pub edge_target: Vec<u32>,
    pub edge_weight: Vec<f64>,
    pub edge_flow: Vec<f64>,
    pub out_offsets: Vec<u32>,
    pub in_offsets: Vec<u32>,
    pub in_edge_idx: Vec<u32>,
    pub sum_link_weight: f64,
    pub sum_weighted_degree: f64,
    pub self_link_weight: f64,
}

impl Graph {
    pub fn from_parsed(parsed: ParsedNetwork, directed_flow: bool) -> Result<Self, String> {
        let mut node_ids: Vec<u32> = parsed.vertices.keys().copied().collect();
        node_ids.sort_unstable();

        let mut id_to_idx: FxHashMap<u32, u32> = FxHashMap::default();
        id_to_idx.reserve(node_ids.len());
        for (idx, id) in node_ids.iter().copied().enumerate() {
            id_to_idx.insert(id, idx as u32);
        }

        let mut nodes = Vec::with_capacity(node_ids.len());
        for id in node_ids.iter().copied() {
            let v = parsed.vertices.get(&id).ok_or_else(|| {
                format!("Missing vertex metadata for node {}", id)
            })?;
            nodes.push(NodeRecord {
                id,
                name: v.name.clone(),
                input_weight: v.weight,
                data: FlowData::default(),
            });
        }

        let mut edges: Vec<(u32, u32, f64)> = Vec::with_capacity(parsed.links.len());
        for ((s_id, t_id), w) in parsed.links {
            if w <= 0.0 {
                continue;
            }
            let s = *id_to_idx
                .get(&s_id)
                .ok_or_else(|| format!("Unknown source node id {}", s_id))?;
            let t = *id_to_idx
                .get(&t_id)
                .ok_or_else(|| format!("Unknown target node id {}", t_id))?;
            edges.push((s, t, w));
        }

        edges.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
        });

        let m = edges.len();
        let n = nodes.len();
        let mut edge_source = Vec::with_capacity(m);
        let mut edge_target = Vec::with_capacity(m);
        let mut edge_weight = Vec::with_capacity(m);
        let mut edge_flow = vec![0.0; m];

        let mut out_counts = vec![0u32; n];
        let mut in_counts = vec![0u32; n];

        let mut sum_link_weight = 0.0;
        let mut self_link_weight = 0.0;

        for (s, t, w) in edges.iter().copied() {
            edge_source.push(s);
            edge_target.push(t);
            edge_weight.push(w);
            out_counts[s as usize] += 1;
            in_counts[t as usize] += 1;
            sum_link_weight += w;
            if s == t {
                self_link_weight += w;
            }
        }

        let mut out_offsets = vec![0u32; n + 1];
        for i in 0..n {
            out_offsets[i + 1] = out_offsets[i] + out_counts[i];
        }

        let mut in_offsets = vec![0u32; n + 1];
        for i in 0..n {
            in_offsets[i + 1] = in_offsets[i] + in_counts[i];
        }

        let mut in_fill = vec![0u32; n];
        let mut in_edge_idx = vec![0u32; m];
        for e in 0..m {
            let t = edge_target[e] as usize;
            let pos = in_offsets[t] + in_fill[t];
            in_edge_idx[pos as usize] = e as u32;
            in_fill[t] += 1;
        }

        let sum_weighted_degree = if directed_flow {
            2.0 * sum_link_weight
        } else {
            2.0 * sum_link_weight - self_link_weight
        };

        Ok(Self {
            nodes,
            edge_source,
            edge_target,
            edge_weight,
            edge_flow: {
                edge_flow.shrink_to_fit();
                edge_flow
            },
            out_offsets,
            in_offsets,
            in_edge_idx,
            sum_link_weight,
            sum_weighted_degree,
            self_link_weight,
        })
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_source.len()
    }

    #[inline]
    pub fn out_range(&self, node_idx: usize) -> std::ops::Range<usize> {
        self.out_offsets[node_idx] as usize..self.out_offsets[node_idx + 1] as usize
    }

    #[inline]
    pub fn in_range(&self, node_idx: usize) -> std::ops::Range<usize> {
        self.in_offsets[node_idx] as usize..self.in_offsets[node_idx + 1] as usize
    }

    #[inline]
    pub fn out_degree(&self, node_idx: usize) -> usize {
        (self.out_offsets[node_idx + 1] - self.out_offsets[node_idx]) as usize
    }

    pub fn node_name_or_id(&self, node_idx: usize) -> String {
        self.nodes[node_idx]
            .name
            .clone()
            .unwrap_or_else(|| self.nodes[node_idx].id.to_string())
    }
}
