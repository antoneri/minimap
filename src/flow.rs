use crate::graph::Graph;

const TELEPORT_PROBABILITY: f64 = 0.15;

#[derive(Debug, Clone, Copy)]
pub struct FlowConfig {
    pub directed: bool,
}

fn normalize(v: &mut [f64]) {
    let sum: f64 = v.iter().sum();
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

pub fn calculate_flow(graph: &mut Graph, cfg: FlowConfig) {
    let n = graph.node_count();
    let m = graph.edge_count();

    let mut node_flow = vec![0.0f64; n];
    let mut node_teleport_weights = vec![0.0f64; n];
    let mut node_out_degree = vec![0usize; n];
    let mut sum_link_out_weight = vec![0.0f64; n];

    for e in 0..m {
        let s = graph.edge_source[e] as usize;
        let t = graph.edge_target[e] as usize;
        let w = graph.edge_weight[e];

        node_out_degree[s] += 1;
        sum_link_out_weight[s] += w;
        node_flow[s] += w / graph.sum_weighted_degree;

        if s != t {
            if !cfg.directed {
                node_out_degree[t] += 1;
                sum_link_out_weight[t] += w;
            }
            node_flow[t] += w / graph.sum_weighted_degree;
        }
    }

    if cfg.directed {
        for e in 0..m {
            let s = graph.edge_source[e] as usize;
            node_teleport_weights[s] += graph.edge_weight[e] / graph.sum_link_weight;
        }
        normalize(&mut node_teleport_weights);

        for e in 0..m {
            let s = graph.edge_source[e] as usize;
            if sum_link_out_weight[s] > 0.0 {
                graph.edge_flow[e] = graph.edge_weight[e] / sum_link_out_weight[s];
            } else {
                graph.edge_flow[e] = 0.0;
            }
        }

        let mut node_flow_tmp = vec![0.0f64; n];

        let mut alpha = TELEPORT_PROBABILITY;
        let mut beta = 1.0 - alpha;
        let mut iterations = 0usize;
        let mut err = 0.0f64;
        let dangling_rank_last = loop {
            let old_err = err;

            let mut dangling_rank = 0.0;
            for i in 0..n {
                if node_out_degree[i] == 0 {
                    dangling_rank += node_flow[i];
                }
            }

            let tele_flow = alpha + beta * dangling_rank;
            for i in 0..n {
                node_flow_tmp[i] = tele_flow * node_teleport_weights[i];
            }

            for e in 0..m {
                let s = graph.edge_source[e] as usize;
                let t = graph.edge_target[e] as usize;
                node_flow_tmp[t] += beta * graph.edge_flow[e] * node_flow[s];
            }

            let mut node_flow_diff = -1.0f64;
            err = 0.0;
            for i in 0..n {
                node_flow_diff += node_flow_tmp[i];
                err += (node_flow_tmp[i] - node_flow[i]).abs();
            }

            node_flow.copy_from_slice(&node_flow_tmp);

            if node_flow_diff.abs() > 1.0e-10 {
                let denom = node_flow_diff + 1.0;
                if denom != 0.0 {
                    for x in node_flow.iter_mut() {
                        *x /= denom;
                    }
                }
            }

            // Match Infomap perturbation behavior to avoid getting stuck in equilibrium.
            if (err - old_err).abs() < 1.0e-17 {
                alpha += 1.0e-12;
                beta = 1.0 - alpha;
            }

            iterations += 1;
            if iterations >= 200 || (err <= 1.0e-15 && iterations >= 50) {
                break dangling_rank;
            }
        };

        let mut sum_node_rank = 1.0 - dangling_rank_last;
        if sum_node_rank <= 0.0 {
            sum_node_rank = 1.0;
        }

        for x in node_flow.iter_mut() {
            *x = 0.0;
        }
        beta = 1.0;
        for e in 0..m {
            let s = graph.edge_source[e] as usize;
            let t = graph.edge_target[e] as usize;
            let f = graph.edge_flow[e] * beta * node_flow_tmp[s] / sum_node_rank;
            graph.edge_flow[e] = f;
            node_flow[t] += f;
        }
    } else {
        for e in 0..m {
            let s = graph.edge_source[e] as usize;
            let t = graph.edge_target[e] as usize;
            let mut f = graph.edge_weight[e] / graph.sum_weighted_degree;
            if s != t {
                f *= 2.0;
            }
            graph.edge_flow[e] = f;
        }
    }

    finalize(graph, cfg, &node_flow, &node_teleport_weights, &node_out_degree);
}

fn finalize(
    graph: &mut Graph,
    cfg: FlowConfig,
    node_flow: &[f64],
    node_teleport_weights: &[f64],
    node_out_degree: &[usize],
) {
    let n = graph.node_count();

    for i in 0..n {
        let tele_w = if cfg.directed { node_teleport_weights[i] } else { 0.0 };
        let tele_flow = node_flow[i]
            * if node_out_degree[i] == 0 {
                1.0
            } else {
                TELEPORT_PROBABILITY
            };

        graph.nodes[i].data.flow = node_flow[i];
        graph.nodes[i].data.teleport_weight = tele_w;
        graph.nodes[i].data.teleport_flow = tele_flow;
        graph.nodes[i].data.enter_flow = node_flow[i];
        graph.nodes[i].data.exit_flow = node_flow[i];
        graph.nodes[i].data.dangling_flow = if node_out_degree[i] == 0 { node_flow[i] } else { 0.0 };

        graph.nodes[i].data.enter_flow -= tele_flow * tele_w;
        graph.nodes[i].data.exit_flow -= tele_flow * tele_w;

        let norm = if cfg.directed { 1.0 } else { 2.0 };
        for e in graph.out_range(i) {
            let s = graph.edge_source[e] as usize;
            let t = graph.edge_target[e] as usize;
            if s == t {
                graph.nodes[i].data.enter_flow -= graph.edge_flow[e] / norm;
                graph.nodes[i].data.exit_flow -= graph.edge_flow[e] / norm;
                break;
            }
        }
    }

    if cfg.directed {
        let mut enter_flow = vec![0.0f64; n];
        let mut exit_flow = vec![0.0f64; n];

        for e in 0..graph.edge_count() {
            let s = graph.edge_source[e] as usize;
            let t = graph.edge_target[e] as usize;
            let f = graph.edge_flow[e];
            exit_flow[s] += f;
            enter_flow[t] += f;
        }

        for i in 0..n {
            graph.nodes[i].data.enter_flow = enter_flow[i];
            graph.nodes[i].data.exit_flow = exit_flow[i];
        }
    }
}
