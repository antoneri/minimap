use crate::graph::FlowData;

#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaFlow {
    pub module: u32,
    pub delta_exit: f64,
    pub delta_enter: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MoveDeltaContext {
    // Cached old-module terms reused across candidate destinations for one moved node.
    de_old: f64,
    current_flow: f64,
    current_enter_flow: f64,
    current_exit_flow: f64,
    old_enter_before: f64,
    old_enter_after: f64,
    old_exit_before: f64,
    old_exit_after: f64,
    old_total_before: f64,
    old_total_after: f64,
}

#[inline]
pub fn plogp(p: f64) -> f64 {
    if p > 0.0 { p * p.log2() } else { 0.0 }
}

#[derive(Debug, Clone)]
pub struct MapEquationObjective {
    pub codelength: f64,
    pub index_codelength: f64,
    pub module_codelength: f64,

    node_flow_log_node_flow: f64,
    flow_log_flow: f64,
    exit_log_exit: f64,
    enter_log_enter: f64,
    enter_flow: f64,
    enter_flow_log_enter_flow: f64,

    exit_network_flow_log_exit_network_flow: f64,
}

impl MapEquationObjective {
    pub fn new(node_data: &[FlowData]) -> Self {
        Self::from_flowdata_iter(node_data.iter())
    }

    pub fn from_flowdata_iter<'a, I>(node_data: I) -> Self
    where
        I: IntoIterator<Item = &'a FlowData>,
    {
        let mut node_flow_log_node_flow = 0.0;
        for n in node_data {
            node_flow_log_node_flow += plogp(n.flow);
        }
        Self {
            codelength: 0.0,
            index_codelength: 0.0,
            module_codelength: 0.0,
            node_flow_log_node_flow,
            flow_log_flow: 0.0,
            exit_log_exit: 0.0,
            enter_log_enter: 0.0,
            enter_flow: 0.0,
            enter_flow_log_enter_flow: 0.0,
            exit_network_flow_log_exit_network_flow: 0.0,
        }
    }

    pub fn init_partition(&mut self, module_data: &[FlowData], module_indices: &[u32]) {
        self.flow_log_flow = 0.0;
        self.exit_log_exit = 0.0;
        self.enter_log_enter = 0.0;
        self.enter_flow = 0.0;

        for &m in module_indices {
            let d = module_data[m as usize];
            self.flow_log_flow += plogp(d.flow + d.exit_flow);
            self.enter_log_enter += plogp(d.enter_flow);
            self.exit_log_exit += plogp(d.exit_flow);
            self.enter_flow += d.enter_flow;
        }

        self.enter_flow_log_enter_flow = plogp(self.enter_flow);
        self.index_codelength = self.enter_flow_log_enter_flow
            - self.enter_log_enter
            - self.exit_network_flow_log_exit_network_flow;
        self.module_codelength =
            -self.exit_log_exit + self.flow_log_flow - self.node_flow_log_node_flow;
        self.codelength = self.index_codelength + self.module_codelength;
    }

    pub fn get_delta_on_move(
        &self,
        current: &FlowData,
        old_delta: &DeltaFlow,
        new_delta: &DeltaFlow,
        module_data: &[FlowData],
    ) -> f64 {
        let context = self.prepare_move_context(current, old_delta, module_data);
        self.get_delta_on_move_with_context(&context, new_delta, module_data)
    }

    pub(crate) fn prepare_move_context(
        &self,
        current: &FlowData,
        old_delta: &DeltaFlow,
        module_data: &[FlowData],
    ) -> MoveDeltaContext {
        let old_m = old_delta.module as usize;
        let de_old = old_delta.delta_enter + old_delta.delta_exit;

        let old_enter_before = plogp(module_data[old_m].enter_flow);
        let old_enter_after = plogp(module_data[old_m].enter_flow - current.enter_flow + de_old);
        let old_exit_before = plogp(module_data[old_m].exit_flow);
        let old_exit_after = plogp(module_data[old_m].exit_flow - current.exit_flow + de_old);
        let old_total_before = plogp(module_data[old_m].exit_flow + module_data[old_m].flow);
        let old_total_after = plogp(
            module_data[old_m].exit_flow + module_data[old_m].flow
                - current.exit_flow
                - current.flow
                + de_old,
        );

        MoveDeltaContext {
            de_old,
            current_flow: current.flow,
            current_enter_flow: current.enter_flow,
            current_exit_flow: current.exit_flow,
            old_enter_before,
            old_enter_after,
            old_exit_before,
            old_exit_after,
            old_total_before,
            old_total_after,
        }
    }

    pub(crate) fn get_delta_on_move_with_context(
        &self,
        context: &MoveDeltaContext,
        new_delta: &DeltaFlow,
        module_data: &[FlowData],
    ) -> f64 {
        let new_m = new_delta.module as usize;
        let de_new = new_delta.delta_enter + new_delta.delta_exit;

        let delta_enter =
            plogp(self.enter_flow + context.de_old - de_new) - self.enter_flow_log_enter_flow;

        let delta_enter_log_enter = -context.old_enter_before
            - plogp(module_data[new_m].enter_flow)
            + context.old_enter_after
            + plogp(module_data[new_m].enter_flow + context.current_enter_flow - de_new);

        let delta_exit_log_exit = -context.old_exit_before - plogp(module_data[new_m].exit_flow)
            + context.old_exit_after
            + plogp(module_data[new_m].exit_flow + context.current_exit_flow - de_new);

        let delta_flow_log_flow = -context.old_total_before
            - plogp(module_data[new_m].exit_flow + module_data[new_m].flow)
            + context.old_total_after
            + plogp(
                module_data[new_m].exit_flow
                    + module_data[new_m].flow
                    + context.current_exit_flow
                    + context.current_flow
                    - de_new,
            );

        delta_enter - delta_enter_log_enter - delta_exit_log_exit + delta_flow_log_flow
    }

    pub fn update_on_move(
        &mut self,
        current: &FlowData,
        old_delta: &DeltaFlow,
        new_delta: &DeltaFlow,
        module_data: &mut [FlowData],
    ) {
        let old_m = old_delta.module as usize;
        let new_m = new_delta.module as usize;

        let de_old = old_delta.delta_enter + old_delta.delta_exit;
        let de_new = new_delta.delta_enter + new_delta.delta_exit;

        self.enter_flow -= module_data[old_m].enter_flow + module_data[new_m].enter_flow;
        self.enter_log_enter -=
            plogp(module_data[old_m].enter_flow) + plogp(module_data[new_m].enter_flow);
        self.exit_log_exit -=
            plogp(module_data[old_m].exit_flow) + plogp(module_data[new_m].exit_flow);
        self.flow_log_flow -= plogp(module_data[old_m].exit_flow + module_data[old_m].flow)
            + plogp(module_data[new_m].exit_flow + module_data[new_m].flow);

        module_data[old_m].sub_assign(current);
        module_data[new_m].add_assign(current);

        module_data[old_m].enter_flow += de_old;
        module_data[old_m].exit_flow += de_old;
        module_data[new_m].enter_flow -= de_new;
        module_data[new_m].exit_flow -= de_new;

        self.enter_flow += module_data[old_m].enter_flow + module_data[new_m].enter_flow;
        self.enter_log_enter +=
            plogp(module_data[old_m].enter_flow) + plogp(module_data[new_m].enter_flow);
        self.exit_log_exit +=
            plogp(module_data[old_m].exit_flow) + plogp(module_data[new_m].exit_flow);
        self.flow_log_flow += plogp(module_data[old_m].exit_flow + module_data[old_m].flow)
            + plogp(module_data[new_m].exit_flow + module_data[new_m].flow);

        self.enter_flow_log_enter_flow = plogp(self.enter_flow);
        self.index_codelength = self.enter_flow_log_enter_flow
            - self.enter_log_enter
            - self.exit_network_flow_log_exit_network_flow;
        self.module_codelength =
            -self.exit_log_exit + self.flow_log_flow - self.node_flow_log_node_flow;
        self.codelength = self.index_codelength + self.module_codelength;
    }
}
