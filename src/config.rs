use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub raw_args: String,
    pub network_file: PathBuf,
    pub out_dir: PathBuf,
    pub out_name: String,
    pub directed: bool,
    pub multilevel: bool,
    pub seed: u32,
    pub num_trials: u32,
    pub trial_threads: Option<usize>,
    pub parity_rng: bool,
    pub silent: bool,
    pub print_tree: bool,
    pub print_clu: bool,
    pub print_ftree: bool,
    pub clu_level: i32,
}

impl Config {
    pub fn any_output_enabled(&self) -> bool {
        self.print_tree || self.print_clu || self.print_ftree
    }
}
