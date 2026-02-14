use minimap::flow::{FlowConfig, calculate_flow};
use std::env;
use std::process::ExitCode;
use std::time::Instant;

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let cfg = minimap::cli::parse_args(&args)?;

    let start_system = std::time::SystemTime::now();
    let start = Instant::now();

    let parsed = minimap::parser::parse_network_file(&cfg.network_file)?;
    let mut graph = minimap::graph::Graph::from_parsed(parsed, cfg.directed)?;

    calculate_flow(
        &mut graph,
        FlowConfig {
            directed: cfg.directed,
        },
    );

    let trial = minimap::optimizer::run_trials(
        &graph,
        cfg.seed,
        cfg.num_trials,
        cfg.directed,
        cfg.trial_threads,
    );

    let elapsed = start.elapsed();
    let _written = minimap::output::write_outputs(&cfg, &graph, &trial, start_system, elapsed)?;

    if !cfg.silent {
        println!(
            "Partitioned into 2 levels with {} top modules, codelength {}",
            trial.num_modules,
            trial.codelength
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}
