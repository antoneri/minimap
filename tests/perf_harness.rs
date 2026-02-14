use std::process::Command;
use std::time::Instant;

const INFOMAP_BIN: &str = "/Users/anton/kod/infomap/Infomap";
const NETWORK_W: &str = "/Users/anton/kod/infomap/examples/networks/modular_w.net";

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

#[test]
#[ignore = "manual perf gate harness"]
fn compare_runtime_gate() {
    let mini_bin = std::env::var("CARGO_BIN_EXE_minimap").expect("missing minimap bin");

    let runs = 5;
    let mut oracle = Vec::with_capacity(runs);
    let mut cand = Vec::with_capacity(runs);

    for _ in 0..runs {
        let out_oracle = tempfile::tempdir().unwrap();
        let t0 = Instant::now();
        let status = Command::new(INFOMAP_BIN)
            .args([
                "--silent",
                "--seed",
                "123",
                "--num-trials",
                "1",
                "--two-level",
                "--tree",
                NETWORK_W,
                out_oracle.path().to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
        oracle.push(t0.elapsed().as_secs_f64());

        let out_cand = tempfile::tempdir().unwrap();
        let t1 = Instant::now();
        let status = Command::new(&mini_bin)
            .args([
                "--silent",
                "--seed",
                "123",
                "--num-trials",
                "1",
                "--two-level",
                "--tree",
                NETWORK_W,
                out_cand.path().to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
        cand.push(t1.elapsed().as_secs_f64());
    }

    let mo = median(oracle);
    let mc = median(cand);
    println!("oracle median: {mo:.6}s, candidate median: {mc:.6}s, ratio: {:.3}", mc / mo);
}
