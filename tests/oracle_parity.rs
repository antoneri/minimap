use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const INFOMAP_BIN: &str = "/Users/anton/kod/infomap/Infomap";
const NETWORK_W: &str = "/Users/anton/kod/infomap/examples/networks/modular_w.net";
const NETWORK_WD: &str = "/Users/anton/kod/infomap/examples/networks/modular_wd.net";

fn normalize(content: &str) -> String {
    let re_started = Regex::new(r"(?m)^# started at .*$").unwrap();
    let re_completed = Regex::new(r"(?m)^# completed in .* s$").unwrap();
    let s = re_started.replace_all(content, "# started at <normalized>");
    let s = re_completed.replace_all(&s, "# completed in <normalized> s");
    s.into_owned()
}

fn run(cmd: &str, args: &[String]) {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {}: {}", cmd, e));
    assert!(status.success(), "command failed: {} {:?}", cmd, args);
}

fn compare_files(a: &Path, b: &Path) {
    let ca = fs::read_to_string(a).unwrap();
    let cb = fs::read_to_string(b).unwrap();
    assert_eq!(normalize(&ca), normalize(&cb), "mismatch in {:?} vs {:?}", a, b);
}

#[test]
#[ignore = "oracle parity harness; run manually during optimization"]
fn parity_against_infomap_samples() {
    let tmp = tempfile::tempdir().unwrap();
    let out_u = tmp.path().join("shared_u");
    fs::create_dir_all(&out_u).unwrap();

    let mini_bin = if let Ok(path) = std::env::var("CARGO_BIN_EXE_minimap") {
        PathBuf::from(path)
    } else {
        let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("debug")
            .join(if cfg!(windows) { "minimap.exe" } else { "minimap" });
        if !fallback.exists() {
            let status = Command::new("cargo")
                .args(["build", "--bin", "minimap"])
                .status()
                .expect("failed to invoke cargo build for minimap");
            assert!(status.success(), "cargo build --bin minimap failed");
        }
        fallback
    };

    let args_w = vec![
        "--silent".to_string(),
        "--seed".to_string(),
        "123".to_string(),
        "--num-trials".to_string(),
        "1".to_string(),
        "--two-level".to_string(),
        "--tree".to_string(),
        "--clu".to_string(),
        "--ftree".to_string(),
        NETWORK_W.to_string(),
        out_u.display().to_string(),
    ];
    run(INFOMAP_BIN, &args_w);

    for ext in ["tree", "clu", "ftree"] {
        fs::copy(
            out_u.join(format!("modular_w.{}", ext)),
            out_u.join(format!("oracle_modular_w.{}", ext)),
        )
        .unwrap();
    }

    run(&mini_bin.display().to_string(), &args_w);

    for ext in ["tree", "clu", "ftree"] {
        compare_files(
            &PathBuf::from(out_u.join(format!("oracle_modular_w.{}", ext))),
            &PathBuf::from(out_u.join(format!("modular_w.{}", ext))),
        );
    }

    let out_d = tmp.path().join("shared_d");
    fs::create_dir_all(&out_d).unwrap();

    let args_wd = vec![
        "--silent".to_string(),
        "--seed".to_string(),
        "123".to_string(),
        "--num-trials".to_string(),
        "1".to_string(),
        "--two-level".to_string(),
        "--directed".to_string(),
        "--tree".to_string(),
        "--clu".to_string(),
        "--ftree".to_string(),
        NETWORK_WD.to_string(),
        out_d.display().to_string(),
    ];
    run(INFOMAP_BIN, &args_wd);

    for ext in ["tree", "clu", "ftree"] {
        fs::copy(
            out_d.join(format!("modular_wd.{}", ext)),
            out_d.join(format!("oracle_modular_wd.{}", ext)),
        )
        .unwrap();
    }

    run(&mini_bin.display().to_string(), &args_wd);

    for ext in ["tree", "clu", "ftree"] {
        compare_files(
            &PathBuf::from(out_d.join(format!("oracle_modular_wd.{}", ext))),
            &PathBuf::from(out_d.join(format!("modular_wd.{}", ext))),
        );
    }
}
