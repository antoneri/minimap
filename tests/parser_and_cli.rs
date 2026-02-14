use std::fs::File;
use std::io::Write;

#[test]
fn parse_basic_link_list_and_vertices() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.net");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "*Vertices").unwrap();
    writeln!(f, "1 \"A\" 1").unwrap();
    writeln!(f, "2 \"B\" 1").unwrap();
    writeln!(f, "*Links").unwrap();
    writeln!(f, "1 2 2.5").unwrap();
    writeln!(f, "1 2 0.5").unwrap();

    let parsed = minimap::parser::parse_network_file(&path).unwrap();
    assert_eq!(parsed.vertices.len(), 2);
    assert_eq!(parsed.links.len(), 1);
    assert!((parsed.links[&(1, 2)] - 3.0).abs() < 1e-12);
}

#[test]
fn cli_defaults_to_tree_output() {
    let args = vec![
        "in.net".to_string(),
        "out".to_string(),
        "--silent".to_string(),
    ];
    let cfg = minimap::cli::parse_args(&args).unwrap();
    assert!(cfg.print_tree);
    assert!(!cfg.print_clu);
    assert!(!cfg.print_ftree);
}
