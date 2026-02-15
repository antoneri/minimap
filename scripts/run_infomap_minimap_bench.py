#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

INFOMAP_OMP_THREADS = "12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare minimap and Infomap on parity/performance scenarios."
    )
    parser.add_argument(
        "--minimap",
        default="/Users/anton/kod/minimap/target/release/minimap",
        help="Path to minimap binary.",
    )
    parser.add_argument(
        "--infomap",
        default="/Users/anton/kod/infomap/Infomap",
        help="Path to Infomap binary.",
    )
    parser.add_argument(
        "--out-dir",
        default=f"/tmp/minimap_bench_{int(time.time())}",
        help="Output directory for generated data and results.",
    )
    parser.add_argument(
        "--perf-reps",
        type=int,
        default=3,
        help="Repetitions for 100k performance cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used for generators and tool runs.",
    )
    parser.add_argument(
        "--strict-parity",
        action="store_true",
        help="Fail with non-zero exit code if parity matrix exceeds thresholds.",
    )
    parser.add_argument(
        "--parity-seeds",
        default="123,777,2024",
        help="Comma-separated seeds for strict 4-mode parity checks.",
    )
    parser.add_argument(
        "--parity-threshold-two-level",
        type=float,
        default=1e-9,
        help="Max allowed |codelength delta| for two-level parity checks.",
    )
    parser.add_argument(
        "--parity-threshold-multilevel",
        type=float,
        default=2e-3,
        help="Max allowed |codelength delta| for multilevel parity checks.",
    )
    parser.add_argument(
        "--parity-two-level-u-net",
        default="/Users/anton/kod/infomap/examples/networks/modular_w.net",
        help="Network for two-level undirected parity checks.",
    )
    parser.add_argument(
        "--parity-two-level-d-net",
        default="/Users/anton/kod/infomap/examples/networks/modular_wd.net",
        help="Network for two-level directed parity checks.",
    )
    parser.add_argument(
        "--parity-multilevel-u-net",
        default="/tmp/minimap_hier_u.net",
        help="Network for multilevel undirected parity checks.",
    )
    parser.add_argument(
        "--parity-multilevel-d-net",
        default="/tmp/minimap_hier_d.net",
        help="Network for multilevel directed parity checks.",
    )
    return parser.parse_args()


def parse_seed_list(seed_csv: str) -> list[int]:
    seeds: list[int] = []
    for token in seed_csv.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value '{token}' in --parity-seeds") from exc
    if not seeds:
        raise ValueError("No valid seeds provided in --parity-seeds")
    return seeds


def generate_perf_100k(path: Path, seed: int) -> None:
    random.seed(seed)
    n_nodes = 20_000
    n_edges = 100_000

    edges = set()
    while len(edges) < n_edges:
        a = random.randint(1, n_nodes)
        b = random.randint(1, n_nodes)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edges.add((a, b))

    with path.open("w") as f:
        for a, b in sorted(edges):
            f.write(f"{a} {b} 1\n")


def _bottom_group(i: int) -> int:
    return (i - 1) // 16


def _mid_group(i: int) -> int:
    return _bottom_group(i) // 8


def _top_group(i: int) -> int:
    return _mid_group(i) // 4


def generate_hierarchical(path: Path, seed: int, directed: bool) -> None:
    random.seed(seed)
    n = 2048

    p_same_bottom = 0.40
    p_same_mid = 0.08
    p_same_top = 0.02
    p_cross_top = 0.004

    with path.open("w") as f:
        if directed:
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i == j:
                        continue
                    if _bottom_group(i) == _bottom_group(j):
                        p = p_same_bottom
                    elif _mid_group(i) == _mid_group(j):
                        p = p_same_mid
                    elif _top_group(i) == _top_group(j):
                        p = p_same_top
                    else:
                        p = p_cross_top
                    if random.random() < p:
                        f.write(f"{i} {j} 1\n")
        else:
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    if _bottom_group(i) == _bottom_group(j):
                        p = p_same_bottom
                    elif _mid_group(i) == _mid_group(j):
                        p = p_same_mid
                    elif _top_group(i) == _top_group(j):
                        p = p_same_top
                    else:
                        p = p_cross_top
                    if random.random() < p:
                        f.write(f"{i} {j} 1\n")


def parse_tree(path: Path) -> dict:
    text = path.read_text()
    m = re.search(r"(?m)^# codelength ([0-9eE+.\-]+)(?: bits)?$", text)
    if not m:
        raise RuntimeError(f"No codelength header in {path}")
    codelength = float(m.group(1))

    prefixes = {}
    max_depth = 1
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        parts = tuple(int(x) for x in first.split(":"))
        max_depth = max(max_depth, len(parts))
        for d in range(1, len(parts)):
            prefixes.setdefault(d, set()).add(parts[:d])

    shape = [len(prefixes.get(d, set())) for d in range(1, max_depth)]
    top_modules = shape[0] if shape else 0
    return {
        "codelength": codelength,
        "depth": max_depth,
        "shape": shape,
        "top_modules": top_modules,
    }


def parse_time_output(stderr: str) -> tuple[float, float]:
    real_match = re.search(r"(?m)^\s*real\s+([0-9.]+)$", stderr)
    rss_match = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", stderr)
    if not rss_match:
        rss_match = re.search(r"(?m)^\s*maximum resident set size\s+(\d+)\s*$", stderr)
    if not real_match or not rss_match:
        raise RuntimeError(f"Could not parse /usr/bin/time output:\n{stderr}")
    wall = float(real_match.group(1))
    rss_raw = int(rss_match.group(1))
    # macOS /usr/bin/time -lp reports bytes.
    rss_kib = rss_raw / 1024.0
    return wall, rss_kib


def run_once(
    tool: str,
    infomap: Path,
    minimap: Path,
    network: Path,
    directed: bool,
    two_level: bool,
    seed: int,
    minimap_extra_flags: list[str] | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"bench_{tool}_") as tmp:
        out_dir = Path(tmp)
        if tool == "infomap":
            cmd = [
                str(infomap),
                "--silent",
                "--seed",
                str(seed),
                "--num-trials",
                "1",
            ]
            if directed:
                cmd.append("--directed")
            if two_level:
                cmd.append("--two-level")
            cmd.extend(["--tree", str(network), str(out_dir)])
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = INFOMAP_OMP_THREADS
        else:
            cmd = [str(minimap), str(network), str(out_dir)]
            if directed:
                cmd.append("--directed")
            cmd.append("--two-level" if two_level else "--multilevel")
            cmd.extend(
                [
                    "--seed",
                    str(seed),
                    "--num-trials",
                    "1",
                    "--threads",
                    "1",
                    "--tree",
                    "--silent",
                ]
            )
            if minimap_extra_flags:
                cmd.extend(minimap_extra_flags)
            env = None

        timed_cmd = ["/usr/bin/time", "-lp"] + cmd
        proc = subprocess.run(timed_cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        wall, rss_kib = parse_time_output(proc.stderr)
        tree_files = sorted(out_dir.glob("*.tree"))
        if len(tree_files) != 1:
            raise RuntimeError(f"Expected exactly one .tree in {out_dir}, found {tree_files}")
        tree = parse_tree(tree_files[0])
        tree["wall_time_s"] = wall
        tree["peak_rss_kib"] = rss_kib
        return tree


def run_case(
    case_name: str,
    network: Path,
    directed: bool,
    two_level: bool,
    reps: int,
    infomap: Path,
    minimap: Path,
    seed: int,
) -> dict:
    outputs = {}
    for tool in ("infomap", "minimap"):
        runs = [
            run_once(tool, infomap, minimap, network, directed, two_level, seed)
            for _ in range(reps)
        ]
        outputs[tool] = {
            "codelength": runs[0]["codelength"],
            "depth": runs[0]["depth"],
            "shape": runs[0]["shape"],
            "top_modules": runs[0]["top_modules"],
            "wall_time_s_median": statistics.median(r["wall_time_s"] for r in runs),
            "peak_rss_kib_median": statistics.median(r["peak_rss_kib"] for r in runs),
            "runs": runs,
        }

    outputs["delta"] = {
        "codelength_minimap_minus_infomap": outputs["minimap"]["codelength"]
        - outputs["infomap"]["codelength"],
        "wall_time_ratio_minimap_over_infomap": outputs["minimap"]["wall_time_s_median"]
        / outputs["infomap"]["wall_time_s_median"],
        "rss_ratio_minimap_over_infomap": outputs["minimap"]["peak_rss_kib_median"]
        / outputs["infomap"]["peak_rss_kib_median"],
    }
    outputs["case"] = case_name
    outputs["directed"] = directed
    outputs["mode"] = "two-level" if two_level else "multilevel"
    outputs["repetitions"] = reps
    return outputs


def run_parity_matrix(
    infomap: Path,
    minimap: Path,
    two_level_u_net: Path,
    two_level_d_net: Path,
    multilevel_u_net: Path,
    multilevel_d_net: Path,
    seeds: list[int],
    threshold_two_level: float,
    threshold_multilevel: float,
) -> dict:
    modes = [
        ("two-level", False, True, two_level_u_net, threshold_two_level),
        ("multilevel", False, False, multilevel_u_net, threshold_multilevel),
        ("two-level", True, True, two_level_d_net, threshold_two_level),
        ("multilevel", True, False, multilevel_d_net, threshold_multilevel),
    ]

    groups = []
    overall_pass = True

    for mode_name, directed, two_level, network, threshold in modes:
        per_seed = []
        deltas = []
        for seed in seeds:
            infomap_run = run_once(
                "infomap",
                infomap,
                minimap,
                network,
                directed,
                two_level,
                seed,
            )
            minimap_run = run_once(
                "minimap",
                infomap,
                minimap,
                network,
                directed,
                two_level,
                seed,
                minimap_extra_flags=["--parity-rng"],
            )
            delta = minimap_run["codelength"] - infomap_run["codelength"]
            deltas.append(delta)
            per_seed.append(
                {
                    "seed": seed,
                    "infomap_codelength": infomap_run["codelength"],
                    "minimap_codelength": minimap_run["codelength"],
                    "delta": delta,
                    "infomap_shape": infomap_run["shape"],
                    "minimap_shape": minimap_run["shape"],
                    "infomap_depth": infomap_run["depth"],
                    "minimap_depth": minimap_run["depth"],
                }
            )

        max_abs_delta = max(abs(d) for d in deltas)
        mean_abs_delta = statistics.mean(abs(d) for d in deltas)
        passed = max_abs_delta <= threshold
        overall_pass = overall_pass and passed
        groups.append(
            {
                "mode": mode_name,
                "directed": directed,
                "network": str(network),
                "threshold": threshold,
                "max_abs_delta": max_abs_delta,
                "mean_abs_delta": mean_abs_delta,
                "passed": passed,
                "per_seed": per_seed,
            }
        )

    return {
        "seeds": seeds,
        "threshold_two_level": threshold_two_level,
        "threshold_multilevel": threshold_multilevel,
        "passed": overall_pass,
        "groups": groups,
    }


def main() -> None:
    args = parse_args()

    minimap = Path(args.minimap)
    infomap = Path(args.infomap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not minimap.exists():
        raise FileNotFoundError(f"minimap binary not found: {minimap}")
    if not infomap.exists():
        raise FileNotFoundError(f"infomap binary not found: {infomap}")

    perf_net = out_dir / "perf_100k.net"
    hier_u = Path("/tmp/minimap_hier_u.net")
    hier_d = Path("/tmp/minimap_hier_d.net")
    generate_perf_100k(perf_net, args.seed)
    if not hier_u.exists():
        hier_u = out_dir / "hier_u_2048.net"
        generate_hierarchical(hier_u, args.seed, directed=False)
    if not hier_d.exists():
        hier_d = out_dir / "hier_d_2048.net"
        generate_hierarchical(hier_d, args.seed, directed=True)

    parity_two_level_u = Path(args.parity_two_level_u_net)
    parity_two_level_d = Path(args.parity_two_level_d_net)
    parity_multilevel_u = Path(args.parity_multilevel_u_net)
    parity_multilevel_d = Path(args.parity_multilevel_d_net)

    if not parity_two_level_u.exists():
        parity_two_level_u = hier_u
    if not parity_two_level_d.exists():
        parity_two_level_d = hier_d
    if not parity_multilevel_u.exists():
        parity_multilevel_u = hier_u
    if not parity_multilevel_d.exists():
        parity_multilevel_d = hier_d

    parity_seeds = parse_seed_list(args.parity_seeds)
    parity_matrix = run_parity_matrix(
        infomap=infomap,
        minimap=minimap,
        two_level_u_net=parity_two_level_u,
        two_level_d_net=parity_two_level_d,
        multilevel_u_net=parity_multilevel_u,
        multilevel_d_net=parity_multilevel_d,
        seeds=parity_seeds,
        threshold_two_level=args.parity_threshold_two_level,
        threshold_multilevel=args.parity_threshold_multilevel,
    )
    parity_path = out_dir / "parity_matrix.json"
    parity_path.write_text(json.dumps(parity_matrix, indent=2))

    cases = [
        ("perf100k", perf_net, False, True, args.perf_reps),
        ("perf100k", perf_net, False, False, args.perf_reps),
        ("perf100k", perf_net, True, True, args.perf_reps),
        ("perf100k", perf_net, True, False, args.perf_reps),
        ("hier_diag", hier_u, False, True, 1),
        ("hier_diag", hier_u, False, False, 1),
        ("hier_diag", hier_d, True, True, 1),
        ("hier_diag", hier_d, True, False, 1),
    ]

    results = []
    for case_name, network, directed, two_level, reps in cases:
        result = run_case(
            case_name=case_name,
            network=network,
            directed=directed,
            two_level=two_level,
            reps=reps,
            infomap=infomap,
            minimap=minimap,
            seed=args.seed,
        )
        results.append(result)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print(f"Wrote benchmark summary: {summary_path}")
    print(f"Wrote parity matrix: {parity_path}")
    print()
    print("parity_mode, directed, max_abs_delta, threshold, passed")
    for group in parity_matrix["groups"]:
        print(
            f"{group['mode']}, {group['directed']}, "
            f"{group['max_abs_delta']:.12f}, {group['threshold']:.12f}, {group['passed']}"
        )
    print(f"parity_overall_passed={parity_matrix['passed']}")
    print()
    print(
        "case, mode, directed, tool, codelength, shape, depth, median_wall_s, median_peak_rss_kib"
    )
    for r in results:
        for tool in ("infomap", "minimap"):
            t = r[tool]
            print(
                f"{r['case']}, {r['mode']}, {r['directed']}, {tool}, "
                f"{t['codelength']:.12f}, {t['shape']}, {t['depth']}, "
                f"{t['wall_time_s_median']:.4f}, {t['peak_rss_kib_median']:.1f}"
            )
        d = r["delta"]
        print(
            f"  delta: codelength={d['codelength_minimap_minus_infomap']:+.12f}, "
            f"time_ratio={d['wall_time_ratio_minimap_over_infomap']:.3f}, "
            f"rss_ratio={d['rss_ratio_minimap_over_infomap']:.3f}"
        )

    if args.strict_parity and not parity_matrix["passed"]:
        print("STRICT PARITY CHECK FAILED")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
