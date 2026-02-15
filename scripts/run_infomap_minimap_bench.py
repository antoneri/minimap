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
    return parser.parse_args()


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
            env["OMP_NUM_THREADS"] = "1"
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


if __name__ == "__main__":
    main()
