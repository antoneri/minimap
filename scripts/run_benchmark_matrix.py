#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run minimap vs Infomap benchmark matrix: "
            "4 modes x trials {1,8} x datasets {100k,1M}."
        )
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
        default=f"/tmp/minimap_bench_matrix_{int(time.time())}",
        help="Output directory for artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for network generation and tool runs.",
    )
    parser.add_argument(
        "--infomap-omp-threads",
        type=int,
        default=12,
        help="OMP_NUM_THREADS value for Infomap runs.",
    )
    parser.add_argument(
        "--infomap-inner-parallelization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --inner-parallelization to Infomap.",
    )
    parser.add_argument(
        "--minimap-max-threads",
        type=int,
        default=12,
        help="Max minimap --threads. Actual per run is min(num-trials, this value).",
    )
    parser.add_argument(
        "--regen-networks",
        action="store_true",
        help="Regenerate dataset networks even if files already exist.",
    )
    parser.add_argument(
        "--skip-infomap-1m-trials8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip Infomap runs for dataset=1m and num-trials=8 "
            "(very expensive, usually low additional signal)."
        ),
    )
    parser.add_argument(
        "--infomap-fallback-no-inner-parallelization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If an Infomap run fails with --inner-parallelization, retry once "
            "without that flag."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing benchmark_matrix.json in --out-dir.",
    )
    return parser.parse_args()


def generate_network(path: Path, n_nodes: int, n_edges: int, seed: int) -> None:
    random.seed(seed)
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


def parse_time_output(stderr: str) -> tuple[float, float]:
    real_match = re.search(r"(?m)^\s*real\s+([0-9.]+)$", stderr)
    rss_match = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", stderr)
    if not rss_match:
        rss_match = re.search(r"(?m)^\s*maximum resident set size\s+(\d+)\s*$", stderr)
    if not real_match or not rss_match:
        raise RuntimeError(f"Could not parse /usr/bin/time output:\n{stderr}")
    wall = float(real_match.group(1))
    rss_kib = int(rss_match.group(1)) / 1024.0
    return wall, rss_kib


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
    return {
        "codelength": codelength,
        "depth": max_depth,
        "shape": shape,
        "top_modules": shape[0] if shape else 0,
    }


def run_tool(
    tool: str,
    minimap: Path,
    infomap: Path,
    network: Path,
    directed: bool,
    two_level: bool,
    seed: int,
    trials: int,
    infomap_omp_threads: int,
    infomap_inner_parallelization: bool,
    minimap_threads: int,
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
                str(trials),
            ]
            if infomap_inner_parallelization:
                cmd.append("--inner-parallelization")
            if directed:
                cmd.append("--directed")
            if two_level:
                cmd.append("--two-level")
            cmd.extend(["--tree", str(network), str(out_dir)])
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = str(infomap_omp_threads)
            workers = infomap_omp_threads
        else:
            cmd = [
                str(minimap),
                str(network),
                str(out_dir),
                "--seed",
                str(seed),
                "--num-trials",
                str(trials),
                "--threads",
                str(minimap_threads),
                "--tree",
                "--silent",
            ]
            if directed:
                cmd.append("--directed")
            cmd.append("--two-level" if two_level else "--multilevel")
            env = None
            workers = minimap_threads

        timed_cmd = ["/usr/bin/time", "-lp"] + cmd
        proc = subprocess.run(timed_cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        wall_s, peak_rss_kib = parse_time_output(proc.stderr)
        tree_files = sorted(out_dir.glob("*.tree"))
        if len(tree_files) != 1:
            raise RuntimeError(f"Expected one .tree in {out_dir}, found {tree_files}")
        tree = parse_tree(tree_files[0])
        return {
            "tool": tool,
            "workers": workers,
            "wall_s": wall_s,
            "peak_rss_kib": peak_rss_kib,
            **tree,
        }


def case_name(directed: bool, two_level: bool) -> str:
    direction = "directed" if directed else "undirected"
    level = "two-level" if two_level else "multilevel"
    return f"{direction}/{level}"


def to_markdown(results: list[dict], dataset_key: str, trials: int) -> str:
    rows = [r for r in results if r["dataset"] == dataset_key and r["trials"] == trials]
    rows.sort(key=lambda r: (r["directed"], r["two_level"], r["tool"]))
    lines = [
        "| Mode | Tool/config | Effective workers | Wall time | Peak RSS (KiB) | Top modules | Depth | Codelength |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        mode = case_name(r["directed"], r["two_level"])
        tool_cfg = (
            f"`infomap` (`OMP_NUM_THREADS={r['workers']}`)"
            if r["tool"] == "infomap"
            else f"`minimap --threads {r['workers']}`"
        )
        lines.append(
            f"| {mode} | {tool_cfg} | {r['workers']} | "
            f"{r['wall_s']:.2f} s | {r['peak_rss_kib']:.0f} | {r['top_modules']} | "
            f"{r['depth']} | {r['codelength']:.4f} |"
        )
    return "\n".join(lines)


def run_key(record: dict) -> tuple:
    return (
        record["dataset"],
        record["trials"],
        record["directed"],
        record["two_level"],
        record["tool"],
    )


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

    datasets = {
        "100k": {"file": out_dir / "perf_100k.net", "n_nodes": 20_000, "n_edges": 100_000},
        "1m": {"file": out_dir / "perf_1m.net", "n_nodes": 200_000, "n_edges": 1_000_000},
    }

    for spec in datasets.values():
        net = spec["file"]
        if args.regen_networks or not net.exists():
            print(f"Generating {net.name} ...", flush=True)
            generate_network(net, spec["n_nodes"], spec["n_edges"], args.seed)

    modes = [
        (False, True),   # undirected two-level
        (False, False),  # undirected multilevel
        (True, True),    # directed two-level
        (True, False),   # directed multilevel
    ]
    trials_list = [1, 8]

    summary_path = out_dir / "benchmark_matrix.json"
    results: list[dict] = []
    if args.resume and summary_path.exists():
        loaded = json.loads(summary_path.read_text())
        if isinstance(loaded, list):
            results = loaded
            print(f"Resuming from {summary_path} with {len(results)} existing rows.", flush=True)
    completed = {run_key(r) for r in results}
    for dataset_key, spec in datasets.items():
        net = spec["file"]
        for trials in trials_list:
            minimap_threads = max(1, min(trials, args.minimap_max_threads))
            for directed, two_level in modes:
                for tool in ("infomap", "minimap"):
                    if (
                        tool == "infomap"
                        and dataset_key == "1m"
                        and trials == 8
                        and args.skip_infomap_1m_trials8
                    ):
                        print(
                            "Skipping dataset=1m trials=8 tool=infomap "
                            "(--skip-infomap-1m-trials8)",
                            flush=True,
                        )
                        continue

                    key = (dataset_key, trials, directed, two_level, tool)
                    if key in completed:
                        print(
                            "Skipping completed "
                            f"dataset={dataset_key} trials={trials} "
                            f"mode={case_name(directed, two_level)} tool={tool}",
                            flush=True,
                        )
                        continue

                    print(
                        f"Running dataset={dataset_key} trials={trials} "
                        f"mode={case_name(directed, two_level)} tool={tool}",
                        flush=True,
                    )
                    inner_used = args.infomap_inner_parallelization
                    try:
                        run = run_tool(
                            tool=tool,
                            minimap=minimap,
                            infomap=infomap,
                            network=net,
                            directed=directed,
                            two_level=two_level,
                            seed=args.seed,
                            trials=trials,
                            infomap_omp_threads=args.infomap_omp_threads,
                            infomap_inner_parallelization=inner_used,
                            minimap_threads=minimap_threads,
                        )
                    except RuntimeError as err:
                        if (
                            tool == "infomap"
                            and inner_used
                            and args.infomap_fallback_no_inner_parallelization
                        ):
                            print(
                                "Infomap run failed; retrying without "
                                "--inner-parallelization for this case.",
                                flush=True,
                            )
                            inner_used = False
                            run = run_tool(
                                tool=tool,
                                minimap=minimap,
                                infomap=infomap,
                                network=net,
                                directed=directed,
                                two_level=two_level,
                                seed=args.seed,
                                trials=trials,
                                infomap_omp_threads=args.infomap_omp_threads,
                                infomap_inner_parallelization=inner_used,
                                minimap_threads=minimap_threads,
                            )
                        else:
                            raise err
                    run["dataset"] = dataset_key
                    run["trials"] = trials
                    run["directed"] = directed
                    run["two_level"] = two_level
                    if tool == "infomap":
                        run["inner_parallelization"] = inner_used
                    results.append(run)
                    completed.add(key)
                    summary_path.write_text(json.dumps(results, indent=2))

    summary_path.write_text(json.dumps(results, indent=2))

    markdown_path = out_dir / "benchmark_matrix.md"
    md_lines = [
        "# Benchmark Matrix",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- seed: {args.seed}",
        f"- infomap_omp_threads: {args.infomap_omp_threads}",
        f"- infomap_inner_parallelization: {args.infomap_inner_parallelization}",
        f"- skip_infomap_1m_trials8: {args.skip_infomap_1m_trials8}",
        "",
        "## 100k (`--num-trials 1`)",
        "",
        to_markdown(results, "100k", 1),
        "",
        "## 100k (`--num-trials 8`)",
        "",
        to_markdown(results, "100k", 8),
        "",
        "## 1M (`--num-trials 1`)",
        "",
        to_markdown(results, "1m", 1),
        "",
        "## 1M (`--num-trials 8`)",
        "",
        to_markdown(results, "1m", 8),
        "",
    ]
    markdown_path.write_text("\n".join(md_lines))

    print(f"Wrote JSON: {summary_path}")
    print(f"Wrote Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
