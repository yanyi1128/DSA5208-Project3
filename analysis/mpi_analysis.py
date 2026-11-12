"""Audit Vanda benchmark artifacts and plot MPI scaling (Python + matplotlib).

Run from any directory: python analysis/mpi_analysis.py
Raw inputs are never modified. Warm-ups are excluded; no outlier is removed.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BINDING = re.compile(r'\[([^:\]\s]+):\d+\] MCW rank (\d+) bound to socket (\d+)\[core (\d+)\[hwt (\d+)\]\]')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def host_name(value):
    return value.lower().split('.')[0]


def parse_bindings(content, p):
    matches = BINDING.findall(content)
    require(len(matches) == p, f'Expected {p} binding records, found {len(matches)}')
    mapping = {int(rank): (host_name(host), int(socket), int(core), int(thread))
               for host, rank, socket, core, thread in matches}
    require(set(mapping) == set(range(p)), 'Missing or duplicate MPI ranks')
    require(len({v[:3] for v in mapping.values()}) == p, 'Ranks share physical cores')
    return mapping


def metrics(times, baseline, p, mode):
    median = statistics.median(times)
    ratio = baseline / median
    return dict(median_s=median, min_s=min(times), max_s=max(times),
                mean_s=statistics.mean(times), stdev_s=statistics.stdev(times),
                time_ratio=ratio,
                efficiency_pct=100 * ratio / (p if mode == 'strong' else 1))


def audit(directory):
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    args = meta['arguments']
    mode, repeats = args['mode'], args['repeats']
    raw, old = records(directory / 'raw.csv'), records(directory / 'summary.csv')
    ranks = sorted(args['ranks'])
    require(1 in ranks and repeats >= 2, 'Need P=1 baseline and >=2 measurements')
    require(len(raw) == len(ranks) * (repeats + 1), 'Unexpected raw record count')
    require(len(old) == len(ranks) and {int(r['ranks']) for r in old} == set(ranks),
            'Summary rank counts mismatch')
    allocated = Counter(host_name(h) for h in meta['nodefile'].splitlines() if h.strip())
    results, details, hashes = [], {}, {}
    baseline = statistics.median(float(r['elapsed_s']) for r in raw
                                 if r['phase'] == 'measured' and int(r['ranks']) == 1)
    for p in ranks:
        group = [r for r in raw if int(r['ranks']) == p]
        require(sorted(int(r['repeat']) for r in group) == list(range(repeats + 1)),
                f'P={p}: missing or duplicate repeat')
        n = args['n'] if mode == 'strong' else int(math.floor(args['n'] * math.sqrt(p) + .5))
        timings, maps, extra_lines = [], [], []
        for row in group:
            repeat = int(row['repeat'])
            require(row['mode'] == mode and row['phase'] == ('measured' if repeat else 'warmup'),
                    f'P={p}: inconsistent mode/phase')
            require(int(row['n']) == n and int(row['steps']) == args['steps']
                    and row['implementation'] == 'mpi' and int(row['gpus']) == 0
                    and float(row['r']) == .1, f'P={p}: inconsistent solver configuration')
            elapsed = float(row['elapsed_s'])
            require(math.isfinite(elapsed) and elapsed > 0, 'Invalid elapsed time')
            require(row['validation'] in ('pass', 'not_run'), 'Failed or unknown validation')
            prefix = directory / f'p{p}_{repeat}'
            stdout = records(prefix.with_suffix('.stdout'))
            require(len(stdout) == 1, f'{prefix}: expected one solver record')
            for key, value in stdout[0].items():
                equal = (math.isclose(float(value), float(row[key]), rel_tol=1e-12)
                         if key == 'elapsed_s' else value == row[key])
                require(equal, f'{prefix}: stdout differs from raw.csv ({key})')
            stderr = prefix.with_suffix('.stderr').read_text()
            mapping = parse_bindings(stderr, p)
            counts = Counter(v[0] for v in mapping.values())
            require(all(count <= allocated.get(host, 0) for host, count in counts.items()),
                    f'{prefix}: binding exceeds node allocation')
            maps.append(mapping)
            extra_lines.extend(line for line in stderr.splitlines() if line.strip() and not BINDING.search(line))
            if repeat:
                timings.append(elapsed)
        base, extra = divmod(n, p)
        work = []
        for rank in range(p):
            start = rank * base + min(rank, extra)
            end = start + base + (rank < extra)
            work.append(max(0, min(end, n - 1) - max(start, 1)) * (n - 2))
        computed = metrics(timings, baseline, p, mode)
        prior = next(r for r in old if int(r['ranks']) == p)
        for key in ('median_s', 'min_s', 'max_s'):
            require(math.isclose(computed[key], float(prior[key]), rel_tol=1e-12), f'Summary mismatch: P={p} {key}')
        require(int(prior['n']) == n and int(prior['steps']) == args['steps']
                and prior['mode'] == mode and int(prior['min_interior_cells']) == min(work)
                and int(prior['max_interior_cells']) == max(work), 'Summary workload mismatch')
        counts = Counter(v[0] for v in maps[0].values())
        results.append(dict(mode=mode, n=n, steps=args['steps'], ranks=p,
                            nodes_used=len(counts), measured_repeats=repeats, **computed,
                            avg_interior_cells=(n-2)**2/p, min_interior_cells=min(work),
                            max_interior_cells=max(work),
                            workload_ratio_to_p1=((n-2)**2/p)/((args['n']-2)**2),
                            max_min_time_ratio=max(timings)/min(timings)))
        details[str(p)] = dict(measured_seconds=timings, hosts=dict(counts),
                               bindings_identical_across_repeats=all(m == maps[0] for m in maps),
                               reference_binding=maps[0], other_stderr_lines=extra_lines)
    for path in sorted(directory.iterdir()):
        if path.suffix in ('.csv', '.json', '.stdout', '.stderr'):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return results, dict(job=meta['pbs'].get('PBS_JOBID'), input_directory=str(directory.resolve()),
                         binary_sha256=meta['binary_sha256'], allocated_slots=dict(allocated),
                         modules=meta['modules'], validation_states=sorted({r['validation'] for r in raw}),
                         source_revision=meta['revision'], cpu=meta['cpu'], configurations=details,
                         input_sha256=hashes)


def plots(tables, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    def save(fig, name):
        for extension in ('png', 'pdf'):
            fig.savefig(folder / f'{name}.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    for mode, rows in tables.items():
        p = [r['ranks'] for r in rows]
        t = [r['median_s'] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
        ax.errorbar(p, t, yerr=[[r['median_s']-r['min_s'] for r in rows],
                               [r['max_s']-r['median_s'] for r in rows]],
                    fmt='o-', capsize=4, label='Median; bars = measured min/max')
        ax.plot(p, [t[0]/v if mode == 'strong' else t[0] for v in p], '--', color='gray', label='Ideal scaling')
        ax.set(xscale='log', yscale='log', xlabel='MPI ranks (one rank per CPU core)', ylabel='Iteration time (s)',
               title=f"{mode.capitalize()} scaling | {rows[0]['steps']} steps\n{rows[0]['measured_repeats']} measurements; warm-up excluded")
        ax.set_xticks(p, [str(v) for v in p])
        ax.grid(alpha=.2)
        ax.legend()
        if mode == 'strong':
            ax.annotate('P=4: all measurements retained', (4, t[2]), xytext=(12, t[2]*.65), fontsize=9,
                        arrowprops={'arrowstyle': '->'})
        save(fig, f'mpi_{mode}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for mode, rows in tables.items():
        p = [r['ranks'] for r in rows]
        axes[1].plot(p, [r['efficiency_pct'] for r in rows], 'o-', label=mode.capitalize())
        if mode == 'strong':
            axes[0].plot(p, [r['time_ratio'] for r in rows], 'o-', label='Strong: T(1)/T(P)')
            axes[0].plot(p, p, '--', color='gray', label='Ideal speedup')
    axes[0].set(ylabel='Speedup', yscale='log', title='Strong scaling speedup')
    axes[1].set(ylabel='Efficiency (%)', title='Strong: T1/(P TP); weak: T1/TP', ylim=(0, 105))
    for ax in axes:
        ax.set(xscale='log', xlabel='MPI ranks')
        ax.set_xticks(p, [str(v) for v in p])
        ax.grid(alpha=.2)
        ax.legend()
    save(fig, 'mpi_efficiency')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'results/runs')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    tables, audits = {}, {}
    for path in sorted(args.input_root.rglob('metadata.json')):
        mode = json.loads(path.read_text(encoding='utf-8'))['arguments']['mode']
        require(mode in ('strong', 'weak') and mode not in tables,
                f'Multiple/unknown {mode} datasets: specify a root containing exactly one run per mode')
        tables[mode], audits[mode] = audit(path.parent)
    require(set(tables) == {'strong', 'weak'}, 'Need both strong and weak datasets')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, rows in tables.items():
        with (args.output_dir / f'mpi_{mode}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / 'mpi_audit.json').write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding='utf-8')
    plots(tables, args.output_dir)
    for mode, rows in tables.items():
        last = rows[-1]
        print(f"{mode}: audit PASS; P={last['ranks']}, T={last['median_s']:.6f}s, efficiency={last['efficiency_pct']:.3f}%")
    print(f'Output: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
