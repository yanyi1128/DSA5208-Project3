#!/usr/bin/env python3
"""Summarize actual benchmark_cuda.py records without mixing profiling and timing.

Matplotlib is needed only for plotting; --summary-only uses Python's standard library.
"""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_cuda import EXTRA_FIELDS, FIELDS, parse_record, parse_profile


def read_measurements(paths):
    groups = defaultdict(list)
    for supplied in paths:
        path = Path(supplied).resolve()
        if path.is_dir():
            path /= 'raw.csv'
        metadata_path = path.parent / 'metadata.json'
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata.get('status') != 'complete':
                raise ValueError('Experiment is not marked complete: {}'.format(path.parent))
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXTRA_FIELDS + FIELDS:
                raise ValueError('Not a benchmark_cuda.py raw CSV: {}'.format(path))
            count = 0
            for row in reader:
                count += 1
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('Incomplete raw CSV row')
                if row['mode'] not in ['strong', 'weak', 'size'] or row['timing_mode'] not in ['ordinary', 'profile']:
                    raise ValueError('Unknown experiment or timing mode')
                if row['phase'] not in ['warmup', 'measured']:
                    raise ValueError('Unknown measurement phase')
                n, steps, ranks = [int(row[key]) for key in ['n', 'steps', 'ranks']]
                block_x, block_y = int(row['block_x']), int(row['block_y'])
                repeat = int(row['repeat'])
                if n < 4 or steps <= 0 or not 1 <= ranks <= n or block_x < 1 or block_y < 1 or block_x * block_y > 1024:
                    raise ValueError('Invalid recorded configuration')
                if (row['phase'] == 'warmup' and repeat != 0) or (row['phase'] == 'measured' and repeat < 1):
                    raise ValueError('Inconsistent repetition/phase')
                if row['validation'] not in ['pass', 'not_run']:
                    raise ValueError('Failed or unknown validation state')
                from io import StringIO
                text = StringIO()
                writer = csv.DictWriter(text, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({key: row[key] for key in FIELDS})
                parse_record(text.getvalue(), n, steps, ranks, row['validation'] == 'pass')
                if (row['timing_mode'] == 'profile') != bool(row['profile_file']):
                    raise ValueError('Profile flag and profile file disagree')
                key = (str(path), row['mode'], row['timing_mode'], steps, block_x, block_y, n, ranks)
                groups[key].append(row)
            if not count:
                raise ValueError('Empty measurement file')
    return groups


def summarize(groups):
    summaries = []
    for key, records in groups.items():
        source, mode, timing_mode, steps, block_x, block_y, n, ranks = key
        measured = [row for row in records if row['phase'] == 'measured']
        warmups = [row for row in records if row['phase'] == 'warmup']
        repetitions = [int(row['repeat']) for row in measured]
        if len(warmups) != 1 or len(measured) < 5 or sorted(repetitions) != list(range(1, len(measured) + 1)):
            raise ValueError('Require one warm-up and at least five distinct, consecutive measurements per configuration')
        states = {row['validation'] for row in measured}
        if len(states) != 1:
            raise ValueError('Do not mix validated and unvalidated measurement protocols')
        times = [float(row['elapsed_s']) for row in measured]
        summaries.append(dict(source=source, mode=mode, timing_mode=timing_mode, n=n, steps=steps,
                              ranks=ranks, gpus=ranks, block_x=block_x, block_y=block_y,
                              measured_repeats=len(times), validation=next(iter(states)),
                              median_s=statistics.median(times), min_s=min(times), max_s=max(times),
                              speedup_vs_one_gpu='NA', efficiency_pct='NA',
                              serial_median_s='NA', serial_relative_ratio='NA',
                              mpi_cpu_ranks='NA', mpi_median_s='NA', mpi_relative_ratio='NA'))
    curves = defaultdict(list)
    for row in summaries:
        curves[curve_key(row)].append(row)
    for key, rows in curves.items():
        mode = key[1]
        if mode == 'size':
            if len({row['ranks'] for row in rows}) != 1:
                raise ValueError('A size sweep must hold rank/GPU count fixed')
            continue
        baseline = [row for row in rows if row['ranks'] == 1]
        if len(baseline) != 1:
            raise ValueError('Each scaling curve needs exactly one one-GPU baseline')
        baseline = baseline[0]
        for row in rows:
            expected_n = baseline['n'] if mode == 'strong' else int(math.floor(baseline['n'] * math.sqrt(row['ranks']) + 0.5))
            if row['n'] != expected_n:
                raise ValueError('Problem size does not follow the declared scaling protocol')
            ratio = baseline['median_s'] / row['median_s']
            row['efficiency_pct'] = 100 * ratio / row['ranks'] if mode == 'strong' else 100 * ratio
            if mode == 'strong':
                row['speedup_vs_one_gpu'] = ratio
    return sorted(summaries, key=lambda row: (curve_key(row), row['n'], row['ranks']))


def curve_key(row):
    return (row['source'], row['mode'], row['timing_mode'], row['steps'], row['block_x'], row['block_y'])


def serial_baselines(path):
    result = defaultdict(list)
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError('Serial CSV lacks solver fields')
        for row in reader:
            if row.get('phase', 'measured') == 'warmup':
                continue
            if row['implementation'] != 'serial' or row['ranks'] != '1' or row['gpus'] != '0':
                raise ValueError('Expected serial, one CPU thread, zero GPUs')
            if row['validation'] not in ['pass', 'not_run']:
                raise ValueError('Serial baseline contains failed validation')
            if not math.isclose(float(row['r']), 0.1, rel_tol=0, abs_tol=1e-15):
                raise ValueError('Serial baseline uses a different diffusion coefficient')
            n, steps, elapsed = int(row['n']), int(row['steps']), float(row['elapsed_s'])
            if n < 4 or steps <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Invalid serial baseline')
            result[(n, steps)].append(elapsed)
    if any(len(times) < 5 for times in result.values()):
        raise ValueError('Serial comparison requires at least five measured samples per configuration')
    return {key: statistics.median(times) for key, times in result.items()}


def mpi_baselines(path, ranks):
    """Read B's measured strong-scaling summary, selecting an explicit CPU budget."""
    result = {}
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'mode', 'n', 'steps', 'ranks', 'median_s', 'measured_repeats'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('MPI CSV lacks B\'s strong-scaling summary fields')
        for row in reader:
            if row['mode'] != 'strong' or int(row['ranks']) != ranks:
                continue
            key = int(row['n']), int(row['steps'])
            elapsed = float(row['median_s'])
            if min(key) <= 0 or not math.isfinite(elapsed) or elapsed <= 0 or int(row['measured_repeats']) < 5:
                raise ValueError('Invalid MPI baseline configuration or repetitions')
            if key in result:
                raise ValueError('Ambiguous duplicate MPI baseline for a given N and K')
            result[key] = elapsed
    if not result:
        raise ValueError('No MPI strong-scaling rows for the selected CPU rank count')
    return result


def profile_critical_rank_rows(groups):
    result = []
    for key, records in groups.items():
        source, mode, timing_mode, steps, block_x, block_y, n, ranks = key
        if timing_mode != 'profile':
            continue
        for record in records:
            if record['phase'] != 'measured':
                continue
            directory = Path(source).parent
            path = (directory / record['profile_file']).resolve()
            try:
                path.relative_to(directory)
            except ValueError:
                raise ValueError('Profile file must belong to its experiment directory')
            rows = parse_profile(path, n, steps, ranks, block_x, block_y)
            critical = max(rows, key=lambda row: float(row['total_s']))
            if not math.isclose(float(critical['total_s']), float(record['elapsed_s']), rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError('Profile max rank time disagrees with solver elapsed time')
            result.append(dict(source=source, mode=mode, timing_mode=timing_mode,
                               repeat=record['repeat'], ranks=ranks, **critical))
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output, summaries, profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves = defaultdict(list)
    for row in summaries:
        curves[curve_key(row)].append(row)
    for index, (key, rows) in enumerate(sorted(curves.items()), 1):
        _, mode, timing_mode, steps, block_x, block_y = key
        rows.sort(key=lambda row: row['n'] if mode == 'size' else row['ranks'])
        x = [row['n'] if mode == 'size' else row['ranks'] for row in rows]
        y = [row['median_s'] for row in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax.errorbar(x, y, yerr=[[row['median_s'] - row['min_s'] for row in rows],
                              [row['max_s'] - row['median_s'] for row in rows]], marker='o', capsize=4, label='Median and measured min/max')
        if mode in ['strong', 'weak']:
            baseline = next(row for row in rows if row['ranks'] == 1)
            ideal = [baseline['median_s'] / value if mode == 'strong' else baseline['median_s'] for value in x]
            ax.plot(x, ideal, '--', label='Ideal from one GPU')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Grid side N' if mode == 'size' else 'MPI ranks = exclusive GPUs')
        ax.set_ylabel('Complete-iteration time (s)')
        ax.set_title('{}; {}; K={}; block={}x{}'.format(mode, timing_mode, steps, block_x, block_y))
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8)
        for extension in ['png', 'pdf']:
            fig.savefig(output / 'curve_{}_runtime.{}'.format(index, extension), dpi=180)
        plt.close(fig)
        if mode in ['strong', 'weak']:
            fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
            ax.plot(x, [row['efficiency_pct'] for row in rows], 'o-', label=mode + ' efficiency')
            ax.axhline(100, linestyle='--', color='gray')
            ax.set_xscale('log', base=2)
            ax.set_xlabel('MPI ranks = exclusive GPUs')
            ax.set_ylabel('Efficiency (%)')
            ax.set_title('One-GPU baseline; ' + timing_mode)
            ax.grid(True, alpha=0.25)
            ax.legend()
            for extension in ['png', 'pdf']:
                fig.savefig(output / 'curve_{}_efficiency.{}'.format(index, extension), dpi=180)
            plt.close(fig)
            if mode == 'strong':
                fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
                ax.plot(x, [row['speedup_vs_one_gpu'] for row in rows], 'o-', label='Speedup over one GPU')
                ax.plot(x, x, '--', label='Ideal')
                ax.set_xscale('log', base=2)
                ax.set_xlabel('MPI ranks = exclusive GPUs')
                ax.set_ylabel('Speedup')
                ax.grid(True, alpha=0.25)
                ax.legend()
                for extension in ['png', 'pdf']:
                    fig.savefig(output / 'curve_{}_speedup.{}'.format(index, extension), dpi=180)
                plt.close(fig)
    profile_groups = defaultdict(list)
    for row in profiles:
        profile_groups[(row['source'], int(row['n']), int(row['steps']), row['ranks'], int(row['block_x']), int(row['block_y']))].append(row)
    for index, (key, rows) in enumerate(sorted(profile_groups.items()), 1):
        columns = ['kernel_s', 'd2h_s', 'h2d_s', 'mpi_s', 'total_s']
        values = [statistics.median(float(row[column]) for row in rows) for column in columns]
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        labels = ['Kernel', 'D2H', 'H2D', 'MPI', 'Total']
        floor = min(value for value in values if value > 0) / 10
        ax.bar(labels, [value if value > 0 else math.nan for value in values])
        ax.set_yscale('log')
        ax.set_ylim(bottom=floor)
        for position, value in enumerate(values):
            if value == 0:
                ax.text(position, floor * 1.2, '0', ha='center', va='bottom')
        ax.set_ylabel('Time on each run\'s slowest rank (s)')
        ax.set_title('Instrumented medians; N={}, K={}, GPUs={}'.format(key[1], key[2], key[3]))
        for extension in ['png', 'pdf']:
            fig.savefig(output / 'profile_{}.{}'.format(index, extension), dpi=180)
        plt.close(fig)
    for index, row in enumerate(summaries, 1):
        if row['timing_mode'] != 'ordinary':
            continue
        labels, values = [], []
        if row['serial_median_s'] != 'NA':
            labels.append('Serial: 1 CPU thread')
            values.append(row['serial_median_s'])
        if row['mpi_median_s'] != 'NA':
            labels.append('MPI: {} CPU ranks'.format(row['mpi_cpu_ranks']))
            values.append(row['mpi_median_s'])
        if not values:
            continue
        labels.append('MPI+CUDA: {} GPU(s)'.format(row['gpus']))
        values.append(row['median_s'])
        fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
        bars = ax.bar(labels, values)
        ax.set_yscale('log')
        ax.set_ylabel('Median complete-iteration time (s)')
        ax.set_title('N={}, K={}, CUDA block={}x{}'.format(row['n'], row['steps'], row['block_x'], row['block_y']))
        for bar, value in zip(bars, values):
            ax.annotate('{:.4g}'.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        ax.text(0.5, -0.15, 'Different compute resources; check hardware and compiler metadata.',
                transform=ax.transAxes, ha='center', fontsize=8)
        for extension in ['png', 'pdf']:
            fig.savefig(output / 'comparison_{}.{}'.format(index, extension), dpi=180)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True, help='Completed experiment directories or their raw.csv files')
    parser.add_argument('--output', type=Path, required=True, help='New analysis directory')
    parser.add_argument('--serial-csv', type=Path, help='Optional actual serial measurements with matching N, K and timing protocol')
    parser.add_argument('--mpi-csv', type=Path, help='Optional B MPI strong-scaling summary, with matching N and K')
    parser.add_argument('--mpi-ranks', type=int, help='CPU rank count to select from --mpi-csv')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    if bool(args.mpi_csv) != (args.mpi_ranks is not None) or (args.mpi_ranks is not None and args.mpi_ranks < 1):
        parser.error('Use --mpi-csv together with a positive --mpi-ranks')
    groups = read_measurements(args.input)
    summaries = summarize(groups)
    profiles = profile_critical_rank_rows(groups)
    if args.serial_csv:
        serial = serial_baselines(args.serial_csv)
        for row in summaries:
            if row['timing_mode'] == 'ordinary' and (row['n'], row['steps']) in serial:
                row['serial_median_s'] = serial[(row['n'], row['steps'])]
                row['serial_relative_ratio'] = row['serial_median_s'] / row['median_s']
    if args.mpi_csv:
        mpi = mpi_baselines(args.mpi_csv, args.mpi_ranks)
        for row in summaries:
            if row['timing_mode'] == 'ordinary' and (row['n'], row['steps']) in mpi:
                row['mpi_cpu_ranks'] = args.mpi_ranks
                row['mpi_median_s'] = mpi[(row['n'], row['steps'])]
                row['mpi_relative_ratio'] = row['mpi_median_s'] / row['median_s']
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'profile_critical_ranks.csv', profiles)
    notes = {
        'input': [str(path.resolve()) for path in args.input],
        'ordinary_vs_profile': 'Kept in separate curves. Profiling synchronizes stages and changes runtime.',
        'strong': 'Speedup = one-GPU time / P-GPU time; efficiency = speedup / P.',
        'weak': 'Efficiency = one-GPU baseline time / P-GPU time, without dividing by P again.',
        'serial_comparison': 'Optional serial_relative_ratio mixes CPU/GPU hardware and implementations; it is not GPU scaling efficiency. User must verify hardware, compiler and timing comparability.',
        'mpi_comparison': 'Optional MPI baseline is selected by an explicit CPU rank count from B\'s strong-scaling summary, matched by N and K. It is a cross-resource time ratio, not GPU scaling efficiency.',
        'profiles': 'Use the same slowest-total rank within each measured run; never sum independently maximized stages across ranks. Component medians need not sum to the median total.',
        'validation': 'not_run remains not_run; separate small-grid gates do not validate every measured grid.',
        'error_bars': 'Minimum and maximum observed among measured repetitions, not confidence intervals.',
    }
    (args.output / 'analysis_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    if not args.summary_only:
        make_plots(args.output, summaries, profiles)
    print('Analysis saved to {}'.format(args.output))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, ImportError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        sys.exit(1)
