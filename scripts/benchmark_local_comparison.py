#!/usr/bin/env python3
"""Paired local serial/one-GPU measurements; never Atlas/Vanda evidence."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys

from benchmark_cuda import FIELDS, command_info, parse_record, run_logged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validated-run', type=Path, required=True,
                   help='Completed local cuda_job.py run, with source snapshot and both binaries')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024, 2048, 4096])
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    if os.environ.get('PBS_JOBID'):
        p.error('Local comparison only; do not run this wrapper inside PBS')
    if a.steps < 1 or a.repeats < 5 or min(a.sizes) < 4 or len(set(a.sizes)) != len(a.sizes):
        p.error('Positive steps, at least five repeats and unique sizes >=4 are required')
    run = a.validated_run.resolve()
    previous = json.loads((run / 'job_status.json').read_text())
    if previous['status'] != 'complete' or previous['execution_environment'] != 'local':
        p.error('Input must be a completed local run')
    a.output.mkdir(parents=True, exist_ok=False)
    meta = {'status': 'running', 'execution_environment': 'local',
            'is_atlas_or_vanda': False, 'validated_run': str(run),
            'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'platform': platform.platform(), 'sizes': a.sizes, 'steps': a.steps,
            'repeats': a.repeats, 'gpu_block': [32, 8],
            'protocol': 'Full validation for both implementations at every N, then one ordinary warmup each. Five measured fresh processes per implementation. Order alternates serial/GPU, then GPU/serial between repetitions. Timer excludes allocation, validation and output.',
            'limitation': 'Uncontrolled laptop thermal/power state; CPU/GPU ratio is not MPI strong-scaling efficiency.',
            'hardware': {'cpu': command_info(['lscpu']), 'gpu': command_info(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'])},
            'toolchain': {'cuda': command_info(['nvcc', '--version']), 'compiler': command_info(['g++', '--version']), 'mpi': command_info(['mpirun', '--version'])},
            'binary_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in ['heat_serial', 'heat_mpi_cuda']}}
    shutil.copy2(__file__, a.output / 'benchmark_local_comparison.py')
    status = a.output / 'metadata.json'
    def save():
        status.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    def invoke(kind, n, label, validate=False):
        command = ([str(run / 'heat_serial')] if kind == 'serial' else
                   ['mpirun', '-np', '1', str(run / 'heat_mpi_cuda')])
        command += ['--n', str(n), '--steps', str(a.steps)]
        if kind != 'serial':
            command += ['--block-x', '32', '--block-y', '8']
        if validate:
            command.append('--validate')
        output = run_logged(command, a.output / label, 600)
        return parse_record(output, n, a.steps, 1, validated=validate, implementation=kind)
    save()
    try:
        with (a.output / 'raw.csv').open('w', newline='') as f, (a.output / 'validation.csv').open('w', newline='') as vf:
            writer = csv.DictWriter(f, fieldnames=['phase', 'repeat'] + FIELDS)
            validation = csv.DictWriter(vf, fieldnames=FIELDS)
            writer.writeheader(); validation.writeheader()
            summary = []
            for n in a.sizes:
                meta['current_n'] = n; save()
                samples = {'serial': [], 'mpi_cuda': []}
                for kind in samples:
                    validation.writerow(invoke(kind, n, f'validate_{kind}_{n}', True)); vf.flush()
                    writer.writerow(dict(phase='warmup', repeat=0, **invoke(kind, n, f'warmup_{kind}_{n}'))); f.flush()
                for repeat in range(1, a.repeats + 1):
                    order = ['serial', 'mpi_cuda'] if repeat % 2 else ['mpi_cuda', 'serial']
                    for kind in order:
                        row = invoke(kind, n, f'measured_{kind}_{n}_{repeat}')
                        writer.writerow(dict(phase='measured', repeat=repeat, **row)); f.flush()
                        samples[kind].append(float(row['elapsed_s']))
                result = {'n': n, 'steps': a.steps, 'measured_repeats': a.repeats}
                for kind, values in samples.items():
                    result.update({kind + '_median_s': statistics.median(values), kind + '_min_s': min(values), kind + '_max_s': max(values)})
                result['serial_over_gpu_ratio'] = result['serial_median_s'] / result['mpi_cuda_median_s']
                summary.append(result)
                print('Complete N={}: {}'.format(n, result), flush=True)
        with (a.output / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        meta['status'] = 'complete'
    except BaseException as e:
        meta['status'] = 'failed'; meta['error'] = repr(e); raise
    finally:
        meta['utc_end'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()

if __name__ == '__main__':
    main()
