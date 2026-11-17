#!/usr/bin/env python3
"""One allocated GPU: snapshot, build, validate, and optionally benchmark.

This does not submit jobs. Load a compatible CUDA/MPI environment first.
The default requires PBS; --local runs on your own GPU with local provenance.
--dry-run prints a plan without creating files or launching any process.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import socket
import sys
import uuid

from benchmark_cuda import execution_environment, origin_note, run_logged

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIRECTORIES = ['src', 'include', 'scripts', 'analysis', 'tests']


def make_plan(run_dir, phase, launcher_args, build_jobs, local=False):
    source = run_dir / 'source'
    gpu = run_dir / 'heat_mpi_cuda'
    serial = run_dir / 'heat_serial'
    common = ['--exe', str(gpu), '--ranks', '1', '--allocated-gpus', '1',
              '--launcher-args=' + launcher_args, '--timeout', '120']
    if local:
        common.append('--local')
    # Outer Python stages have no independent timeout: each MPI/serial command
    # has its own timeout and PBS enforces the overall allocation walltime.
    plan = [
        ('gpu_info', ['nvidia-smi', '-L'], 30),
        ('cuda_version', ['nvcc', '--version'], 30),
        ('mpi_compiler', ['mpicxx', '--showme:command'], 30),
        ('host_compiler', ['g++', '--version'], 30),
        ('mpi_version', ['mpirun', '--version'], 30),
        ('build', ['make', '-f', 'Makefile.cuda', '-j' + str(build_jobs), 'all', 'serial',
                   'BUILD_DIR=' + str(run_dir / 'build'), 'TARGET=' + str(gpu),
                   'SERIAL_TARGET=' + str(serial)], 900),
        ('validate_32x8', [sys.executable, str(source / 'scripts/test_cuda.py'), *common,
                           '--serial-exe', str(serial), '--block-x', '32', '--block-y', '8',
                           '--output', str(run_dir / 'validation_32x8')], None),
    ]
    if phase == 'pilot':
        return plan
    plan.append(('validate_16x16', [sys.executable, str(source / 'scripts/test_cuda.py'), *common,
                                   '--serial-exe', str(serial), '--block-x', '16', '--block-y', '16',
                                   '--output', str(run_dir / 'validation_16x16')], None))
    for block_x, block_y in [(32, 8), (16, 16)]:
        name = 'size_{}x{}'.format(block_x, block_y)
        plan.append((name, [sys.executable, str(source / 'scripts/benchmark_cuda.py'), *common,
                            '--mode', 'size', '--sizes', '256', '512', '1024', '2048', '4096',
                            '--steps', '1000', '--repeats', '5', '--block-x', str(block_x),
                            '--block-y', str(block_y), '--output', str(run_dir / name)], None))
    summary = [sys.executable, str(source / 'analysis/cuda_analysis.py'),
               '--input', str(run_dir / 'size_32x8'), str(run_dir / 'size_16x16'),
               '--output', str(run_dir / 'analysis'), '--summary-only']
    # Summarize only the measurements produced by this job. Historical CPU
    # CSVs may belong to different machines or have incomplete provenance;
    # a CPU/GPU comparison must be requested separately with verified inputs.
    plan.append(('summary', summary, None))
    return plan


def snapshot_source(destination):
    """Freeze the actual sources used by this job, excluding other run directories."""
    destination.mkdir()
    for directory in SNAPSHOT_DIRECTORIES:
        shutil.copytree(ROOT / directory, destination / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ['Makefile.cuda', 'SPEC.md']:
        shutil.copy2(ROOT / filename, destination / filename)
    (destination / 'results').mkdir()
    # Fixed numerical fields are used by host-test. The fresh GPU validation
    # generates its own serial fields; no historical timing CSV is required.
    for name in ['serial_reference_64_100.csv', 'serial_reference_128_200.csv']:
        shutil.copy2(ROOT / 'results' / name, destination / 'results' / name)
    return {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob('*')) if p.is_file()}


def execute_plan(plan, run_dir, metadata, runner=run_logged):
    status_file = run_dir / 'job_status.json'

    def save():
        status_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    save()
    try:
        for name, command, timeout in plan:
            metadata['current_stage'] = name
            save()
            print('START {}: {}'.format(name, shlex.join(command)), flush=True)
            runner(command, run_dir / name, timeout)
            metadata['completed_stages'].append(name)
            save()
            print('PASS ' + name, flush=True)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'failed'
        metadata['error'] = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        metadata['utc_last_update'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['pilot', 'benchmark'], default='pilot')
    parser.add_argument('--output', type=Path, help='New directory; default is a unique results/runs directory')
    parser.add_argument('--launcher-args', default='')
    parser.add_argument('--build-jobs', type=int, default=1, help='Do not exceed allocated CPU cores')
    parser.add_argument('--local', action='store_true', help='Use your own local GPU; record separate local evidence, never cluster results')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if sys.version_info < (3, 8):
        parser.error('Python >= 3.8 is required')
    if args.build_jobs < 1:
        parser.error('--build-jobs must be positive')
    try:
        environment = execution_environment(args.local, require_allocation=False)
    except ValueError as error:
        parser.error(str(error))
    job_id = 'local' if args.local else os.environ.get('PBS_JOBID', 'preview')
    unique = re.sub(r'[^A-Za-z0-9_.-]', '_', job_id) + '_' + uuid.uuid4().hex[:8]
    results_parent = ROOT / 'results' / ('local_runs' if args.local else 'runs')
    run_dir = (args.output or results_parent / ('cuda_' + unique)).resolve()
    if any(c.isspace() for c in str(run_dir)):
        parser.error('The Makefile requires a build/output path without whitespace')
    for directory in SNAPSHOT_DIRECTORIES:
        copied = (ROOT / directory).resolve()
        if run_dir == copied or copied in run_dir.parents:
            parser.error('--output must not be inside a source snapshot directory: ' + directory)
    plan = make_plan(run_dir, args.phase, args.launcher_args, args.build_jobs, local=args.local)
    print(origin_note(environment), flush=True)
    if args.dry_run:
        for name, command, _ in plan:
            print(name + ': ' + shlex.join(command))
        print('Preview only: no files created, no commands executed, no PBS job submitted.')
        return 0
    try:
        execution_environment(args.local)
    except ValueError as error:
        parser.error(str(error))
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'preparing', 'pbs_job_id': None if args.local else job_id, 'phase': args.phase,
                'execution_environment': environment, 'data_origin_note': origin_note(environment),
                'hostname': socket.gethostname(), 'operating_system': platform.platform(),
                'kernel': platform.release(), 'python_version': platform.python_version(),
                'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'completed_stages': [], 'modules': os.environ.get('LOADEDMODULES', ''),
                'note': 'Single GPU only. Ordinary size timings; no multi-GPU or profiling measurements. A running/preparing status after termination is incomplete.'}
    status_file = run_dir / 'job_status.json'
    status_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Results directory: ' + str(run_dir), flush=True)
    original_cwd = Path.cwd()
    try:
        hashes = snapshot_source(run_dir / 'source')
        (run_dir / 'source_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
        os.chdir(run_dir / 'source')
        metadata['status'] = 'running'
        execute_plan(plan, run_dir, metadata)
    except BaseException as error:
        if metadata['status'] != 'failed':
            metadata.update(status='failed', error='{}: {}'.format(type(error).__name__, error))
            status_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        raise
    finally:
        os.chdir(original_cwd)
    print('Completed {}: {}'.format(args.phase, run_dir), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        sys.exit(1)
