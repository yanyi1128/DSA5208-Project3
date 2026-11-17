#!/usr/bin/env python3
"""Collect fresh-process MPI+CUDA measurements on PBS or an explicit local GPU.

The default requires a PBS allocation. --local labels an owned local-machine
experiment, never an Atlas/Vanda result. This script never submits a job or
chooses a queue. --dry-run launches nothing.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['implementation', 'n', 'steps', 'r', 'ranks', 'gpus', 'elapsed_s',
          'validation', 'max_abs_error', 'relative_l2_error']
PROFILE_FIELDS = ['rank', 'device', 'n', 'steps', 'block_x', 'block_y',
                  'kernel_s', 'd2h_s', 'h2d_s', 'mpi_s', 'total_s']
EXTRA_FIELDS = ['mode', 'timing_mode', 'phase', 'repeat', 'block_x', 'block_y',
                'profile_file']


def execution_environment(local=False, require_allocation=True):
    """Keep explicit local evidence separate from scheduler-authorized runs."""
    if local and os.environ.get('PBS_JOBID'):
        raise ValueError('--local cannot be combined with PBS_JOBID; use the normal PBS mode')
    if not local and require_allocation and not os.environ.get('PBS_JOBID'):
        raise ValueError('Run inside an allocated PBS GPU job; use --local on your own GPU or --dry-run elsewhere')
    return 'local' if local else 'pbs'


def origin_note(environment):
    if environment == 'local':
        return 'LOCAL GPU experiment: not an Atlas/Vanda measurement; do not relabel as cluster data.'
    return 'PBS allocation experiment; identify the actual cluster and hardware from recorded evidence.'


def parse_record(text, n, steps, ranks, validated=False, implementation='mpi_cuda'):
    lines = [line for line in text.splitlines() if line.strip()]
    reader = csv.DictReader(io.StringIO('\n'.join(lines)))
    rows = list(reader)
    if (reader.fieldnames != FIELDS or len(rows) != 1 or None in rows[0]
            or any(value is None for value in rows[0].values())):
        raise ValueError('Expected exactly one solver CSV header and one complete record')
    row = rows[0]
    expected = {'implementation': implementation, 'n': str(n), 'steps': str(steps),
                'ranks': str(ranks), 'gpus': str(ranks if implementation == 'mpi_cuda' else 0),
                'validation': 'pass' if validated else 'not_run'}
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError('Solver record does not match the requested configuration or validation state')
    if not math.isclose(float(row['r']), 0.1, rel_tol=0, abs_tol=1e-15):
        raise ValueError('Unexpected diffusion coefficient')
    elapsed = float(row['elapsed_s'])
    if not math.isfinite(elapsed) or elapsed < 0 or (steps > 0 and elapsed == 0):
        raise ValueError('Invalid elapsed time')
    for field in ['max_abs_error', 'relative_l2_error']:
        if validated:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid validation metric')
        elif row[field] != 'NA':
            raise ValueError('Unvalidated run must retain NA error metrics')
    return row


def parse_profile(path, n, steps, ranks, block_x, block_y):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames != PROFILE_FIELDS or len(rows) != ranks:
            raise ValueError('Profile must contain exactly one row per rank with the documented header')
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Incomplete profile row')
        rank = int(row['rank'])
        if rank in seen or not 0 <= rank < ranks or int(row['device']) < 0:
            raise ValueError('Invalid or duplicated profile rank/device')
        seen.add(rank)
        for key, expected in [('n', n), ('steps', steps), ('block_x', block_x), ('block_y', block_y)]:
            if int(row[key]) != expected:
                raise ValueError('Profile configuration mismatch')
        for key in PROFILE_FIELDS[6:]:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError('Nonfinite or negative profile duration')
    return rows


def configurations(mode, n, sizes, ranks):
    if mode == 'size':
        if not sizes or len(ranks) != 1:
            raise ValueError('Size sweep requires --sizes and exactly one --ranks value')
        return [(size, ranks[0]) for size in sizes]
    if n is None or sizes:
        raise ValueError('Strong/weak scaling requires --n and does not use --sizes')
    if 1 not in ranks:
        raise ValueError('Include one rank/one GPU as the scaling baseline')
    return [(n if mode == 'strong' else int(math.floor(n * math.sqrt(p) + 0.5)), p)
            for p in ranks]


def solver_command(args, n, steps, ranks, extra=()):
    return (shlex.split(args.launcher) + shlex.split(args.launcher_args)
            + ['-np', str(ranks), str(args.exe.resolve()), '--n', str(n), '--steps', str(steps),
               '--block-x', str(args.block_x), '--block-y', str(args.block_y)] + list(extra))


def run_logged(command, prefix, timeout, env=None):
    """Retain diagnostics; terminate the launcher process group on timeout on Linux."""
    prefix = Path(prefix)
    Path(str(prefix) + '.command.json').write_text(
        json.dumps({'argv': command, 'cwd': str(Path.cwd()), 'timeout_s': timeout}, indent=2),
        encoding='utf-8')
    stdout_path = Path(str(prefix) + '.stdout')
    stderr_path = Path(str(prefix) + '.stderr')
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   start_new_session=(os.name == 'posix'))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return stdout_path.read_text(encoding='utf-8')


def command_info(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {'command': command, 'exit_code': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'error': str(error)}


def collect_metadata(args, commands):
    environment = execution_environment(getattr(args, 'local', False))
    nodefile = os.environ.get('PBS_NODEFILE') if environment == 'pbs' else None
    git_root = command_info(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'])
    own_repository = (git_root.get('exit_code') == 0 and
                      Path(git_root.get('stdout', '').strip()).resolve() == ROOT.resolve())
    source_files = [ROOT / 'Makefile.cuda']
    for folder, pattern in [('src', '*'), ('include', '*.h'), ('scripts', '*cuda*'), ('analysis', 'cuda_analysis.py')]:
        source_files.extend(path for path in (ROOT / folder).glob(pattern) if path.is_file())
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(set(source_files)) if path.is_file()}
    result = {
        'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'running', 'hostname': socket.gethostname(),
        'execution_environment': environment,
        'data_origin_note': origin_note(environment),
        'operating_system': platform.platform(),
        'kernel': platform.release(),
        'python_version': platform.python_version(),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'commands': commands, 'binary_sha256': hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        'git_root': git_root,
        'source_revision': (command_info(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
                            if own_repository else {'unavailable': 'This source snapshot is not its own Git repository; a parent repository SHA is not its revision.'}),
        'working_tree': (command_info(['git', '-C', str(ROOT), 'status', '--short'])
                         if own_repository else None),
        'source_sha256': source_hashes,
        'mpi_version': command_info(shlex.split(args.launcher) + ['--version']),
        'nvcc_version': command_info(['nvcc', '--version']),
        'mpi_compiler': command_info(['mpicxx', '--version']),
        'gpu_on_launch_host': command_info(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv']),
        'cpu_on_launch_host': command_info(['lscpu']),
        'modules': os.environ.get('LOADEDMODULES', ''),
        'pbs': ({key: os.environ[key] for key in ['PBS_JOBID', 'PBS_QUEUE', 'PBS_NP', 'PBS_NCPUS', 'PBS_NODEFILE'] if key in os.environ}
                if environment == 'pbs' else {}),
        'job_details': (command_info(['qstat', '-f', os.environ['PBS_JOBID']])
                        if environment == 'pbs' else None),
        'timing_note': ('Profile mode adds synchronizations; never mix with ordinary timing.' if args.profile else 'Ordinary complete-iteration time; no per-stage profiling.'),
        'validation_note': 'Small-grid gate runs separately. Each measured record retains its own actual validation state.',
        'hardware_note': 'Hardware queries describe the launch host only; solver stderr records per-rank mapping. Preserve per-node hardware records for multinode comparisons.',
    }
    try:
        result['nodefile'] = Path(nodefile).read_text() if nodefile else None
    except OSError as error:
        result['nodefile_error'] = str(error)
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['strong', 'weak', 'size'], default='strong')
    parser.add_argument('--n', type=int, help='Fixed N or the one-GPU weak-scaling N')
    parser.add_argument('--sizes', nargs='+', type=int, help='Grid sizes for mode=size')
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--ranks', nargs='+', type=int, default=[1])
    parser.add_argument('--allocated-gpus', type=int, required=True, help='Actual GPUs granted to this job, or exclusively available for --local')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--block-x', type=int, default=32)
    parser.add_argument('--block-y', type=int, default=8)
    parser.add_argument('--exe', type=Path, default=ROOT / 'heat_mpi_cuda')
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--launcher-args', default='', help='Site-approved launcher arguments, shell quoted as one value')
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory; never overwrite an earlier run')
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--validate-measurements', action='store_true', help='Also run full-grid reference outside every measured timed region')
    parser.add_argument('--profile', action='store_true', help='Separate instrumented experiment; writes per-rank stage timings')
    parser.add_argument('--local', action='store_true', help='Use your own local GPU and label all metadata as local; never use on a cluster login node')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        environment = execution_environment(args.local, require_allocation=False)
        configs = configurations(args.mode, args.n, args.sizes, args.ranks)
        if not 0 < args.steps <= 2147483647 or args.repeats < 5 or args.timeout <= 0:
            raise ValueError('Use positive steps/timeout and at least five measured repetitions')
        if args.allocated_gpus < 1 or any(p < 1 or p > args.allocated_gpus for _, p in configs):
            raise ValueError('Every rank needs an exclusive allocated GPU')
        if len(set(configs)) != len(configs):
            raise ValueError('Duplicate configurations are not allowed')
        if not (1 <= args.block_x <= 1024 and 1 <= args.block_y <= 1024 and args.block_x * args.block_y <= 1024):
            raise ValueError('Invalid block shape; device-specific limits are checked by the solver')
        if any(n < 4 or p > n or n > 2147483647 for n, p in configs):
            raise ValueError('Require 4 <= N <= INT_MAX and ranks <= N')
        if not shlex.split(args.launcher):
            raise ValueError('Launcher must not be empty')
    except ValueError as error:
        parser.error(str(error))
    commands = [solver_command(args, n, args.steps, p) for n, p in configs]
    print(origin_note(environment), flush=True)
    if args.dry_run:
        for (n, p), command in zip(configs, commands):
            gate = solver_command(args, max(17, p + 1), 7, p, ['--validate'])
            print('Correctness gate: ' + shlex.join(gate))
            print('1 warm-up + {} measured ({}): {}'.format(args.repeats, 'profile' if args.profile else 'ordinary', shlex.join(command)))
        print('No process launched and no output directory created.')
        return 0
    try:
        execution_environment(args.local)
    except ValueError as error:
        parser.error(str(error))
    if not args.exe.is_file():
        parser.error('Compile heat_mpi_cuda first')
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    metadata = collect_metadata(args, commands)
    metadata_path = args.output / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summaries = []
    try:
        with (args.output / 'raw.csv').open('w', newline='', encoding='utf-8') as handle, (args.output / 'gates.csv').open('w', newline='', encoding='utf-8') as gate_handle:
            writer = csv.DictWriter(handle, fieldnames=EXTRA_FIELDS + FIELDS)
            writer.writeheader()
            gates = csv.DictWriter(gate_handle, fieldnames=FIELDS)
            gates.writeheader()
            for index, (n, p) in enumerate(configs):
                gate_n = max(17, p + 1)
                gate_command = solver_command(args, gate_n, 7, p, ['--validate'])
                gate_text = run_logged(gate_command, args.output / ('gate_{}'.format(index)), args.timeout, env)
                gates.writerow(parse_record(gate_text, gate_n, 7, p, True))
                gate_handle.flush()
                times = []
                for repeat in range(args.repeats + 1):
                    phase = 'warmup' if repeat == 0 else 'measured'
                    stem = 'n{}_p{}_r{}'.format(n, p, repeat)
                    profile_file = stem + '.profile.csv' if args.profile else ''
                    extra = ['--validate'] if args.validate_measurements else []
                    if args.profile:
                        extra += ['--profile-output', str((args.output / profile_file).resolve())]
                    command = solver_command(args, n, args.steps, p, extra)
                    print('N={} ranks={} {} {}'.format(n, p, phase, repeat), flush=True)
                    result = run_logged(command, args.output / stem, args.timeout, env)
                    row = parse_record(result, n, args.steps, p, args.validate_measurements)
                    if args.profile:
                        parse_profile(args.output / profile_file, n, args.steps, p, args.block_x, args.block_y)
                    writer.writerow(dict(mode=args.mode, timing_mode='profile' if args.profile else 'ordinary', phase=phase,
                                         repeat=repeat, block_x=args.block_x, block_y=args.block_y, profile_file=profile_file, **row))
                    handle.flush()
                    if repeat:
                        times.append(float(row['elapsed_s']))
                summaries.append(dict(mode=args.mode, timing_mode='profile' if args.profile else 'ordinary', n=n, steps=args.steps,
                                      ranks=p, gpus=p, block_x=args.block_x, block_y=args.block_y, measured_repeats=len(times),
                                      median_s=statistics.median(times), min_s=min(times), max_s=max(times)))
        with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        metadata['status'] = 'complete'
    except Exception as error:
        metadata['status'] = 'failed'
        metadata['error'] = str(error)
        raise
    finally:
        metadata['utc_end'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Completed: {}'.format(args.output))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        sys.exit(1)
