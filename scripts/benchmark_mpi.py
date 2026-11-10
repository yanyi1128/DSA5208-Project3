"""Run one MPI scaling curve inside a CPU allocation; no scheduler submission."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIELDS = 'implementation,n,steps,r,ranks,gpus,elapsed_s,validation,max_abs_error,relative_l2_error'.split(',')


def grid_size(mode, base, ranks):
    # Exact integer rounding of base*sqrt(ranks), including upward ties.
    if mode == 'strong':
        return base
    squared = base * base * ranks
    low = math.isqrt(squared)
    return low + (4 * squared >= (2 * low + 1) ** 2)


def parse_record(stdout, n, steps, ranks, validate):
    reader = csv.DictReader(io.StringIO(stdout))
    rows = list(reader)
    if reader.fieldnames != FIELDS or len(rows) != 1:
        raise ValueError('Expected exactly one solver CSV record with SPEC header')
    row = rows[0]
    if (row['implementation'] != 'mpi' or int(row['n']) != n
            or int(row['steps']) != steps or int(row['ranks']) != ranks
            or int(row['gpus']) != 0 or float(row['r']) != 0.1):
        raise ValueError('Solver configuration does not match requested configuration')
    elapsed = float(row['elapsed_s'])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Expected finite positive elapsed time')
    if row['validation'] != ('pass' if validate else 'not_run'):
        raise ValueError('Unexpected validation status')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['strong', 'weak'], required=True)
    parser.add_argument('--n', type=int, required=True, help='Fixed N for strong; one-rank N0 for weak')
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', required=True)
    parser.add_argument('--allocated-cores', type=int, required=True,
                        help='Actual CPU cores granted to this job, not login-node cores')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--exe', type=Path, default=ROOT / 'heat_mpi')
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--launcher-args', default='--bind-to core --map-by slot')
    parser.add_argument('--timeout', type=int, default=3600, help='Seconds per MPI launch')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths rejected')
    parser.add_argument('--validate', action='store_true', help='Small experiments only: serial reference uses root memory/time')
    parser.add_argument('--dry-run', action='store_true', help='Print plan only, without running or writing')
    args = parser.parse_args()
    if (args.n < 4 or args.steps < 1 or args.repeats < 5 or args.timeout < 1
            or args.allocated_cores < 1 or not args.ranks
            or min(args.ranks) < 1 or max(args.ranks) > args.allocated_cores
            or len(set(args.ranks)) != len(args.ranks)):
        parser.error('Check N, positive steps/ranks, >=5 repeats, and allocated-core limit')
    if 1 not in args.ranks:
        parser.error('Include rank count 1 for the baseline of this curve')
    sizes = [grid_size(args.mode, args.n, p) for p in args.ranks]
    if any(n < p or n > 2147483647 for n, p in zip(sizes, args.ranks)):
        parser.error('Every configuration must satisfy ranks <= N <= INT_MAX')
    if args.validate and any(n*n > 2147483647 for n in sizes):
        parser.error('Validation requires N*N <= INT_MAX')
    launcher = shlex.split(args.launcher) + shlex.split(args.launcher_args)
    commands = [launcher + ['-np', str(p), str(args.exe.resolve()), '--n', str(n),
                           '--steps', str(args.steps)] + (['--validate'] if args.validate else [])
                for n, p in zip(sizes, args.ranks)]
    if args.dry_run:
        for command in commands:
            print('1 warm-up + {} measured: {}'.format(args.repeats, shlex.join(command)))
        return
    if not os.environ.get('PBS_JOBID'):
        parser.error('Run inside a PBS CPU allocation; --dry-run is allowed on the login node')
    if not args.exe.is_file():
        parser.error('Compile the Linux MPI executable first')
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def info(command):
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=15)
            return {'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {'error': str(error)}
    nodefile = os.environ.get('PBS_NODEFILE')
    metadata = {
        'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hostname': socket.gethostname(), 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'commands': commands, 'binary_sha256': hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        'revision': info(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
        'working_tree': info(['git', '-C', str(ROOT), 'status', '--short']),
        'mpi_version': info(shlex.split(args.launcher) + ['--version']),
        'compiler': info(['mpic++', '--version']), 'cpu': info(['lscpu']),
        'modules': os.environ.get('LOADEDMODULES', ''),
        'pbs': {k:v for k,v in os.environ.items() if k in ['PBS_JOBID','PBS_QUEUE','PBS_NP','PBS_NCPUS','PBS_NODEFILE']},
        'nodefile': Path(nodefile).read_text() if nodefile else None,
        'job_details': info(['qstat', '-f', os.environ['PBS_JOBID']]),
        'validation_note': 'Each run validated' if args.validate else 'Not validated in this experiment; retain separate correctness evidence',
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary = []
    with (args.output / 'raw.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['mode','phase','repeat'] + FIELDS)
        writer.writeheader()
        for n, ranks, command in zip(sizes, args.ranks, commands):
            times = []
            for repeat in range(args.repeats + 1):
                phase = 'warmup' if repeat == 0 else 'measured'
                print('N={} ranks={} {} {}'.format(n, ranks, phase, repeat), flush=True)
                prefix = args.output / 'p{}_{}'.format(ranks, repeat)
                # Retain stderr/partial output even when a run fails or times out.
                with prefix.with_suffix('.stdout').open('w') as out, prefix.with_suffix('.stderr').open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, env=env,
                                   timeout=args.timeout, check=True)
                row = parse_record(prefix.with_suffix('.stdout').read_text(), n, args.steps, ranks, args.validate)
                writer.writerow(dict(mode=args.mode, phase=phase, repeat=repeat, **row))
                handle.flush()
                if repeat:
                    times.append(float(row['elapsed_s']))
            base, extra = divmod(n, ranks)
            work = []
            for rank in range(ranks):
                start = rank*base + min(rank, extra)
                end = start + base + (rank < extra)
                work.append(max(0, min(end, n-1)-max(start, 1))*(n-2))
            summary.append(dict(mode=args.mode, n=n, steps=args.steps, ranks=ranks,
                                median_s=statistics.median(times), min_s=min(times), max_s=max(times),
                                min_interior_cells=min(work), max_interior_cells=max(work)))
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print('Completed: {}'.format(args.output))


if __name__ == '__main__':
    main()
