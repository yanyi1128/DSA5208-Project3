#!/usr/bin/env python3
"""Validate the CUDA executable against fresh CSV fields exported by A's serial code.

Run inside a PBS GPU allocation, or explicitly use --local on your own GPU.
Local results are labeled as local, not Atlas/Vanda. The default uses only one
GPU; list only rank counts supported by the exclusive GPUs actually available.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from benchmark_cuda import (ROOT, FIELDS, collect_metadata, execution_environment,
                            origin_note, parse_record, run_logged, solver_command)


def test_cases(ranks):
    base = [(8, 0), (8, 1), (17, 7), (8, 2), (33, 1), (33, 2), (33, 9), (33, 10), (64, 100)]
    result = []
    for p in ranks:
        cases = [case for case in base if case[0] >= p]
        cases.extend([(max(4, p), 2), (max(4, 2 * p), 2), (max(17, 2 * p + 1), 7)])
        result.extend((n, steps, p) for n, steps in dict.fromkeys(cases))
    return result


def read_field(path, n):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    if len(rows) != n or any(len(row) != n for row in rows):
        raise ValueError('Expected an exact N-by-N CSV field: {}'.format(path))
    values = [float(value) for row in rows for value in row]
    for i in range(n):
        for j in range(n):
            value = values[i * n + j]
            if not math.isfinite(value) or value < -1e-12 or value > 1 + 1e-12:
                raise ValueError('Nonfinite or out-of-range temperature')
            if (i in [0, n - 1] or j in [0, n - 1]) and value != 0:
                raise ValueError('Physical boundary is not exactly zero')
    return values


def check_analytic(values, n, steps):
    if steps == 0:
        lo, hi = n // 4, (3 * n) // 4
        for i in range(n):
            for j in range(n):
                expected = float(lo <= i < hi and lo <= j < hi and 0 < i < n - 1 and 0 < j < n - 1)
                if values[i * n + j] != expected:
                    raise ValueError('Initialization differs from the SPEC')
    if n == 8 and steps == 1:
        for i, j, expected in [(3, 3, 1), (2, 3, 0.9), (2, 2, 0.8), (1, 3, 0.1)]:
            if abs(values[i * n + j] - expected) > 1e-12 + 1e-10 * abs(expected):
                raise ValueError('Independent analytic one-step check failed')


def compare_fields(actual, reference):
    if len(actual) != len(reference):
        raise ValueError('Different field lengths')
    differences = [abs(a - b) for a, b in zip(actual, reference)]
    if any(not math.isfinite(a) or not math.isfinite(b) or error > 1e-12 + 1e-10 * abs(b)
           for a, b, error in zip(actual, reference, differences)):
        raise ValueError('Field differs from serial reference beyond the SPEC tolerance')
    norm = math.fsum(value * value for value in reference)
    error_norm = math.fsum(error * error for error in differences)
    relative = math.sqrt(error_norm / norm) if norm else (0.0 if error_norm == 0 else math.inf)
    return max(differences, default=0.0), relative


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=ROOT / 'heat_mpi_cuda')
    parser.add_argument('--serial-exe', type=Path, default=ROOT / 'heat_serial')
    parser.add_argument('--ranks', nargs='+', type=int, default=[1])
    parser.add_argument('--allocated-gpus', type=int, required=True)
    parser.add_argument('--launcher', default='mpirun')
    parser.add_argument('--launcher-args', default='')
    parser.add_argument('--block-x', type=int, default=32)
    parser.add_argument('--block-y', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--local', action='store_true', help='Use your own local GPU and label evidence as local; never use on a cluster login node')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        environment = execution_environment(args.local, require_allocation=False)
    except ValueError as error:
        parser.error(str(error))
    if (args.allocated_gpus < 1 or not args.ranks or 1 not in args.ranks
            or len(set(args.ranks)) != len(args.ranks)
            or any(p < 1 or p > args.allocated_gpus for p in args.ranks)):
        parser.error('Include rank count 1; request only distinct counts within the allocated GPU count')
    if args.timeout <= 0 or not shlex.split(args.launcher):
        parser.error('Require a launcher and a positive timeout')
    if min(args.block_x, args.block_y) < 1 or args.block_x * args.block_y > 1024:
        parser.error('Invalid CUDA block shape')
    cases = test_cases(args.ranks)
    print(origin_note(environment), flush=True)
    if args.dry_run:
        for n, steps, ranks in cases:
            print('Serial CSV reference: N={} K={}; CUDA ranks={} GPUs={}'.format(n, steps, ranks, ranks))
        print('Also tests invalid CLI arguments and rejection of an intentionally wrong reference.')
        print('No executable launched; no output directory created. Unavailable GPU counts are not tested.')
        return 0
    try:
        execution_environment(args.local)
    except ValueError as error:
        parser.error(str(error))
    if not args.exe.is_file() or not args.serial_exe.is_file():
        parser.error('Build both heat_serial and heat_mpi_cuda first')
    args.output.mkdir(parents=True, exist_ok=False)
    args.profile = False
    metadata = collect_metadata(args, [])
    metadata['serial_binary_sha256'] = hashlib.sha256(args.serial_exe.read_bytes()).hexdigest()
    metadata['scope'] = {'tested_rank_counts': args.ranks, 'cases': cases,
                         'note': 'No claims are made for unavailable GPU/rank configurations.'}
    metadata_path = args.output / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    references = {}
    try:
        with (args.output / 'validation.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS + ['external_max_abs_error', 'external_relative_l2_error'])
            writer.writeheader()
            for n, steps, ranks in cases:
                key = (n, steps)
                if key not in references:
                    ref_path = args.output / 'serial_n{}_k{}.csv'.format(n, steps)
                    command = [str(args.serial_exe.resolve()), '--n', str(n), '--steps', str(steps), '--validate', '--output', str(ref_path.resolve())]
                    output = run_logged(command, args.output / 'serial_n{}_k{}'.format(n, steps), args.timeout, env)
                    parse_record(output, n, steps, 1, True, 'serial')
                    values = read_field(ref_path, n)
                    check_analytic(values, n, steps)
                    references[key] = (ref_path, values)
                ref_path, reference = references[key]
                stem = 'cuda_n{}_k{}_p{}'.format(n, steps, ranks)
                result_path = args.output / (stem + '.csv')
                command = solver_command(args, n, steps, ranks, ['--validate', '--reference', str(ref_path.resolve()), '--output', str(result_path.resolve())])
                output = run_logged(command, args.output / stem, args.timeout, env)
                row = parse_record(output, n, steps, ranks, True)
                actual = read_field(result_path, n)
                check_analytic(actual, n, steps)
                error, relative = compare_fields(actual, reference)
                writer.writerow(dict(row, external_max_abs_error=error, external_relative_l2_error=relative))
                handle.flush()
                print('PASS N={} K={} ranks={}'.format(n, steps, ranks), flush=True)
        launcher = shlex.split(args.launcher) + shlex.split(args.launcher_args) + ['-np', '1', str(args.exe.resolve())]
        invalid = [('--n', '3', '--steps', '1'), ('--n', '8', '--steps', '-1'),
                   ('--n', 'bad', '--steps', '1'), ('--n', '8', '--steps', '1', '--unknown'),
                   ('--n', '8', '--steps', '1', '--block-x', '0')]
        negative_results = []
        for index, options in enumerate(invalid):
            command = launcher + list(options)
            try:
                run_logged(command, args.output / 'invalid_{}'.format(index), args.timeout, env)
            except subprocess.CalledProcessError as error:
                negative_results.append({'command': command, 'exit_code': error.returncode, 'expected_failure': True})
            else:
                raise ValueError('Invalid arguments unexpectedly succeeded: {}'.format(options))
        original_path, original = references[(8, 1)]
        wrong = original[:]
        wrong[1 * 8 + 1] += 0.01
        wrong_path = args.output / 'intentionally_wrong_reference.csv'
        with wrong_path.open('w', newline='', encoding='utf-8') as handle:
            csv.writer(handle).writerows([wrong[i * 8:(i + 1) * 8] for i in range(8)])
        command = solver_command(args, 8, 1, 1, ['--reference', str(wrong_path.resolve())])
        try:
            run_logged(command, args.output / 'wrong_reference', args.timeout, env)
        except subprocess.CalledProcessError as error:
            negative_results.append({'command': command, 'exit_code': error.returncode, 'expected_failure': True})
        else:
            raise ValueError('An intentionally incorrect serial reference unexpectedly passed')
        (args.output / 'negative_tests.json').write_text(json.dumps(negative_results, indent=2), encoding='utf-8')
        metadata['status'] = 'complete'
    except Exception as error:
        metadata['status'] = 'failed'
        metadata['error'] = str(error)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('GPU validation completed: {}'.format(args.output))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        sys.exit(1)
