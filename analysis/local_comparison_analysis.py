"""Plot the labelled local experiments from curated CSVs (not Vanda data)."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/local_20261116'))
    parser.add_argument('--output', type=Path, default=Path('results/local_20261116/figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = sorted(read(args.input / 'comparison_summary.csv'), key=lambda r: int(r['n']))
    n = [int(r['n']) for r in rows]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def axes(title):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(n, labels=[str(x) for x in n])
        ax.set_xlabel('Grid side N (including boundary cells)')
        ax.set_ylabel('Complete-iteration solve time (seconds)')
        ax.set_title(title, fontsize=11)
        ax.grid(True, which='major', alpha=.25)
        return fig, ax

    def line(ax, key, label, colour):
        m = [float(r[key + '_median_s']) for r in rows]
        lo = [float(r[key + '_median_s']) - float(r[key + '_min_s']) for r in rows]
        hi = [float(r[key + '_max_s']) - float(r[key + '_median_s']) for r in rows]
        ax.errorbar(n, m, yerr=[lo, hi], fmt='o-', capsize=4, color=colour, label=label)

    fig, ax = axes('LOCAL serial C++: Ryzen 9 5900HX, K=1000\nMedian of five runs; bars show measured min/max')
    line(ax, 'serial', 'Serial, one CPU thread', '#1261a0')
    ref = [float(rows[0]['serial_median_s']) * (x / n[0]) ** 2 for x in n]
    ax.plot(n, ref, '--', color='#666666', label='N squared guide (anchored at N=256)')
    ax.legend(fontsize=9)
    fig.savefig(args.output / 'local_serial_size.png', dpi=220)
    plt.close(fig)

    fig, ax = axes('LOCAL paired serial / one-GPU experiment, K=1000\nRyzen 9 5900HX / RTX 3070 Laptop; median and min/max')
    line(ax, 'serial', 'Serial, one CPU thread', '#1261a0')
    line(ax, 'mpi_cuda', 'MPI+CUDA, one rank / GPU, block 32x8', '#b45f06')
    ax.legend(fontsize=9)
    fig.savefig(args.output / 'local_serial_gpu_comparison.png', dpi=220)
    plt.close(fig)

    blocks = read(args.input / 'block_summary.csv')
    fig, ax = axes('LOCAL laptop GPU: RTX 3070 Laptop, 1 GPU, K=1000\nMedian of five runs; bars show measured min/max')
    for bx, by, colour in [('32', '8', '#1261a0'), ('16', '16', '#b45f06')]:
        selected = sorted([r for r in blocks if r['block_x'] == bx and r['block_y'] == by], key=lambda r: int(r['n']))
        med = [float(r['median_s']) for r in selected]
        low = [float(r['median_s']) - float(r['min_s']) for r in selected]
        high = [float(r['max_s']) - float(r['median_s']) for r in selected]
        ax.errorbar(n, med, yerr=[low, high], fmt='o-', capsize=4, color=colour, label='Block ' + bx + 'x' + by)
    ax.legend()
    fig.savefig(args.output / 'local_cuda_size.png', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
