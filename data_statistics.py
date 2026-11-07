#!/usr/bin/env python3
"""Count labels in every NPZ below processed_eeg and save a class-count plot.

Run: python data_statistics.py
Options: --dataset PATH --output PATH.png --show
Requires numpy and matplotlib. Each file counts as one labeled segment;
counts do not represent signal samples or recording duration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys


def count_labels(dataset, np):
    """Inspect all NPZ labels, collecting invalid files without hiding them."""
    counts = Counter()
    errors = []
    files = sorted(dataset.rglob('*.npz'))
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                label = data['label']
                if label.size != 1:
                    raise ValueError('expected one class label per file')
                value = label.item()
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                label = str(value).strip()
                if not label or label.lower() in ('n/a', 'nan', 'none'):
                    raise ValueError('missing class label')
                counts[label] += 1
        except Exception as exc:
            errors.append((path, str(exc)))
    return files, counts, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path,
                        default=Path(__file__).parent / 'processed_eeg')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'data_statistics.png')
    parser.add_argument('--show', action='store_true', help='Also open the plot window')
    args = parser.parse_args(argv)
    if not args.dataset.is_dir():
        parser.error(f'Dataset folder does not exist: {args.dataset}')
    if args.output.suffix.lower() != '.png':
        parser.error('--output must end in .png')
    try:
        import numpy as np
        import matplotlib
        if not args.show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error('Install dependencies: python -m pip install numpy matplotlib')

    files, counts, errors = count_labels(args.dataset, np)
    if not files:
        parser.error(f'No NPZ files found in {args.dataset.absolute()}')
    for path, reason in errors:
        print(f'INVALID {path}: {reason}', file=sys.stderr)
    if not counts:
        parser.error('No files with valid labels were found')

    labels = sorted(counts)
    values = [counts[label] for label in labels]
    total = sum(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['label', 'file_count', 'percentage'])
        for label in labels:
            writer.writerow([label, counts[label], f'{100 * counts[label] / total:.2f}'])

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5.5))
    bars = ax.bar(labels, values, color='steelblue', width=0.6)
    ax.bar_label(bars, labels=[f'{v:,}\n({100*v/total:.1f}%)' for v in values], padding=5)
    ax.set(title=f'Class distribution — {args.dataset.name}\n'
                 f'{total:,} labeled segments | {len(labels)} classes | {len(errors)} invalid files',
           xlabel='Class label', ylabel='Number of NPZ files')
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    if len(labels) > 6:
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f'Checked {len(files):,} NPZ files: {total:,} valid labels, {len(errors)} invalid files.')
    for label in labels:
        print(f'  {label}: {counts[label]:,} ({100 * counts[label] / total:.2f}%)')
    print(f'Plot: {args.output.absolute()}')
    print(f'Counts: {csv_path.absolute()}')
    if args.show:
        plt.show()
    plt.close(fig)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
