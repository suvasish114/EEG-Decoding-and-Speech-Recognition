#!/usr/bin/env python3
"""Compare speech versus music spectral responses at individual EEG electrodes.

Install: python -m pip install numpy scipy matplotlib
Run: python difference.py --dataset processed_eeg --bids ds003688

Uses all processed splits for EXPLORATORY analysis, not classifier evaluation.
Requires film EEG paired with matching film events (--events-task same during
processing). 'Acoustic' here means music; speech also contains acoustic content.

Welch band power is computed per original event/electrode in microvolts squared,
then log10 transformed. Adjacent, contiguous music/speech events within the same
recording form nonoverlapping pairs. A sign-flip test of paired log-power
contrasts is exact for <=16 pairs, Monte Carlo otherwise (seeded, add-one p).
Benjamini-Hochberg FDR is applied across all tested electrode/band combinations
WITHIN each participant. These are exploratory tests: adjacent-pair sign flips
assume independent, exchangeable paired contrasts under the null. Fixed stimulus
order, temporal dependence and a single film limit generalization. Channels are
not independent biological replicates; no group-level inference is performed.

Positive contrast means stronger speech power. 10*contrast is the speech/music
difference in dB; a unit log10 contrast is a tenfold geometric power ratio.
Channel names must match the clinical electrodes TSV exactly. Bad/non-neural
channels are excluded when a corresponding channels TSV exists; unmatched names
are reported without inventing locations. No bipolar midpoint inference is used.
Spatial plots are per participant in the supplied coordinate system, without
brain surfaces, atlas labels, or assumptions of cross-participant registration.
50 Hz harmonics +/-1 Hz are excluded from power estimates by default; use
--line-frequency 60 or 0 as appropriate for your recordings. No normalization
removes amplitude differences. Unequal event durations are allowed by Welch.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import welch

BANDS = [('delta', 1, 4), ('theta', 4, 8), ('alpha', 8, 13),
         ('beta', 13, 30), ('low_gamma', 30, 70), ('high_gamma', 70, 150)]


def read_tsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bandpower(signal, rate, line_frequency):
    if signal.ndim != 2 or signal.shape[0] < 1 or signal.shape[1] < 4 or not np.isfinite(signal).all():
        raise ValueError('Expected finite channels x samples EEG')
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Sampling rate must be positive')
    freq, psd = welch(signal.astype(np.float64) * 1e6, fs=rate,
                      nperseg=min(signal.shape[1], max(4, round(2 * rate))), axis=-1)
    df = freq[1] - freq[0]
    clean = np.ones(len(freq), dtype=bool)
    if line_frequency:
        for harmonic in np.arange(line_frequency, rate / 2 + 1, line_frequency):
            clean &= np.abs(freq - harmonic) > 1
    features = np.full((len(signal), len(BANDS)), np.nan)
    for i, (_, lo, hi) in enumerate(BANDS):
        mask = (freq >= lo) & (freq < hi) & clean
        # Do not compare partial bands across recordings with different Nyquists.
        if hi <= rate / 2 and mask.any():
            power = psd[:, mask].sum(axis=1) * df
            valid = power > 0
            features[valid, i] = np.log10(power[valid])
    return features


def sign_flip_p(values, rng, permutations):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    if len(values) <= 16:
        signs = np.asarray(list(itertools.product([-1., 1.], repeat=len(values))))
        return float(np.mean(np.abs(signs @ values / len(values)) >= observed - 1e-12))
    hits = 0
    for start in range(0, permutations, 1024):
        n = min(1024, permutations - start)
        signs = rng.choice([-1., 1.], size=(n, len(values)))
        hits += np.sum(np.abs(signs @ values / len(values)) >= observed - 1e-12)
    return float((hits + 1) / (permutations + 1))


def fdr(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent / 'processed_eeg')
    parser.add_argument('--bids', type=Path, default=Path(__file__).parent / 'ds003688')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'electrode_differences')
    parser.add_argument('--min-pairs', type=int, default=3)
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--line-frequency', type=float, default=50)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--spatial-threshold-db', type=float, default=4.0,
                        help='Minimum absolute dB contrast shown in the combined spatial plot')
    parser.add_argument('--brain-reference', type=Path,
                        help='Brain render to use as an illustrative backdrop for the combined maps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.brain_reference is not None and not args.brain_reference.is_file():
        parser.error(f'Brain reference image is unavailable: {args.brain_reference}')
    if not math.isfinite(args.spatial_threshold_db) or args.spatial_threshold_db < 0:
        parser.error('--spatial-threshold-db must be finite and nonnegative')
    if args.min_pairs < 2 or args.permutations < 1 or not math.isfinite(args.alpha) or not 0 < args.alpha < 1 or not math.isfinite(args.line_frequency) or args.line_frequency < 0:
        parser.error('Require min-pairs >=2, permutations >=1, 0<alpha<1 and line-frequency >=0')
    if args.output.exists():
        parser.error(f'{args.output} exists; choose a new --output folder')
    paths = sorted(p for split in ('train', 'validation', 'test') for p in (args.dataset / split).glob('*.npz'))
    if not paths:
        parser.error('No processed NPZ files found; run data_processing.py --events-task same first')
    locations, coordinates, channel_metadata = {}, {}, {}
    recordings, event_rows, unmatched, excluded = {}, [], set(), []
    seen = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            label = str(data['label'].item()).strip().lower()
            task = str(data['task'].item())
            if task != 'film' or label not in ('speech', 'music'):
                excluded.append({'file': str(path), 'reason': 'not a film speech/music segment'})
                continue
            subject = str(data['subject'].item())
            run = str(data['run'].item())
            source_eeg = str(data['source_eeg'].item())
            source_events = str(data['source_events'].item())
            expected_events = f'{subject}_ses-iemu_task-film_run-{run}_events.tsv'
            expected_eeg = f'{subject}_ses-iemu_task-film_acq-clinical_run-{run}_ieeg.eeg'
            if Path(source_events).name != expected_events or Path(source_eeg).name != expected_eeg:
                parser.error(f'{path}: labels and EEG must originate from matching film acquisitions')
            onset, duration = float(data['onset'].item()), float(data['duration'].item())
            if not math.isfinite(onset) or not math.isfinite(duration) or duration <= 0:
                parser.error(f'{path}: invalid event timing')
            key = (subject, run, int(data['event_row'].item()))
            if key in seen:
                parser.error(f'Duplicate original event in corpus: {key}')
            seen.add(key)
            names = [str(n) for n in data['channel_names']]
            signal = data['signal']
            if len(names) != signal.shape[0] or len(set(names)) != len(names):
                parser.error(f'{path}: inconsistent/duplicate channel names')
            directory = args.bids / subject / 'ses-iemu' / 'ieeg'
            if subject not in locations:
                electrode_file = directory / f'{subject}_ses-iemu_acq-clinical_electrodes.tsv'
                if not electrode_file.is_file():
                    parser.error(f'Missing electrode locations: {electrode_file}')
                locations[subject] = {row['name']: row for row in read_tsv(electrode_file)}
                coord_file = directory / f'{subject}_ses-iemu_acq-clinical_coordsystem.json'
                coordinates[subject] = json.loads(coord_file.read_text()) if coord_file.is_file() else {}
            if (subject, run) not in channel_metadata:
                channel_file = directory / expected_eeg.replace('_ieeg.eeg', '_channels.tsv')
                channel_metadata[subject, run] = {row['name']: row for row in read_tsv(channel_file)} if channel_file.is_file() else {}
            metadata = channel_metadata[subject, run]
            selected = []
            selected_names = []
            for i, name in enumerate(names):
                if name not in locations[subject]:
                    unmatched.add((subject, run, name))
                    continue
                info = metadata.get(name, {})
                if info.get('status', '').lower() == 'bad' or info.get('type', '').upper() not in ('', 'SEEG', 'ECOG', 'EEG'):
                    continue
                selected.append(i)
                selected_names.append(name)
            features = bandpower(signal[selected], float(data['sampling_rate'].item()), args.line_frequency) if selected else np.empty((0, len(BANDS)))
            event = dict(onset=onset, duration=duration, label=label,
                         features=dict(zip(selected_names, features)), file=str(path.absolute()))
            recordings.setdefault((subject, run), []).append(event)
            for name, values in event['features'].items():
                for (band, _, _), value in zip(BANDS, values):
                    event_rows.append(dict(subject=subject, run=run, electrode=name, label=label,
                                           onset=onset, duration=duration, band=band,
                                           log10_power_uv2=float(value), file=event['file']))
    contrasts, pairs, unpaired = {}, [], []
    for (subject, run), events in sorted(recordings.items()):
        events.sort(key=lambda e: e['onset'])
        i = 0
        while i < len(events):
            first = events[i]
            second = events[i + 1] if i + 1 < len(events) else None
            if second is not None and first['label'] != second['label'] and abs(first['onset'] + first['duration'] - second['onset']) <= 0.1:
                speech, music = (first, second) if first['label'] == 'speech' else (second, first)
                for name in sorted(set(speech['features']) & set(music['features'])):
                    for j, (band, _, _) in enumerate(BANDS):
                        s, m = speech['features'][name][j], music['features'][name][j]
                        if np.isfinite(s) and np.isfinite(m):
                            contrast = float(s - m)
                            contrasts.setdefault((subject, name, band), []).append((contrast, float(s), float(m)))
                            pairs.append(dict(subject=subject, run=run, electrode=name, band=band,
                                              speech_onset=speech['onset'], music_onset=music['onset'],
                                              speech_file=speech['file'], music_file=music['file'],
                                              speech_minus_music_log10=contrast))
                i += 2
            else:
                unpaired.append(first['file'])
                i += 1
    if not contrasts:
        parser.error('No adjacent speech/music pairs with matched electrodes. Check labels, locations and event timing.')
    rng = np.random.default_rng(args.seed)
    results = []
    for (subject, name, band), values in sorted(contrasts.items()):
        values = np.asarray(values)
        diff = values[:, 0]
        loc = locations[subject][name]
        def number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('nan')
        results.append(dict(subject=subject, electrode=name, band=band, n_pairs=len(diff),
                            speech_mean_log10_power_uv2=float(values[:, 1].mean()),
                            music_mean_log10_power_uv2=float(values[:, 2].mean()),
                            speech_minus_music_db=float(10 * diff.mean()),
                            paired_contrast_sd_db=float(10 * diff.std(ddof=1)) if len(diff) > 1 else float('nan'),
                            p_value=sign_flip_p(diff, rng, args.permutations) if len(diff) >= args.min_pairs else float('nan'),
                            q_value=float('nan'), significant_fdr=False,
                            x=number(loc.get('x')), y=number(loc.get('y')), z=number(loc.get('z')),
                            coordinate_system=coordinates[subject].get('iEEGCoordinateSystem', 'unknown'),
                            coordinate_units=coordinates[subject].get('iEEGCoordinateUnits', 'unknown')))
    for subject in sorted(locations):
        tested = [row for row in results if row['subject'] == subject and np.isfinite(row['p_value'])]
        for row, q in zip(tested, fdr([row['p_value'] for row in tested])):
            row['q_value'], row['significant_fdr'] = float(q), bool(q <= args.alpha)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'electrode_summary.csv', results)
    write_csv(args.output / 'event_features.csv', event_rows)
    write_csv(args.output / 'paired_contrasts.csv', pairs)
    write_csv(args.output / 'unmatched_channels.csv', [dict(subject=s, run=r, channel=n) for s, r, n in sorted(unmatched)], ['subject', 'run', 'channel'])
    report = dict(contrast='speech minus music; music is the acoustic comparator',
                  analysis='exploratory, all corpus splits combined; not classifier evaluation',
                  fdr_family='all electrode-band tests within participant',
                  assumptions='sign-flip tests assume independent exchangeable paired contrasts; fixed stimulus order and temporal dependence limit inference',
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  coordinate_metadata=coordinates, excluded=excluded, unpaired_files=unpaired,
                  unmatched_channels=len(unmatched), tested_hypotheses=sum(np.isfinite(r['p_value']) for r in results),
                  significant_hypotheses=sum(r['significant_fdr'] for r in results))
    # Cast numpy integer counts for JSON serialization.
    report['tested_hypotheses'] = int(report['tested_hypotheses'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    plot_results(results, args.output)
    plot_combined_spatial(results, args.output, args.spatial_threshold_db, args.brain_reference)
    print(f"Saved {len(results)} electrode/band comparisons; {report['significant_hypotheses']} pass FDR. Output: {args.output.absolute()}")
    return 0


def plot_results(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for subject in sorted({r['subject'] for r in results}):
        rows = [r for r in results if r['subject'] == subject]
        names = sorted({r['electrode'] for r in rows})
        bands = [b[0] for b in BANDS if any(r['band'] == b[0] for r in rows)]
        matrix = np.full((len(names), len(bands)), np.nan)
        significant = np.zeros_like(matrix, dtype=bool)
        for row in rows:
            i, j = names.index(row['electrode']), bands.index(row['band'])
            matrix[i, j] = row['speech_minus_music_db']
            significant[i, j] = row['significant_fdr']
        bound = max(float(np.nanmax(np.abs(matrix))), 0.1)
        fig, ax = plt.subplots(figsize=(9, max(5, len(names) * .18)))
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_xticks(range(len(bands)), bands, rotation=30, ha='right')
        yy, xx = np.where(significant)
        ax.scatter(xx, yy, marker='*', s=15, color='black')
        ax.set_title(f'{subject}: speech − music band power\n* within-participant FDR significant; exploratory')
        fig.colorbar(image, ax=ax, label='Speech − music (dB)')
        fig.tight_layout()
        fig.savefig(output / f'{subject}_electrode_heatmap.png', dpi=150)
        plt.close(fig)
        fig = plt.figure(figsize=(14, 4 * math.ceil(len(bands) / 3)))
        for j, band in enumerate(bands):
            ax = fig.add_subplot(math.ceil(len(bands) / 3), 3, j + 1, projection='3d')
            located = [r for r in rows if r['band'] == band and all(np.isfinite(r[c]) for c in ('x', 'y', 'z'))]
            if located:
                image = ax.scatter([r['x'] for r in located], [r['y'] for r in located], [r['z'] for r in located],
                                   c=[r['speech_minus_music_db'] for r in located], cmap='RdBu_r', vmin=-bound, vmax=bound,
                                   s=[65 if r['significant_fdr'] else 20 for r in located])
                for r in sorted(located, key=lambda r: abs(r['speech_minus_music_db']), reverse=True)[:3]:
                    ax.text(r['x'], r['y'], r['z'], r['electrode'], fontsize=7)
                fig.colorbar(image, ax=ax, shrink=.5, label='dB')
                unit = located[0]['coordinate_units']
                ax.set(xlabel=f'x ({unit})', ylabel=f'y ({unit})', zlabel=f'z ({unit})')
            else:
                ax.text2D(.1, .5, 'No finite coordinates', transform=ax.transAxes)
            ax.set_title(band)
        system = rows[0]['coordinate_system']
        fig.suptitle(f'{subject}: speech − music | {system} coordinates\nLarger markers: FDR significant; labels: three largest absolute contrasts per band')
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(output / f'{subject}_spatial_differences.png', dpi=150)
        plt.close(fig)


def plot_combined_spatial(results, output, threshold_db=4.0, brain_reference=None):
    """Overlay contacts for descriptive coverage in a shared coordinate convention."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    located = [r for r in results if all(np.isfinite(r[key]) for key in
               ('x', 'y', 'z', 'speech_minus_music_db'))]
    conventions = {(r['coordinate_system'], r['coordinate_units']) for r in located}
    if len(conventions) != 1:
        return
    system, units = conventions.pop()
    if system in ('', 'unknown', 'n/a') or units in ('', 'unknown', 'n/a'):
        return

    bound = max(abs(r['speech_minus_music_db']) for r in located) or 1
    participants = len({r['subject'] for r in located})
    highlighted = [r for r in located if abs(r['speech_minus_music_db']) >= threshold_db]
    xmin, xmax = min(r['x'] for r in located) - 5, max(r['x'] for r in located) + 5
    ymin, ymax = min(r['y'] for r in located) - 5, max(r['y'] for r in located) + 5
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharex=True, sharey=True,
                             layout='constrained')
    if brain_reference:
        reference = plt.imread(brain_reference)
        if reference.ndim == 3:
            reference = reference[:, :, :3].mean(axis=2)
        y, x = np.where(reference < reference.max() * .9)
        if len(x):
            reference = reference[y.min():y.max() + 1, x.min():x.max() + 1]
        height, width = reference.shape
        scale = min((xmax - xmin) / width, (ymax - ymin) / height)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        dx, dy = width * scale / 2, height * scale / 2
        for ax in axes.flat:
            ax.imshow(reference, extent=(cx - dx, cx + dx, cy - dy, cy + dy),
                      cmap='gray', alpha=.32, zorder=0)
    for ax, (band, lo, hi) in zip(axes.flat, BANDS):
        rows = [r for r in highlighted if r['band'] == band]
        image = ax.scatter([r['x'] for r in rows], [r['y'] for r in rows],
                           c=[r['speech_minus_music_db'] for r in rows],
                           s=42, edgecolors='#303030', linewidths=.5, cmap='RdBu_r',
                           vmin=-bound, vmax=bound)
        ax.set_title(f'{band.replace("_", " ").title()} ({lo}–{hi} Hz) · n={len(rows)}')
        ax.set_aspect('equal')
    axes[0, 0].set_xlim(xmin, xmax)
    axes[0, 0].set_ylim(ymin, ymax)
    for ax in axes[-1]:
        ax.set_xlabel(f'x ({units})')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'y ({units})')
    backdrop_note = '\nIllustrative brain backdrop · No anatomical registration' if brain_reference else ''
    fig.suptitle(f'Speech − music band power across {participants} participants\n'
                 f'{system} x–y projection · |Contrast| ≥ {threshold_db:g} dB · Effect-size cutoff{backdrop_note}')
    fig.colorbar(image, ax=axes, label='Speech − music (dB)', shrink=.8)
    fig.savefig(output / 'combined_spatial_differences.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
