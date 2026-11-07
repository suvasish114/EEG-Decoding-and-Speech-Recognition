#!/usr/bin/env python3
"""Build paired iEEG/event corpora from ds003688.

Install: python -m pip install numpy mne scipy matplotlib tqdm
Preview: python data_processing.py --dry-run
Build cleaned film dataset: python data_processing.py
Use scientifically matched film labels: python data_processing.py --events-task same
Include film AND rest recordings: python data_processing.py --tasks film rest --events-task same

Each recording uses events from the same subject, session, task and run.
Event onset is the offset in seconds from recording start; the exclusive end
is onset + duration. Only sample timestamps inside that interval are saved.

Each positive-duration event becomes one .npz with signal (channels x samples,
float32 volts), label, event_value, sampling_rate, channel_names and provenance.
Zero-duration and start/end-task markers are excluded by default.
Events crossing recording boundaries are clipped and logged; original timings
are preserved alongside effective onset/duration. Events with no overlap are skipped. Variable channel counts and event lengths
are preserved; batching/padding is the downstream model's responsibility.
Splits are participant-disjoint. Known bad/non-iEEG and flat/nonfinite channels
are removed. Continuous signals are notch and bandpass filtered before epoching.
Long constant plateaus reject overlapping whole events; transients are flagged
for review, not automatically erased. No normalization or resampling is applied. Output manifests pair each signal path with its label and event timing.
BrainVision .vhdr, .vmrk and .eeg companions must be locally available (e.g. via
`datalad get` in a DataLad checkout). Existing output directories are refused.
Default output: processed_eeg_clean (original processed_eeg is preserved).
Existing processed_eeg/summary.json participant assignments are reused when present.
QC plots and JSON logs are saved in quality_control; inspect flagged transients.
The default keeps the acquisition reference. Average and adjacent-contact bipolar
references are optional; choose using the electrode layout and study design.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def discover(root, tasks, events_task):
    records, skipped = [], []
    for subject in sorted(root.glob('sub-*')):
        if not subject.is_dir():
            continue
        session = subject / 'ses-iemu'
        if not session.is_dir():
            skipped.append({'subject': subject.name, 'reason': 'no ses-iemu folder'})
            continue
        for task in sorted(tasks):
            for eeg in sorted(session.rglob(f'{subject.name}_ses-iemu_task-{task}_acq-clinical_run-*_ieeg.eeg')):
                prefix, run = eeg.name.split('_acq-clinical_run-')
                run = run.removesuffix('_ieeg.eeg')
                label_task = task if events_task == 'same' else events_task
                events = eeg.parent / f'{subject.name}_ses-iemu_task-{label_task}_run-{run}_events.tsv'
                records.append({'subject': subject.name, 'task': task, 'run': run,
                                'eeg': eeg, 'header': eeg.with_suffix('.vhdr'),
                                'markers': eeg.with_suffix('.vmrk'), 'events': events})
    return records, skipped


def read_events(path, label_column):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'onset', 'duration', label_column}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing onset, duration or {label_column}')
        events = []
        for row_number, row in enumerate(reader, 2):
            onset, duration = float(row['onset']), float(row['duration'])
            if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
                raise ValueError(f'{path}:{row_number}: invalid event timing')
            if duration == 0:
                continue
            label = row[label_column].strip()
            if not label or label.lower() == 'n/a':
                raise ValueError(f'{path}:{row_number}: missing label')
            events.append((row_number, onset, duration, label, row.get('value', '')))
        return events


def event_sample_bounds(onset, duration, rate):
    """Select only samples with timestamps in [onset, onset + duration).

    Snap floating point products near integer sample boundaries before ceiling
    so decimal TSV timings do not accidentally add or drop a boundary sample.
    """
    def boundary(seconds):
        sample = seconds * rate
        nearest = round(sample)
        if abs(sample - nearest) < 1e-7:
            sample = nearest
        return math.ceil(sample)
    return boundary(onset), boundary(onset + duration)


def split_subjects(subjects, ratios, seed):
    subjects = sorted(set(subjects))
    if len(subjects) < 3:
        raise ValueError('At least three usable participants are needed for disjoint train/validation/test sets.')
    random.Random(seed).shuffle(subjects)
    # Reserve at least one participant per split, then allocate the remainder.
    remaining = len(subjects) - 3
    counts = [1 + int(remaining * r) for r in ratios]
    for i in sorted(range(3), key=lambda i: remaining * ratios[i] % 1, reverse=True)[:len(subjects) - sum(counts)]:
        counts[i] += 1
    assignment, start = {}, 0
    for split, count in zip(('train', 'validation', 'test'), counts):
        assignment.update({subject: split for subject in subjects[start:start + count]})
        start += count
    return assignment


def constant_runs(values, minimum, np):
    """Return long, exactly constant runs as half-open sample intervals."""
    edges = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, len(values)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= minimum]


def clean_recording(raw, record, args, qc_dir):
    """Clean continuous iEEG; fixed rules never use labels or test-set fitting."""
    import numpy as np
    from scipy.signal import welch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    channels_path = record['eeg'].with_name(record['eeg'].name.replace('_ieeg.eeg', '_channels.tsv'))
    sidecar = record['eeg'].with_suffix('.json')
    with channels_path.open(newline='') as stream:
        rows = {r['name']: r for r in csv.DictReader(stream, delimiter='\t')}
    missing = set(raw.ch_names) - set(rows)
    if missing:
        raise ValueError(f'{channels_path}: missing channel metadata for {sorted(missing)}')
    removed = {}
    keep = []
    for name in raw.ch_names:
        row = rows[name]
        if row.get('type', '').upper() not in ('SEEG', 'ECOG', 'DBS'):
            removed[name] = 'not an intracranial EEG channel'
        elif row.get('status', '').lower() == 'bad':
            removed[name] = 'BIDS bad: ' + row.get('status_description', '')
        else:
            keep.append(name)
    if not keep:
        raise ValueError(f'{record["eeg"]}: no good iEEG channels')
    raw.pick(keep).load_data()
    rate = float(raw.info['sfreq'])
    metrics, intervals = [], []
    for i, name in enumerate(raw.ch_names):
        x = raw.get_data(picks=[i])[0]
        if not np.isfinite(x).all():
            removed[name] = 'nonfinite signal'
            continue
        std = float(np.std(x))
        if std <= args.flat_std_uv * 1e-6:
            removed[name] = 'flat/near-flat continuous channel'
            continue
        differences = np.diff(x)
        metrics.append(dict(channel=name, std_uv=std * 1e6,
                            peak_to_peak_uv=float(np.ptp(x)) * 1e6,
                            max_jump_uv=float(np.max(np.abs(differences))) * 1e6))
        # Long exact plateaus are unrecoverable. Isolated sharp spikes are only flagged.
        for a, b in constant_runs(x, max(2, math.ceil(args.plateau_seconds * rate)), np):
            intervals.append(dict(start_sample=a, stop_sample=b, channel=name,
                                  reason='long constant plateau (flat/clipped signal)'))
    drop = [name for name in raw.ch_names if name in removed]
    if drop:
        raw.drop_channels(drop)
    if not raw.ch_names:
        raise ValueError(f'{record["eeg"]}: all channels failed quality checks')
    spread = np.array([m['std_uv'] for m in metrics])
    center = float(np.median(spread))
    for m in metrics:
        m['review_flags'] = []
        if m['std_uv'] > max(center * 10, args.review_amplitude_uv):
            m['review_flags'].append('high variance: inspect; not automatically removed')
        if m['max_jump_uv'] > args.review_amplitude_uv:
            m['review_flags'].append('large transient: inspect; may be neural/epileptiform')
    # Small previews are copied before filtering; the complete recording stays in raw.
    preview_stop = min(raw.n_times, round(10 * rate))
    before = raw.get_data(picks=list(range(min(4, len(raw.ch_names)))), stop=preview_stop)
    psd_stop = min(raw.n_times, round(60 * rate))
    freq, before_psd = welch(raw.get_data(picks=[0], stop=psd_stop)[0], fs=rate,
                             nperseg=min(psd_stop, round(2 * rate)))
    metadata = json.loads(sidecar.read_text())
    line = args.line_frequency if args.line_frequency is not None else metadata.get('PowerLineFrequency')
    if not isinstance(line, (int, float)) or not math.isfinite(line) or line <= 0:
        raise ValueError(f'{sidecar}: missing valid PowerLineFrequency; set --line-frequency')
    upper = min(args.highpass_upper, rate / 2 - 1)
    if upper <= args.low_cutoff:
        raise ValueError('Recording sample rate is too low for requested filter range')
    notches = np.arange(line, upper, line).tolist()
    if notches:
        raw.notch_filter(notches, method='fir', phase='zero', verbose='ERROR')
    raw.filter(args.low_cutoff, upper, method='fir', phase='zero', verbose='ERROR')
    # Retain acquisition reference by default: spatial montage cannot be inferred safely.
    if args.reference == 'average':
        raw._data -= raw._data.mean(axis=0, keepdims=True)
    if args.reference == 'bipolar':
        import re
        import mne
        pairs = []
        names = set(raw.ch_names)
        for name in raw.ch_names:
            match = re.fullmatch(r'(.*?)(\d+)', name)
            if not match:
                continue
            prefix, number = match.groups()
            candidates = [n for n in names if re.fullmatch(re.escape(prefix) + r'\d+', n)
                          and int(re.search(r'\d+$', n).group()) == int(number) + 1]
            for other in candidates:
                group = rows[name].get('group', '')
                if group not in ('', 'n/a') and group == rows[other].get('group'):
                    pairs.append((name, other))
        if not pairs:
            raise ValueError('No adjacent contacts with matching BIDS electrode groups for bipolar reference')
        raw = mne.set_bipolar_reference(raw, anode=[a for a, b in pairs],
                    cathode=[b for a, b in pairs], ch_name=[f'{a}-{b}' for a, b in pairs],
                    drop_refs=True, copy=False, verbose='ERROR')
        raw.pick([f'{a}-{b}' for a, b in pairs])
    after = raw.get_data(picks=list(range(min(4, len(raw.ch_names)))), stop=preview_stop)
    _, after_psd = welch(raw.get_data(picks=[0], stop=psd_stop)[0], fs=rate,
                         nperseg=min(psd_stop, round(2 * rate)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    stride = max(1, int(rate / 200))
    for i, x in enumerate(before):
        axes[0, 0].plot(np.arange(0, len(x), stride) / rate, x[::stride] * 1e6 + i * 300, linewidth=.5)
    for i, x in enumerate(after):
        axes[0, 1].plot(np.arange(0, len(x), stride) / rate, x[::stride] * 1e6 + i * 300, linewidth=.5)
    axes[0, 0].set_title('Before filters: first retained contacts (offset 300 µV)')
    axes[0, 1].set_title('After filters/reference: first output contacts')
    for ax in axes[0]:
        ax.set(xlabel='Seconds from recording start', ylabel='µV + offset')
    axes[1, 0].semilogy(freq, np.maximum(before_psd, 1e-30), label='Before')
    axes[1, 0].semilogy(freq, np.maximum(after_psd, 1e-30), label='After')
    axes[1, 0].set(xlim=(0, min(200, rate / 2)), xlabel='Hz', ylabel='V²/Hz', title='First-contact PSD (first 60 seconds)')
    axes[1, 0].legend()
    axes[1, 1].bar(range(len(metrics)), [m['std_uv'] for m in metrics])
    axes[1, 1].set(xlabel='Retained source channel index', ylabel='Standard deviation (µV)', title='Channel variability before filtering')
    fig.suptitle(record['eeg'].stem, fontsize=9)
    fig.tight_layout()
    fig.savefig(qc_dir / f'{record["eeg"].stem}.png', dpi=120)
    plt.close(fig)
    report = dict(recording=str(record['eeg']), removed_channels=removed,
                  channel_metrics=metrics, rejected_intervals=intervals,
                  line_frequency=line, notches=notches, bandpass=[args.low_cutoff, upper],
                  reference=args.reference, output_channels=raw.ch_names)
    (qc_dir / f'{record["eeg"].stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    return raw, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent / 'ds003688')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'processed_eeg_clean')
    parser.add_argument('--tasks', nargs='+', choices=['film', 'rest'], default=['film'])
    parser.add_argument('--events-task', choices=['same', 'rest', 'film'], default='same')
    parser.add_argument('--label-column', default='trial_type')
    parser.add_argument('--include-markers', action='store_true', help='Include positive-duration start/end task markers as labels')
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=('TRAIN', 'VALIDATION', 'TEST'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='Inspect sources without dependencies or writing output')
    parser.add_argument('--split-summary', type=Path, default=Path(__file__).parent / 'processed_eeg' / 'summary.json', help='Reuse existing subject_splits when available')
    parser.add_argument('--line-frequency', type=float, default=None, help='Override BIDS PowerLineFrequency')
    parser.add_argument('--low-cutoff', type=float, default=0.5)
    parser.add_argument('--high-cutoff', '--highpass-upper', dest='highpass_upper', type=float, default=150, help='Low-pass cutoff in Hz; preserves 70–120 Hz activity')
    parser.add_argument('--flat-std-uv', type=float, default=0.01)
    parser.add_argument('--plateau-seconds', type=float, default=0.5)
    parser.add_argument('--review-amplitude-uv', type=float, default=1000, help='Review flag only; not a rejection threshold')
    parser.add_argument('--reference', choices=['original', 'average', 'bipolar'], default='original')
    args = parser.parse_args(argv)
    if any(not math.isfinite(v) or v <= 0 for v in (args.low_cutoff, args.highpass_upper, args.flat_std_uv, args.plateau_seconds, args.review_amplitude_uv)) or args.low_cutoff >= args.highpass_upper:
        parser.error('Quality thresholds and cutoffs must be positive; low cutoff must be below upper cutoff')
    if args.line_frequency is not None and (not math.isfinite(args.line_frequency) or args.line_frequency <= 0):
        parser.error('--line-frequency must be positive')
    if any(not math.isfinite(r) or r <= 0 for r in args.ratios) or not math.isclose(sum(args.ratios), 1):
        parser.error('--ratios must be positive and sum to 1')
    if not args.dataset.is_dir():
        parser.error(f'Dataset folder does not exist: {args.dataset}')
    if args.events_task != 'same' and any(t != args.events_task for t in args.tasks):
        parser.error('EEG and events must have the same task for aligned labels; use --events-task same.')
    records, skipped = discover(args.dataset, set(args.tasks), args.events_task)
    usable = []
    for record in records:
        record['channels'] = record['eeg'].with_name(record['eeg'].name.replace('_ieeg.eeg', '_channels.tsv'))
        record['sidecar'] = record['eeg'].with_suffix('.json')
        missing = [str(record[k]) for k in ('eeg', 'header', 'markers', 'events', 'channels', 'sidecar') if not record[k].is_file()]
        if missing:
            skipped.append({'subject': record['subject'], 'recording': str(record['eeg']), 'reason': 'missing/unavailable files', 'files': missing})
            continue
        events = read_events(record['events'], args.label_column)
        if not args.include_markers:
            events = [e for e in events if e[3].strip().lower() not in ('start task', 'end task')]
        if not events:
            skipped.append({'recording': str(record['eeg']), 'reason': 'no positive-duration events'})
            continue
        record['event_rows'] = events
        usable.append(record)
    print(f'Found {len(records)} recordings; {len(usable)} locally usable; {len(skipped)} skipped entries.')
    if args.dry_run:
        print(json.dumps({'usable_recordings': [str(r['eeg']) for r in usable], 'skipped': skipped}, indent=2))
        return 0
    if not usable:
        parser.error('No usable recordings. Download BrainVision companions first; use --dry-run to inspect missing files.')
    if args.split_summary.is_file():
        assignment = json.loads(args.split_summary.read_text())['subject_splits']
        if any(assignment.get(r['subject']) not in ('train', 'validation', 'test') for r in usable):
            parser.error('Existing split summary lacks usable subjects; provide a complete --split-summary')
    else:
        assignment = split_subjects([r['subject'] for r in usable], args.ratios, args.seed)
    try:
        import mne
        import numpy as np
        from tqdm import tqdm
        import scipy.signal
        import matplotlib
    except ImportError:
        parser.error('Install dependencies: python -m pip install numpy mne scipy matplotlib tqdm')
    if args.output.exists():
        parser.error(f'Output folder already exists: {args.output}. Choose a new --output folder to avoid mixing old or incomplete samples.')
    args.output.mkdir(parents=True, exist_ok=False)
    qc_dir = args.output / 'quality_control'
    qc_dir.mkdir()
    cleaning_reports = []
    label_counts = {s: {} for s in ('train', 'validation', 'test')}
    (args.output / 'summary.json').write_text(json.dumps({'complete': False, 'subject_splits': assignment}, indent=2))
    adjustments = []
    fields = ['signal_path', 'label', 'event_value', 'subject', 'task', 'run', 'onset', 'duration', 'start_sample', 'stop_sample', 'sampling_rate', 'n_channels', 'n_samples', 'source_eeg', 'source_events', 'event_row', 'original_onset', 'original_duration', 'event_offset', 'clipped']
    counts = {}
    # Filter one continuous recording at a time, before extracting event segments.
    for split in ('train', 'validation', 'test'):
        directory = args.output / split
        directory.mkdir()
        counts[split] = 0
        with (directory / 'manifest.tsv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            selected = [r for r in usable if assignment[r['subject']] == split]
            for record in tqdm(selected, desc=f'Clean {split}', unit='recording'):
                raw = mne.io.read_raw_brainvision(str(record['header']), preload=False, verbose='ERROR')
                try:
                    raw, report = clean_recording(raw, record, args, qc_dir)
                    cleaning_reports.append(report)
                    rate = float(raw.info['sfreq'])
                    for row, onset, duration, label, value in tqdm(record['event_rows'], desc=record['subject'], unit='event', leave=False):
                        tally = label_counts[split].setdefault(label, {'input': 0, 'saved': 0, 'rejected': 0})
                        tally['input'] += 1
                        requested_start, requested_stop = event_sample_bounds(onset, duration, rate)
                        start = max(0, requested_start)
                        stop = min(raw.n_times, requested_stop)
                        if stop <= start:
                            tally['rejected'] += 1
                            skipped.append(dict(recording=str(record['eeg']), event_row=row,
                                                reason='event has no sampled overlap with recording',
                                                original_onset=onset, original_duration=duration))
                            print(f"SKIP {record['events'].name}:{row}: no overlap with recording")
                            continue
                        overlaps = [i for i in report['rejected_intervals'] if i['start_sample'] < stop and i['stop_sample'] > start]
                        if overlaps:
                            tally['rejected'] += 1
                            skipped.append(dict(recording=str(record['eeg']), event_row=row, label=label,
                                                reason='event overlaps unrecoverable plateau', intervals=overlaps))
                            continue
                        clipped = start != requested_start or stop != requested_stop
                        if clipped:
                            adjustment = dict(recording=str(record['eeg']), event_row=row,
                                              original_onset=onset, original_duration=duration,
                                              onset=start / rate, duration=(stop - start) / rate,
                                              removed_seconds=((requested_stop-requested_start)-(stop-start)) / rate)
                            adjustments.append(adjustment)
                            print(f"CLIP {record['events'].name}:{row}: removed {adjustment['removed_seconds']:.6f} seconds")
                        filename = f"{record['eeg'].stem}_event-{row:04d}.npz"
                        metadata = dict(label=label, event_value=value, subject=record['subject'], task=record['task'], run=record['run'], onset=start/rate, duration=(stop-start)/rate, original_onset=onset, original_duration=duration, event_offset=onset + duration, clipped=clipped, start_sample=start, stop_sample=stop, sampling_rate=rate, n_channels=len(raw.ch_names), n_samples=stop-start, source_eeg=str(record['eeg'].absolute()), source_events=str(record['events'].absolute()), event_row=row)
                        np.savez_compressed(directory / filename, signal=raw.get_data(start=start, stop=stop).astype(np.float32), channel_names=np.asarray(raw.ch_names), **metadata)
                        writer.writerow(dict(signal_path=f'{split}/{filename}', **metadata))
                        counts[split] += 1
                        tally['saved'] += 1
                finally:
                    raw.close()
    summary = {'cleaning_reports': cleaning_reports, 'label_retention': label_counts, 'cleaning_settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'complete': True, 'seed': args.seed, 'requested_ratios': args.ratios, 'events_task': args.events_task, 'subject_splits': assignment, 'event_counts': counts, 'include_markers': args.include_markers, 'adjustments': adjustments, 'skipped': skipped, 'interval_convention': '[onset, onset + duration)', 'signal_unit': 'volts', 'signal_axes': ['channels', 'samples']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Corpus written to {args.output.absolute()}: {counts}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
