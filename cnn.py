#!/usr/bin/env python3
# All training and preprocessing settings are configured here.
CONFIG = {
    'dataset': 'processed_eeg',
    'output': 'cnn_results_v2',
    'seed': 42,
    'epochs': 30,
    'batch_size': 4,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,
    'patience': 6,
    'min_delta': 0.0001,
    'target_rate': 256,  # Preserves frequencies below 128 Hz after anti-aliasing.
    'filters': [8, 16],
    'kernel_sizes': [65, 17],
    'strides': [4, 4],
    'dilations': [1, 8],  # About 2.25 seconds of temporal context at 256 Hz.
    'amplitude_scale_uv': 100.0,  # Fixed scaling preserves between-event amplitude.
    'gradient_clip': 1.0,
    'run_spectral_baseline': True,
    'baseline_C': 1.0,  # Fixed in advance; test data never tunes this value.
    'baseline_max_iter': 2000,
    'baseline_bands': [(1, 4), (4, 8), (8, 13), (13, 30), (30, 70), (70, 120)],
    'baseline_quantiles': [0.25, 0.5, 0.75],
    'sanity_samples': 16,
    'sanity_epochs': 500,
    'sanity_learning_rate': 0.001,
    'sanity_dropout': 0.0,
    'sanity_weight_decay': 0.0,
    'sanity_target_accuracy': 0.95,
    'synthetic_seconds': 4.0,
    'synthetic_channels': 4,
    'synthetic_frequencies_hz': [10.0, 80.0],
    'synthetic_amplitudes_uv': [25.0, 100.0],
    'synthetic_noise_fraction': 0.05,
    'dropout': 0.3,
    'num_workers': 0,
    'cpu_threads': 4,
    'device': 'auto',  # Prefer CUDA when available; otherwise use CPU.
    'epochs': 300,
    'patience': 30,
    'min_delta': 0.00001,
}

"""Run: python cnn.py

Uses existing train/validation/test folders; one NPZ is one labeled event.
A shared temporal CNN uses temporal mean/std features and learned electrode
attention. Dilated convolutions retain a longer temporal context. Fixed scaling
preserves amplitude, and each electrode's log RMS is supplied explicitly.
Channel layouts may differ; spatial electrode relationships are not modeled.
Reported train/validation curves both use evaluation mode at each epoch end.
The optimization-pass loss (with dropout) is logged separately. Validation
selects the checkpoint; test is evaluated afterward. History/plots update each
epoch. Run `python cnn.py --sanity-check` to fit a tiny TRAIN-only subset with
regularization disabled, using exactly eight examples per class by default.
Synthetic amplitude/frequency controls run first; --synthetic-check runs only
these controls without needing a dataset. Diagnostics save per-example
probabilities each epoch and never read validation/test signals.
A spectral logistic-regression baseline provides a separate comparison.
Requires numpy, scipy, torch, scikit-learn and matplotlib.
"""
from pathlib import Path
from fractions import Fraction
import csv
import json
import random
import argparse

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)


def read_records(root):
    records, subjects = {}, {}
    for split in ('train', 'validation', 'test'):
        records[split], subjects[split] = [], set()
        for path in sorted((root / split).rglob('*.npz')):
            try:
                with np.load(path, allow_pickle=False) as data:
                    label = str(data['label'].item()).strip()
                    subject = str(data['subject'].item()).strip()
                    if not label or not subject:
                        raise ValueError('empty label or subject')
                    records[split].append({'path': path, 'label': label, 'subject': subject})
                    subjects[split].add(subject)
            except Exception as exc:
                raise ValueError(f'Cannot read {path}: {exc}. Regenerate or repair this NPZ before training.') from exc
        if not records[split]:
            raise ValueError(f'No NPZ files in {root / split}')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if subjects[a] & subjects[b]:
            raise ValueError(f'Participant overlap between {a} and {b}')
    classes = sorted({r['label'] for r in records['train']})
    if len(classes) < 2:
        raise ValueError('Training requires at least two classes')
    for split in ('validation', 'test'):
        if {r['label'] for r in records[split]} != set(classes):
            raise ValueError(f'{split} must contain the same classes as training')
    return records, classes


class EEGDataset(Dataset):
    def __init__(self, records, classes, rate, amplitude_scale_uv=100.0):
        self.records, self.rate = records, rate
        self.scale = amplitude_scale_uv * 1e-6
        self.class_index = {label: i for i, label in enumerate(classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record['path'], allow_pickle=False) as data:
            signal = np.asarray(data['signal'], dtype=np.float32)
            rate = float(data['sampling_rate'].item())
        if signal.ndim != 2 or min(signal.shape) < 1 or not np.isfinite(signal).all():
            raise ValueError(f"{record['path']}: invalid channels x time signal")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"{record['path']}: invalid sampling rate")
        signal = signal - signal.mean(axis=1, keepdims=True)
        if not np.isclose(rate, self.rate):
            ratio = Fraction(self.rate / rate).limit_denominator(10000)
            signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=1)
        signal = signal - signal.mean(axis=1, keepdims=True)
        signal = signal / self.scale
        return torch.from_numpy(np.ascontiguousarray(signal)), self.class_index[record['label']]


def collate_events(batch):
    signals, labels = zip(*batch)
    return list(signals), torch.tensor(labels, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, config):
        super().__init__()
        layers, incoming = [], 1
        for outgoing, kernel, stride, dilation in zip(config['filters'], config['kernel_sizes'], config['strides'], config['dilations']):
            layers.extend([nn.Conv1d(incoming, outgoing, kernel, stride=stride,
                                     dilation=dilation, padding=(kernel // 2) * dilation), nn.ReLU()])
            incoming = outgoing
        self.encoder = nn.Sequential(*layers)
        features = 2 * incoming + 1
        self.attention = nn.Linear(features, 1)
        self.classifier = nn.Sequential(nn.Dropout(config['dropout']), nn.Linear(features, n_classes))

    def forward(self, signals):
        embeddings = []
        for signal in signals:
            rms = signal.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            # Normalize waveform shape while retaining amplitude in a separate feature.
            encoded = self.encoder((signal / rms).unsqueeze(1))
            features = torch.cat([encoded.mean(dim=-1), encoded.std(dim=-1, unbiased=False),
                                  rms.log()], dim=1)
            weights = self.attention(features).softmax(dim=0)
            embeddings.append((weights * features).sum(dim=0))
        return self.classifier(torch.stack(embeddings))


def run_epoch(model, loader, device, optimizer=None, gradient_clip=1.0):
    model.train(optimizer is not None)
    total_loss, targets, probabilities = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for signals, labels in loader:
            signals = [signal.to(device) for signal in signals]
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(signals)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss: inspect signal values and optimizer settings')
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            targets.extend(labels.cpu().tolist())
            probabilities.extend(logits.detach().softmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), np.asarray(targets), np.asarray(probabilities)


def save_history(history, output):
    """Persist comparable evaluation-mode curves even before training completes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = output / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for metric, ylabel in [('loss', 'Cross-entropy loss'), ('accuracy', 'Accuracy')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, label in [('train', 'Training (evaluation mode)'), ('validation', 'Validation')]:
            ax.plot([r['epoch'] for r in history], [r[f'{key}_{metric}'] for r in history], label=label)
        ax.set(xlabel='Epoch', ylabel=ylabel, title=f'Training and validation {metric}')
        if metric == 'accuracy':
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f'{metric}.png', dpi=150)
        plt.close(fig)


def balanced_sanity_records(records, classes, config):
    """Select exactly the same number of training events for each class."""
    count = config['sanity_samples']
    if count < len(classes) or count % len(classes):
        raise ValueError('sanity_samples must be a positive multiple of the class count')
    per_class = count // len(classes)
    rng = random.Random(config['seed'])
    selected = []
    for label in classes:
        candidates = sorted((r for r in records if r['label'] == label), key=lambda r: str(r['path']))
        if len(candidates) < per_class:
            raise ValueError(f'Sanity check needs {per_class} training examples of {label}; found {len(candidates)}')
        selected.extend(rng.sample(candidates, per_class))
    rng.shuffle(selected)
    return selected


def fit_diagnostic(cached, examples, classes, config, output, name):
    """Fit known examples; persist ordered probabilities after every epoch."""
    torch.manual_seed(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(cached, batch_size=config['batch_size'], shuffle=True,
                        generator=generator, collate_fn=collate_events)
    evaluation = DataLoader(cached, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_events)
    settings = dict(config, dropout=config['sanity_dropout'], weight_decay=config['sanity_weight_decay'],
                    learning_rate=config['sanity_learning_rate'])
    device = torch.device(config['device'])
    model = SimpleCNN(len(classes), settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    output.mkdir(parents=True, exist_ok=True)
    history = []
    result = {}
    for epoch in range(1, config['sanity_epochs'] + 1):
        run_epoch(model, loader, device, optimizer, config['gradient_clip'])
        loss, y, p = run_epoch(model, evaluation, device)
        prediction = p.argmax(axis=1)
        accuracy = float(accuracy_score(y, prediction))
        history.append(dict(epoch=epoch, loss=loss, accuracy=accuracy))
        # Evaluation loader is unshuffled, so these rows match the input examples.
        predictions = [dict(example=examples[i], true_label=classes[int(y[i])],
                            predicted_label=classes[int(prediction[i])], correct=bool(prediction[i] == y[i]),
                            probabilities={label: float(p[i, j]) for j, label in enumerate(classes)})
                       for i in range(len(y))]
        result = dict(diagnostic=name, training_files=examples, config=settings,
                      classes=classes, class_counts={label: int(sum(y == j)) for j, label in enumerate(classes)},
                      history=history, target_reached=accuracy >= config['sanity_target_accuracy'],
                      predictions=predictions)
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        with (output / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'loss', 'accuracy'])
            writer.writeheader()
            writer.writerows(history)
        with (output / 'predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['example', 'true_label', 'predicted_label', 'correct'] + [f'p_{c}' for c in classes])
            for row in predictions:
                writer.writerow([row['example'], row['true_label'], row['predicted_label'], row['correct'],
                                 *[row['probabilities'][c] for c in classes]])
        print(f'{name} {epoch}: loss={loss:.4f}, accuracy={accuracy:.3f}', flush=True)
        if result['target_reached']:
            break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['epoch'] for r in history], [r['loss'] for r in history])
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title=name)
    axes[1].plot([r['epoch'] for r in history], [r['accuracy'] for r in history])
    axes[1].axhline(config['sanity_target_accuracy'], linestyle='--', color='gray', label='Target')
    axes[1].set(xlabel='Epoch', ylabel='Training accuracy', ylim=(0, 1.05))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)
    print(f'{name}: target {"reached" if result["target_reached"] else "not reached"}; saved to {output}', flush=True)
    return result


def synthetic_diagnostics(config, output):
    """Separate amplitude and frequency controls; neither uses real EEG labels."""
    rate = config['target_rate']
    frequencies = config['synthetic_frequencies_hz']
    amplitudes = config['synthetic_amplitudes_uv']
    if len(frequencies) != 2 or len(amplitudes) != 2 or max(frequencies) >= rate / 2:
        raise ValueError('Synthetic controls need two frequencies below Nyquist and two amplitudes')
    count = config['sanity_samples']
    if count < 2 or count % 2:
        raise ValueError('Synthetic controls require an even sanity_samples >= 2')
    rng = np.random.default_rng(config['seed'])
    time = np.arange(round(rate * config['synthetic_seconds'])) / rate
    results = {}
    for kind in ('amplitude', 'frequency'):
        cached, examples = [], []
        for i in range(count):
            label = i % 2
            frequency = frequencies[label] if kind == 'frequency' else frequencies[0]
            amplitude = amplitudes[label] if kind == 'amplitude' else amplitudes[0]
            phase = rng.uniform(0, 2 * np.pi, (config['synthetic_channels'], 1))
            waveform = np.sin(2 * np.pi * frequency * time[None, :] + phase)
            waveform += rng.normal(0, config['synthetic_noise_fraction'], waveform.shape)
            waveform -= waveform.mean(axis=1, keepdims=True)
            # Equalize RMS before applying amplitude, preventing the frequency
            # control from being solved merely by different overall amplitudes.
            waveform /= np.maximum(np.sqrt(np.mean(waveform ** 2, axis=1, keepdims=True)), 1e-12)
            signal = waveform * amplitude / config['amplitude_scale_uv']
            cached.append((torch.from_numpy(signal.astype(np.float32)), label))
            examples.append(f'synthetic_{kind}_{i:03d}')
        labels = ['low_amplitude', 'high_amplitude'] if kind == 'amplitude' else ['low_frequency', 'high_frequency']
        results[kind] = fit_diagnostic(cached, examples, labels, config, output / kind, f'Synthetic {kind}')['target_reached']
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def sanity_check(records, classes, config, output):
    """Run synthetic controls and a balanced TRAIN-only overfit diagnostic."""
    selected = balanced_sanity_records(records, classes, config)
    controls = synthetic_diagnostics(config, output / 'synthetic')
    dataset = EEGDataset(selected, classes, config['target_rate'], config['amplitude_scale_uv'])
    # Cache the tiny subset so diagnostics do not repeatedly decompress EEG files.
    cached = [dataset[i] for i in range(len(dataset))]
    result = fit_diagnostic(cached, [str(r['path']) for r in selected], classes, config, output, 'Balanced real EEG')
    print(f'Synthetic controls: {controls}; balanced real EEG target reached: {result["target_reached"]}')
    if not all(controls.values()):
        print('Synthetic control failed: investigate optimization/model implementation before interpreting real EEG.')
    elif not result['target_reached']:
        print('Synthetic controls passed but real EEG did not: inspect representation, alignment and subset examples.')


def spectral_baseline(records, classes, config, output):
    """Fixed spectral/logistic baseline; scaling fits TRAIN only."""
    from scipy.signal import welch
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    features, labels = {}, {}
    for split, rows in records.items():
        dataset = EEGDataset(rows, classes, config['target_rate'], config['amplitude_scale_uv'])
        features[split], labels[split] = [], []
        for i in range(len(dataset)):
            signal, label = dataset[i]
            freq, psd = welch(signal.numpy(), fs=config['target_rate'],
                              nperseg=min(signal.shape[1], round(2 * config['target_rate'])), axis=-1)
            bands = config['baseline_bands']
            if config['target_rate'] <= 240:
                raise ValueError('Spectral baseline requires target_rate > 240 Hz for 70–120 Hz features')
            if len(freq) < 2:
                raise ValueError('Segment too short for spectral baseline')
            power = np.stack([psd[:, (freq >= lo) & (freq < hi)].sum(axis=-1) * (freq[1] - freq[0])
                              for lo, hi in bands], axis=-1)
            power = np.log10(np.maximum(power, 1e-20))
            features[split].append(np.quantile(power, config['baseline_quantiles'], axis=0).ravel())
            labels[split].append(label)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=config['baseline_C'], max_iter=config['baseline_max_iter'],
                                                              random_state=config['seed']))
    model.fit(features['train'], labels['train'])
    results = {}
    for split in ('train', 'validation', 'test'):
        probabilities = model.predict_proba(features[split])
        prediction = probabilities.argmax(axis=1)
        results[split] = dict(loss=float(log_loss(labels[split], probabilities, labels=list(range(len(classes))))),
                             accuracy=float(accuracy_score(labels[split], prediction)),
                             balanced_accuracy=float(balanced_accuracy_score(labels[split], prediction)),
                             macro_f1=float(f1_score(labels[split], prediction, average='macro')))
    (output / 'spectral_baseline.json').write_text(json.dumps(dict(classes=classes, metrics=results), indent=2) + '\n')
    print('Spectral baseline:', results, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train CNN or run a training-only sanity check')
    parser.add_argument('--sanity-check', action='store_true', help='Run synthetic controls and balanced real-EEG diagnostics')
    parser.add_argument('--synthetic-check', action='store_true', help='Run synthetic controls only; no dataset required')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'],
                        default=CONFIG['device'], help='Compute device (auto prefers CUDA, then CPU)')
    args = parser.parse_args()
    config = CONFIG.copy()
    config['device'] = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use a CUDA-enabled PyTorch installation on a machine with an NVIDIA GPU, or select --device cpu.')
    print(f'Using device: {config["device"]}', flush=True)
    if config['device'] == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}', flush=True)
    if (config['epochs'] < 1 or config['batch_size'] < 1 or config['patience'] < 1
            or config['target_rate'] <= 0 or config['learning_rate'] <= 0):
        raise ValueError('Epochs, batch size, patience, sample rate and learning rate must be positive')
    if not (len(config['filters']) == len(config['kernel_sizes']) == len(config['strides']) == len(config['dilations']) > 0):
        raise ValueError('Specify one kernel size and stride per convolution filter count')
    if any(config[key] <= 0 for key in ('amplitude_scale_uv', 'gradient_clip', 'sanity_samples', 'sanity_epochs', 'sanity_learning_rate', 'baseline_C')):
        raise ValueError('Amplitude scale, gradient clip, baseline C and sanity settings must be positive')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    base = Path(__file__).resolve().parent
    if args.synthetic_check:
        synthetic_diagnostics(config, base / config['output'] / 'sanity_check' / 'synthetic')
        return
    records, classes = read_records(base / config['dataset'])
    if args.sanity_check:
        sanity_check(records['train'], classes, config, base / config['output'] / 'sanity_check')
        return
    output = base / config['output']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    generator = torch.Generator().manual_seed(config['seed'])
    loaders = {
        split: DataLoader(EEGDataset(rows, classes, config['target_rate'], config['amplitude_scale_uv']),
                          batch_size=config['batch_size'], shuffle=split == 'train',
                          num_workers=config['num_workers'], collate_fn=collate_events,
                          generator=generator if split == 'train' else None)
        for split, rows in records.items()
    }
    loaders['train_eval'] = DataLoader(loaders['train'].dataset, batch_size=config['batch_size'],
                                      shuffle=False, num_workers=config['num_workers'], collate_fn=collate_events)
    device = torch.device(config['device'])
    model = SimpleCNN(len(classes), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    print(f'Classes: {classes}; parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    for split, rows in records.items():
        print(f'{split}: {len(rows)} events, {len({r["subject"] for r in rows})} participants', flush=True)
    history, best_loss, stale, best_epoch = [], float('inf'), 0, 0
    checkpoint = output / 'best_model.pt'
    for epoch in range(1, config['epochs'] + 1):
        optimization_loss, _, _ = run_epoch(model, loaders['train'], device, optimizer, config['gradient_clip'])
        train_loss, train_y, train_p = run_epoch(model, loaders['train_eval'], device)
        val_loss, val_y, val_p = run_epoch(model, loaders['validation'], device)
        row = dict(epoch=epoch, optimization_loss=optimization_loss, train_loss=train_loss, validation_loss=val_loss,
                   train_accuracy=float(accuracy_score(train_y, train_p.argmax(axis=1))),
                   validation_accuracy=float(accuracy_score(val_y, val_p.argmax(axis=1))))
        history.append(row)
        save_history(history, output)
        print(f'Epoch {epoch:02d}: optimization loss={optimization_loss:.4f}, train eval loss={train_loss:.4f}, validation loss={val_loss:.4f}, '
              f'validation accuracy={row["validation_accuracy"]:.3f}', flush=True)
        if val_loss < best_loss - config['min_delta']:
            best_loss, stale, best_epoch = val_loss, 0, epoch
            torch.save(dict(model_state_dict=model.state_dict(), config=config,
                            classes=classes, epoch=epoch), checkpoint)
        else:
            stale += 1
        if stale >= config['patience']:
            print('Early stopping.', flush=True)
            break
    if config['run_spectral_baseline']:
        spectral_baseline(records, classes, config, output)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    test_loss, truth, probabilities = run_epoch(model, loaders['test'], device)
    predictions = probabilities.argmax(axis=1)
    metrics = dict(best_epoch=best_epoch, classes=classes, test_loss=test_loss,
                   accuracy=float(accuracy_score(truth, predictions)),
                   balanced_accuracy=float(balanced_accuracy_score(truth, predictions)),
                   macro_f1=float(f1_score(truth, predictions, average='macro')),
                   confusion_matrix=confusion_matrix(truth, predictions).tolist(),
                   classification_report=classification_report(truth, predictions,
                       labels=list(range(len(classes))), target_names=classes,
                       output_dict=True, zero_division=0))
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output / 'test_predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'subject', 'true_label', 'predicted_label'] + [f'p_{c}' for c in classes])
        for record, prediction, probs in zip(records['test'], predictions, probabilities):
            writer.writerow([str(record['path']), record['subject'], record['label'], classes[prediction], *probs])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    plot_dir = output / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r['epoch'] for r in history]
    for metric, ylabel in [('loss', 'Cross-entropy loss'), ('accuracy', 'Accuracy')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [r[f'train_{metric}'] for r in history], label='Training')
        ax.plot(epochs, [r[f'validation_{metric}'] for r in history], label='Validation')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=f'Training and validation {metric}')
        if metric == 'accuracy':
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f'{metric}.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Training')
    axes[0].plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Learning curves')
    axes[0].legend()
    axes[1].plot(epochs, [r['train_accuracy'] for r in history], label='Training')
    axes[1].plot(epochs, [r['validation_accuracy'] for r in history], label='Validation')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy curves', ylim=(0, 1))
    axes[1].legend()
    ConfusionMatrixDisplay(np.asarray(metrics['confusion_matrix']), display_labels=classes).plot(ax=axes[2], colorbar=False)
    axes[2].set_title('Test confusion matrix')
    fig.tight_layout()
    fig.savefig(output / 'training_results.png', dpi=150)
    fig.savefig(plot_dir / 'training_results.png', dpi=150)
    plt.close(fig)
    print(f'Test accuracy: {metrics["accuracy"]:.3f}; balanced accuracy: '
          f'{metrics["balanced_accuracy"]:.3f}; macro F1: {metrics["macro_f1"]:.3f}')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
