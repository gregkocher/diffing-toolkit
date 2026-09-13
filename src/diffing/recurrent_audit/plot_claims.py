"""Plot saved claim contrasts as a PDF without running a model."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preview', type=Path)
    a = p.parse_args()
    if a.output.exists() or (a.preview and a.preview.exists()):
        raise FileExistsError('Use new figure paths to preserve prior artifacts')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    payload = a.input.read_bytes()
    data = json.loads(payload)
    cases = [r for r in data['results'] if r['case']['id'].startswith(('temperature_', 'butter_'))]
    if len(cases) != 4:
        raise ValueError('Expected the four separately declared baking probes')
    modes = ['base', 'target'] + [f'disable_loop_{i}' for i in range(1, 5)]
    labels = ['Base', 'Full adapter', 'Off at 1', 'Off at 2', 'Off at 3', 'Off at 4']
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout='constrained')
        for ax, row in zip(axes.flat, cases):
            values = [row['modes'][m]['false_minus_true_sum'] for m in modes]
            ax.bar(range(6), values, color=['#777777', '#b44444'] + ['#3c729f'] * 4)
            ax.axhline(0, color='#333333', lw=.8)
            ax.set_xticks(range(6), labels, rotation=25, ha='right')
            ax.set_title(row['case']['id'].replace('_', ' ').capitalize())
            ax.set_ylabel('log P(false answer) − log P(true answer), nats')
            ax.margins(y=.18)
            for i, value in enumerate(values):
                ax.annotate(f'{value:.2f}', (i, value), xytext=(0, 4 if value >= 0 else -4), textcoords='offset points', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
        fig.suptitle('Recurrence-specific LoRA interventions on four fixed probes\nAll conditions execute four passes; these are likelihood contrasts, not generation rates.', fontsize=12)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.output, metadata={'Title': 'Ouro fixed-probe recurrence interventions', 'Subject': 'Source SHA256 ' + hashlib.sha256(payload).hexdigest()})
        if a.preview:
            fig.savefig(a.preview, dpi=120)
        plt.close(fig)
    print(json.dumps({'pdf': str(a.output), 'source_sha256': hashlib.sha256(payload).hexdigest()}))


if __name__ == '__main__':
    main()
