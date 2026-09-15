"""Render saved native grades and ADL relevance as standalone vector figures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def label(name):
    nmf = name.endswith('_nmf')
    base = name.removesuffix('_nmf')
    index = int(base.rsplit('_', 1)[1]) + 1
    if base.startswith('recurrence_'):
        text = f'Diff mining, pass {index}' + (' (final)' if index == 4 else '')
    elif base.startswith('adl_'):
        text = f'ADL, pass {index}'
    elif base.startswith('jlens_early_'):
        text = f'Early J-lens, pass {index}'
    else:
        text = f'Original J-lens, pass {index}'
    return text + (' — NMF' if nmf else '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10})
    rows = {row['condition']: row for row in data['conditions']}
    names = data['expected_conditions']
    fig, ax = plt.subplots(figsize=(11, 9))
    for y, name in enumerate(names):
        row = rows.get(name, {})
        scores = row.get('per_repetition_mean_scores', [])
        color = '#14826d' if name.startswith('adl_') else '#4666a1'
        offsets = np.linspace(-.16, .16, len(scores)) if len(scores) > 1 else [0]
        for score, offset in zip(scores, offsets):
            ax.scatter(score, y + offset, s=40, color=color, zorder=3)
        exhausted = row.get('budget_exhausted_repetitions', 0)
        violations = row.get('provider_budget_violations', 0)
        missing = data['coverage'][name]['missing_repetitions']
        note = f'{len(scores)} final'
        if exhausted: note += f', {exhausted} exhausted'
        if violations: note += f', {violations} provider violation'
        if missing: note += f', {missing} missing'
        ax.text(5.28, y, note, va='center', fontsize=8)
    ax.set(yticks=range(len(names)), yticklabels=[label(n) for n in names],
           xticks=range(1, 6), xlim=(.75, 7.15), ylim=(len(names)-.5, -.7),
           xlabel='Native SDF hypothesis score (1–5); each point is one auditor repetition',
           title='Ouro objective discovery through the full diffing-toolkit pipeline')
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .015, 'Three grader votes averaged within each final. Exhausted runs have no invented score.\n10 model interactions and 20,000 agent completion tokens per repetition; method evidence layouts differ.', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path = args.output/'native_auditor_scores.pdf'
    assert not path.exists()
    fig.savefig(path)
    plt.close(fig)

    records = [r for r in data['token_relevance']
               if r.get('signal_condition', '').startswith('adl_')
               and r.get('variant') == 'difference' and r.get('target') == 'self']
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.7), sharey=True)
    for recurrence, ax in enumerate(axes):
        values = np.full((3, 5), np.nan)
        for row in records:
            if row['signal_condition'] == f'adl_{recurrence}':
                y = [0, 11, 23].index(row['layer'])
                x = row['position']
                assert np.isnan(values[y, x]), 'Duplicate native ADL judgment'
                values[y, x] = row['fraction_relevant']
        ax.imshow(values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        for y in range(3):
            for x in range(5):
                text = '—' if np.isnan(values[y, x]) else f'{values[y, x]:.0%}'
                ax.text(x, y, text, ha='center', va='center', fontsize=9,
                        color='white' if values[y, x] > .6 else 'black')
        ax.set(title=f'Recurrence pass {recurrence+1}', xticks=range(5),
               yticks=range(3), yticklabels=[0, 11, 23], xlabel='Token position')
    axes[0].set_ylabel('Shared transformer layer')
    fig.suptitle('ADL: native judged relevance among top-20 difference tokens', y=.98)
    fig.text(.02, .02, 'Each cell is one saved list, judged with three token permutations. Repeated tokens across cells are not independent observations.', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .93))
    path = args.output/'adl_token_relevance.pdf'
    assert not path.exists()
    fig.savefig(path)
    plt.close(fig)
    print(json.dumps({'figures': ['native_auditor_scores.pdf', 'adl_token_relevance.pdf']}))


if __name__ == '__main__':
    main()
