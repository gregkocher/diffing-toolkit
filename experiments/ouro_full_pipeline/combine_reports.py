"""Combine disjoint native toolkit reports without regrading hypotheses."""
import argparse
import json
from pathlib import Path

EXPECTED = (
    [f'recurrence_{i}' for i in range(4)]
    + [f'adl_{i}' for i in range(4)]
    + [f'jlens_{i}' for i in range(3)]
    + [f'jlens_early_{i}' for i in range(3)]
    + ['recurrence_3_nmf', 'jlens_2_nmf', 'jlens_early_2_nmf']
)


def combine(paths):
    reports = [json.loads(p.read_text()) for p in paths]
    rows = {}
    runs, censored = [], []
    for report in reports:
        for row in report['conditions']:
            name = row['condition']
            assert name in EXPECTED, name
            assert name not in rows, f'Duplicate condition ownership: {name}'
            rows[name] = row
        runs.extend(report['runs'])
        censored.extend(report['censored_runs'])
    identities = [(r['condition'], r['run']) for r in runs + censored]
    assert len(identities) == len(set(identities)), 'Duplicate auditor repetition'
    coverage = {}
    for name in EXPECTED:
        row = rows.get(name, {})
        count = sum(row.get(k, 0) for k in (
            'completed_repetitions', 'budget_exhausted_repetitions',
            'provider_budget_violations'))
        assert count <= 3, (name, count)
        coverage[name] = {'terminal_repetitions': count, 'missing_repetitions': 3-count}
    return {
        'source_reports': [str(p.resolve()) for p in paths],
        'generation_protocol': 'native_single_prompt_v1',
        'expected_conditions': EXPECTED, 'expected_repetitions_per_condition': 3,
        'all_planned_repetitions_terminal': all(v['missing_repetitions'] == 0 for v in coverage.values()),
        'coverage': coverage, 'conditions': [rows[n] for n in EXPECTED if n in rows],
        'runs': runs, 'censored_runs': censored,
        'notes': [
            'Uses saved native SDF grades only; no new evaluator or post-hoc regrading.',
            'Each final is graded three times; reported repetition scores average those grades.',
            'Budget-exhausted and provider-budget-violation runs receive no invented score.',
            'Missing repetitions include unfinished or failed infrastructure, not negative discoveries.',
            'Three auditor repetitions provide descriptive evidence, not a precise success-rate estimate.',
            'NMF has three 100-token topic overviews; frequency conditions have one 100-token overview.',
            'Raw token relevance reports remain in each source report; copied cache entries are not pooled as independent observations.',
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = combine(a.reports)
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output/'COMBINED_NATIVE_RESULTS.json'
    assert not target.exists(), 'Use a new output directory to preserve prior reports'
    target.write_text(json.dumps(result, indent=2)+'\n')
    rows = {r['condition']: r for r in result['conditions']}
    lines = ['# Ouro native full-pipeline results', '',
        'All comparisons use the false-cake organism versus original Ouro.', '',
        '| Condition | Graded finals | Budget exhausted | Provider budget violations | Missing | SDF scores by repetition | Mean |',
        '|---|---:|---:|---:|---:|---|---:|']
    for name in EXPECTED:
        row = rows.get(name, {})
        scores = ', '.join(f'{s:.2f}' for s in row.get('per_repetition_mean_scores', [])) or '—'
        mean = row.get('mean_score')
        average = f'{mean:.2f}' if mean is not None else '—'
        lines.append(f"| {name} | {row.get('completed_repetitions',0)} | {row.get('budget_exhausted_repetitions',0)} | {row.get('provider_budget_violations',0)} | {result['coverage'][name]['missing_repetitions']} | {scores} | {average} |")
    lines += ['', *result['notes']]
    (a.output/'COMBINED_NATIVE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'all_planned_repetitions_terminal': result['all_planned_repetitions_terminal'],
        'graded_finals': len(result['runs']), 'censored': len(result['censored_runs']),
        'coverage': result['coverage']}, indent=2))


if __name__ == '__main__':
    main()
