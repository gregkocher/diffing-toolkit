"""Summarize native saved grades and budgets without adding a new evaluator."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def condition_from_config(config):
    method = config['diffing']['method']
    if method['name'] == 'activation_difference_lens':
        return 'adl_' + str(method.get('recurrence_index') or 0)
    extraction = method['logit_extraction']
    name = extraction['method']
    if name == 'recurrence_logits':
        condition = 'recurrence_' + str(extraction[name]['recurrence_idx'])
    elif name == 'jlens':
        settings = extraction[name]
        early = '/early_lenses/' in settings['local_lens_path']
        condition = ('jlens_early_' if early else 'jlens_') + str(settings['recurrence_idx'])
    else:
        raise ValueError(name)
    if method['agent']['overview']['ordering_type'] == 'nmf':
        condition += '_nmf'
    return condition


def summarize(root, excluded_conditions=()):
    runs = []
    excluded = []
    for stats_path in sorted((root / 'cache').rglob('stats.json')):
        directory = stats_path.parent
        config_path = directory / 'config.json'
        if not config_path.exists():
            continue  # Native pipeline writes resolved config after grading completes.
        config = json.loads(config_path.read_text())
        method = config['diffing']['method']
        if method['agent'].get('generation_protocol_version') != 'native_single_prompt_v1':
            excluded.append({'path': str(directory), 'reason': 'Legacy generation protocol; batched right-padding probes require a clean rerun'})
            continue
        if method['name'] == 'activation_difference_lens' and method['agent'].get('tool_interface_version') != 'dataset_aliases_enabled_tools_v1':
            excluded.append({'path': str(directory), 'reason': 'Legacy ADL tool interface; retained separately, not mixed with corrected audits'})
            continue
        if condition_from_config(config) in excluded_conditions:
            continue
        grades = [json.loads(p.read_text()) for p in sorted(directory.glob('hypothesis_grade_*.json'))]
        if len(grades) != config['diffing']['evaluation']['grader']['num_repeat']:
            continue
        scores = [g['score'] for g in grades]
        runs.append({
            'condition': condition_from_config(config), 'run': directory.name,
            'path': str(directory), 'scores': scores, 'mean_score': statistics.mean(scores),
            'description': (directory / 'description.txt').read_text(),
            'stats': json.loads(stats_path.read_text()),
            'agent_model': config['diffing']['evaluation']['agent']['llm']['model_id'],
            'budgets': config['diffing']['evaluation']['agent']['budgets'],
        })
    censored = []
    for status_path in sorted((root / 'cache').rglob('run_status.json')):
        status = json.loads(status_path.read_text())
        if status['status'] not in {'budget_exhausted', 'provider_budget_violation'}:
            continue
        directory = status_path.parent
        config = json.loads((directory / 'config.json').read_text())
        if config['diffing']['method']['agent'].get('generation_protocol_version') != 'native_single_prompt_v1':
            continue
        condition = condition_from_config(config)
        if condition in excluded_conditions:
            continue
        censored.append({'condition': condition, 'run': directory.name, 'path': str(directory),
            'status': status['status'], 'stats': json.loads((directory / 'stats.json').read_text())})
    groups = defaultdict(list)
    for run in runs:
        groups[run['condition']].append(run)
    conditions = []
    for name in sorted(set(groups) | {r['condition'] for r in censored}):
        members = groups[name]
        means = [r['mean_score'] for r in members]
        conditions.append({'condition': name, 'completed_repetitions': len(members),
            'per_repetition_mean_scores': means, 'mean_score': statistics.mean(means) if means else None,
            'budget_exhausted_repetitions': sum(r['condition'] == name and r['status'] == 'budget_exhausted' for r in censored),
            'provider_budget_violations': sum(r['condition'] == name and r['status'] == 'provider_budget_violation' for r in censored),
            'model_interactions_used': [r['stats']['model_interactions_used'] for r in members]})
    relevance = []
    for p in sorted((root / 'cache').rglob('*_eval.json')):
        data = json.loads(p.read_text())
        if 'percentage' in data and 'labels' in data:
            relevance.append({'path': str(p), 'ordering_type': data.get('ordering_type_id'),
                'ordering_id': data.get('ordering_id'), 'display_label': data.get('display_label'),
                'num_tokens': len(data['labels']), 'fraction_relevant': data['percentage'],
                'weighted_fraction_relevant': data.get('weighted_percentage')})
    totals = {key: sum(r['stats'].get(key, 0) for r in runs) for key in
        ('agent_llm_calls_used', 'agent_prompt_tokens', 'agent_completion_tokens', 'agent_total_tokens', 'model_interactions_used')}
    return {'conditions': conditions, 'runs': runs, 'censored_runs': censored, 'excluded_conditions': list(excluded_conditions), 'excluded_legacy_runs': excluded, 'token_relevance': relevance,
        'completed_agent_usage_only': totals,
        'notes': ['Scores are the existing SDF hypothesis rubric (1–5), not a new grading scheme.',
            'Average grader scores within each auditor repetition, then average auditor repetitions.',
            'Missing/incomplete runs are excluded, never counted as negative results.',
            'NMF and frequency rankings are separate conditions with different overview sizes.',
            'Usage excludes token/hypothesis graders and incomplete auditor runs; it is not total API billing.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-condition', action='append', default=[])
    args = parser.parse_args()
    result = summarize(args.root, args.exclude_condition)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / 'NATIVE_RESULTS.json'
    if out.exists():
        raise FileExistsError('Choose a new output directory; preserve prior reports')
    out.write_text(json.dumps(result, indent=2))
    lines = ['# Native toolkit full-evaluation results', '',
        'Production finals and explicitly censored runs appear below. Smoke results are excluded; missing hypotheses receive no fabricated score.', '',
        '| Condition | Completed finals | Budget exhausted | Provider errors | Scores by repetition | Mean SDF score (1–5) |',
        '|---|---:|---:|---:|---|---:|']
    for row in result['conditions']:
        values = ', '.join(f'{v:.2f}' for v in row['per_repetition_mean_scores'])
        average = '—' if row['mean_score'] is None else f"{row['mean_score']:.2f}"
        lines.append(f"| {row['condition']} | {row['completed_repetitions']} | {row['budget_exhausted_repetitions']} | {row['provider_budget_violations']} | {values} | {average} |")
    lines += ['', *result['notes'], '', 'Exact descriptions, grader scores and artifact paths are in `NATIVE_RESULTS.json`.']
    (args.output / 'NATIVE_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'conditions': result['conditions'], 'usage': result['completed_agent_usage_only']}), flush=True)


if __name__ == '__main__':
    main()
