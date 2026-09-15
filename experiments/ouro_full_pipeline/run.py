"""Run standard toolkit full evaluation from preserved Ouro caches.

No custom judging or agent loop: every condition invokes main.py. Launch this
script under experiments/ouro_campaign/run_bounded.py for a campaign deadline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_once(archive, target, expected):
    actual = digest(archive)
    if actual != expected:
        raise ValueError(f'Archive checksum mismatch: {archive}')
    receipt = target / 'EXTRACTED.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['sha256'] == expected
        return
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # Only cache artifacts; never restore old checkout/config/key files.
        selected = [m for m in tar.getmembers() if m.name.startswith(('methods_v2/', 'fit128/'))]
        for member in selected:
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unexpected archive member: {member.name}')
            resolved = (target / member.name).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ValueError(f'Unsafe archive path: {member.name}')
        tar.extractall(target, members=selected)
    receipt.write_text(json.dumps({'archive': str(archive), 'sha256': actual}, indent=2))


def stage(root):
    extract_once(root / 'methods_export.tar', root / 'imports/adl',
        '383cd076078e73cea94fef2aae44f2fc69537c64a9bcfeca7b2408cde59ecb06')
    extract_once(root / 'ouro_full_eval_inputs.tar.gz', root / 'imports/jlens',
        '3954985524516b55c32eb587d6067f055e17bf706b9c7e1bbe8fdd8b0e5ec958')
    for r in range(4):
        src = Path(f'/workspace/methods_v2/recurrence_{r}/results/diffing_results')
        dst = root / f'cache/recurrence_{r}/diffing_results'
        if not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('saved_tensors', 'agent'))
    for r in range(3):
        src = root / f'imports/jlens/methods_v2/extension/loop{r+1}/results/diffing_results'
        dst = root / f'cache/jlens_{r}/diffing_results'
        if not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('saved_tensors', 'agent'))
    src = root / 'imports/adl/methods_v2/adl_v2/results/diffing_results'
    dst = root / 'cache/adl/diffing_results'
    if not dst.exists():
        shutil.copytree(src, dst)


def command_for(repo, root, condition, smoke=False):
    ordering = 'nmf' if condition.endswith('_nmf') else 'top_k_occurring'
    cache_condition = condition.removesuffix('_nmf')
    method, recurrence = cache_condition.rsplit('_', 1)
    recurrence = int(recurrence)
    is_adl = method == 'adl'
    experiment = 'ouro_adl_full' if is_adl else 'ouro_diff_mining_full'
    storage = root / ('cache/adl' if is_adl else f'cache/{cache_condition}')
    if smoke:
        target = root / f'smokes/{condition}/results'
        if not target.exists():
            shutil.copytree(storage, target)
        storage = target
    cmd = [sys.executable, str(repo / 'main.py'), 'model=ouro_1_4B',
        'organism=ouro_cake_eos1221', 'infrastructure=runpod',
        'diffing/method=' + ('activation_difference_lens' if is_adl else 'diff_mining'),
        f'+experiment=[{experiment},ouro_openai]', f'infrastructure.storage.base_dir={storage}',
        f'diffing.evaluation.agent.num_repeat={1 if smoke else 3}',
        f'diffing.evaluation.agent.budgets.model_interactions=[{2 if smoke else 10}]',
        f'diffing.evaluation.agent.budgets.agent_llm_calls={8 if smoke else 30}',
        f'diffing.evaluation.agent.budgets.token_budget_generated={15000 if smoke else 20000}',
        f'diffing.evaluation.grader.num_repeat={1 if smoke else 3}',
        f'diffing.method.token_relevance.grader.permutations={1 if smoke else 3}',
        f'diffing.method.token_relevance.k_candidate_tokens={20 if is_adl or smoke else 100}']
    if not is_adl:
        cmd.append(f'diffing.method.agent.overview.ordering_type={ordering}')
    if is_adl:
        if ordering != 'top_k_occurring':
            raise ValueError('NMF overview applies only to diff mining')
        # The original first-pass cache predates the explicit recurrence selector.
        cmd.append(f'diffing.method.recurrence_index={"null" if recurrence == 0 else recurrence}')
        if smoke:
            cmd.extend([
                'diffing.method.token_relevance.tasks=[{dataset:/workspace/standard_v1/corpus.jsonl,layer:1.0,positions:[1],source:logitlens}]',
                'diffing.method.token_relevance.grade_base=false',
                'diffing.method.token_relevance.grade_ft=false',
            ])
    elif method == 'recurrence':
        cmd.extend(['diffing.method.logit_extraction.method=recurrence_logits',
            f'diffing.method.logit_extraction.recurrence_logits.recurrence_idx={recurrence}'])
    elif method in {'jlens', 'jlens_early'}:
        lens_path = (root / f'early_lenses/loop{recurrence+1}_n128.pt' if method == 'jlens_early' else root / f'imports/jlens/fit128/loop{recurrence+1}_n128.pt')
        cmd.extend(['diffing.method.logit_extraction.method=jlens',
            'diffing.method.logit_extraction.jlens.layer=1.0',
            'diffing.method.logit_extraction.jlens.lens_source=base',
            f'diffing.method.logit_extraction.jlens.recurrence_idx={recurrence}',
            f'diffing.method.logit_extraction.jlens.local_lens_path={lens_path}'])
    else:
        raise ValueError(condition)
    return cmd


def prepare_early_lenses(root):
    from huggingface_hub import hf_hub_download
    repo_id = 'wasd12345/ouro-jacobian-lenses-20260915-early-20260915T112740Z'
    revision = 'fc737e53a739534f32c26d1daaa189af90643333'
    output = root / 'early_lenses'
    output.mkdir(exist_ok=True)
    files = []
    token = Path('/root/.hf_token').read_text().strip()
    for index in range(1, 4):
        filename = f'loop{index}_n128.pt'
        source = Path(hf_hub_download(repo_id, filename, revision=revision, token=token))
        target = output / filename
        if target.exists():
            assert digest(target) == digest(source)
        else:
            shutil.copy2(source, target)
        files.append({'filename': filename, 'sha256': digest(target), 'bytes': target.stat().st_size})
    receipt = {'repo_id': repo_id, 'revision': revision, 'files': files,
        'calibration_positions': list(range(47)), 'audit_positions': list(range(64)),
        'reference_model': 'original Ouro; target organism is unchanged'}
    (output / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/full_audit_20260915'))
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--skip-stage', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--conditions', nargs='+', default=['recurrence_3', 'adl_3', 'jlens_2'])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if not args.skip_stage:
        stage(args.root)
    if args.stage_only:
        return
    if any(c.startswith('jlens_early_') for c in args.conditions):
        prepare_early_lenses(args.root)
    paths = json.loads(Path('/workspace/standard_v1/model_paths.json').read_text())
    link = repo / 'ouro_target_adapter'
    if not link.exists():
        link.symlink_to(paths['adapter'], target_is_directory=True)
    assert link.resolve() == Path(paths['adapter']).resolve()
    env = os.environ.copy()
    env.update(OURO_BASE_PATH=paths['base'], PYTHONPATH=str(repo / 'src'),
        HF_HOME='/workspace/hf', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    run_dir = args.root / ('smoke_runs' if args.smoke else 'full_runs')
    run_dir.mkdir(exist_ok=True)
    for condition in args.conditions:
        condition_dir = run_dir / condition
        condition_dir.mkdir(exist_ok=True)
        if (condition_dir / 'COMPLETE.json').exists():
            raise FileExistsError(f'Refusing to overwrite completed run: {condition_dir}')
        output = condition_dir / f'attempt_{time.time_ns()}'
        output.mkdir()
        cmd = command_for(repo, args.root, condition, args.smoke)
        record = {'condition': condition, 'smoke': args.smoke, 'command': cmd,
            'started_unix': time.time(), 'source_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'launcher_sha256': digest(__file__), 'model_paths': paths,
            'algorithm': 'standard main.py pipeline.mode=full',
            'auditor_ordering': 'ADL standard overview' if condition.startswith('adl') else ('nmf' if condition.endswith('_nmf') else 'top_k_occurring')}
        (output / 'STARTED.json').write_text(json.dumps(record, indent=2))
        with (output / 'main.log').open('w') as log:
            result = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, finished_unix=time.time())
        result_name = 'COMPLETE.json' if result.returncode == 0 else 'FAILED.json'
        (output / result_name).write_text(json.dumps(record, indent=2))
        if result.returncode == 0:
            (condition_dir / 'COMPLETE.json').write_text(json.dumps(record, indent=2))
        print(json.dumps({'condition': condition, 'exit_code': result.returncode}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
