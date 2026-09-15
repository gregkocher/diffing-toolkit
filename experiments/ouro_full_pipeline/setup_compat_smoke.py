import json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/workspace/full_audit_20260915');compat=root/'compat_v3';repo=Path('/workspace/diffing-toolkit-full-20260915-v3');python='/workspace/toolkit-env/bin/python'
compat.mkdir(exist_ok=True)
if not (compat/'cache').exists():(compat/'cache').symlink_to(root/'cache',target_is_directory=True)
if not (compat/'imports').exists():(compat/'imports').symlink_to(root/'imports',target_is_directory=True)
def copy_file(src,dst):
    if Path(src).suffix=='.pt' and Path(src).stat().st_size>1_000_000:
        os.link(src,dst);return dst
    return shutil.copy2(src,dst)
# Preserve cached token grades, but force fresh agent/native queries in newroot.
src=root/'smokes/adl_3/results';dst=compat/'smokes/adl_3/results'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('agent'),copy_function=copy_file)
out=root/'workers/smoke_adl_jlens_v3';out.mkdir(parents=True)
cmd=[python,str(repo/'experiments/ouro_campaign/run_bounded.py'),'--hours','1','--output',str(out/'bounded'),'--',python,str(repo/'experiments/ouro_full_pipeline/run.py'),'--root',str(compat),'--skip-stage','--smoke','--conditions','adl_3','jlens_2']
with (out/'runner.log').open('w') as log:p=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'LAUNCH.json').write_text(json.dumps({'runner_pid':p.pid,'command':cmd,'started_unix':time.time()},indent=2));print(p.pid,flush=True)
