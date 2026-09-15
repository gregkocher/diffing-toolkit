import json,os,signal,time
from pathlib import Path
root=Path('/workspace/full_audit_20260915');records=[]
for pid,script in [(2551,'ouro_migrate.py'),(3354,'launch_on_smokes.py')]:
    proc=Path('/proc')/str(pid)
    if not proc.exists():
        records.append({'pid':pid,'script':script,'state':'already_exited'});continue
    expected=str(root/script)
    command=(proc/'cmdline').read_bytes().split(b'\0')
    if not any(value.decode()==expected for value in command):
        raise RuntimeError(f'Unexpected owner for PID {pid}; refusing signal')
    start=(proc/'stat').read_text().split()[21]
    assert (proc/'stat').read_text().split()[21]==start
    os.kill(pid,signal.SIGTERM)
    records.append({'pid':pid,'script':script,'proc_start_ticks':start,'state':'terminated_owned_watcher'})
(root/'WATCHERS_QUIESCED.json').write_text(json.dumps({'time':time.time(),'watchers':records},indent=2));print(json.dumps(records))
