"""Run a campaign workload in its own process group with a fixed time limit."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--hours",type=float,default=3)
p.add_argument("--output",type=Path,default=Path("/workspace/methods_v2"))
p.add_argument("command",nargs=argparse.REMAINDER)
a=p.parse_args()
assert 0<a.hours<=3.5 and a.command
if a.command[0]=="--":a.command=a.command[1:]
a.output.mkdir(parents=True,exist_ok=True)
assert not (a.output/"WORK_COMPLETE.json").exists()
with (a.output/"workload.log").open("w") as log:
 child=subprocess.Popen(a.command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record={"pid":child.pid,"pgid":child.pid,"proc_start_ticks":Path(f"/proc/{child.pid}/stat").read_text().split()[21],"command":a.command,"started_unix":time.time(),"deadline_unix":time.time()+a.hours*3600}
 (a.output/"job.json").write_text(json.dumps(record,indent=2))
 reason="finished"
 try:code=child.wait(timeout=a.hours*3600)
 except subprocess.TimeoutExpired:
  reason="time_limit"
  os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=30)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 record.update({"exit_code":code,"reason":reason,"finished_unix":time.time()})
 (a.output/"WORK_COMPLETE.json").write_text(json.dumps(record,indent=2))
 print(json.dumps(record),flush=True)
