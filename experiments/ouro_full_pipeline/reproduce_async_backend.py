import sys,json
import anyio
import torch
from nnsight import NNsight
warm = '--warm' in sys.argv
print(json.dumps({'warm':warm,'object_save_before':hasattr(object,'save')}),flush=True)
if warm:anyio.run(anyio.sleep,0)
model=NNsight(torch.nn.Linear(2,2))
with model.trace(torch.ones(1,2)):
    result=model.output.save()
try:
    anyio.run(anyio.sleep,0)
    print(json.dumps({'status':'success','object_save_after':hasattr(object,'save')}),flush=True)
except Exception as exc:
    print(json.dumps({'status':'error','type':type(exc).__name__,'message':str(exc)}),flush=True)
    raise
