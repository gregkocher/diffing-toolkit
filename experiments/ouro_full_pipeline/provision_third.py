"""Provision the explicitly assigned third MATS GPU; credentials stay in files."""
from pathlib import Path
import json,urllib.request
key=Path('/Users/gkocher/Desktop/RESEARCH/MATS_SUMMER_2026/MATS_runpod_API_key.txt').read_text().strip()
base='https://rest.runpod.io/v1/pods'
def request(url,data=None):
 r=urllib.request.Request(url,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'Mozilla/5.0'},data=None if data is None else json.dumps(data).encode())
 try: return json.loads(urllib.request.urlopen(r,timeout=60).read(),strict=False)
 except urllib.error.HTTPError as e:
  print("REST_ERROR",e.code,e.read().decode()[:1500]);raise
pods=request(base)
rows=[{k:p.get(k) for k in ['id','name','desiredStatus','costPerHr','imageName','gpuCount']} for p in pods]
print(json.dumps(rows))
ours=[p for p in pods if p.get('name','').startswith('CLAUDE_POD_GREG---') and p.get('desiredStatus')=='RUNNING']
rate=sum(float(p.get('costPerHr') or 0) for p in ours)
assert rate+3.49<=15,(rate,rows)
name='CLAUDE_POD_GREG---ouro-full-20260915-third'
assert not any(p.get('name')==name for p in pods)
body={'name':name,'imageName':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','gpuTypeIds':['NVIDIA H100 80GB HBM3'],'gpuCount':1,'containerDiskInGb':200,'volumeInGb':0,'volumeMountPath':'/workspace','ports':['22/tcp'],'supportPublicIp':True,'cloudType':'SECURE','interruptible':False,'computeType':'GPU','env':{'PUBLIC_KEY':Path.home().joinpath('.ssh/id_ed25519_runpod_personal.pub').read_text().strip()}}
x=request(base,body)
out=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/full_pipeline_20260915/third_pod_created.json')
out.write_text(json.dumps({k:v for k,v in x.items() if k!='env'},indent=2))
print('CREATED',json.dumps({k:x.get(k) for k in ['id','name','costPerHr','publicIp','portMappings','desiredStatus']}))
