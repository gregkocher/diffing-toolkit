"""Check calibration/audit overlap using the actual consecutive token prefixes."""
import argparse, hashlib, json
from pathlib import Path
from transformers import AutoTokenizer
import torch
p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--audit-root',type=Path,default=Path('/workspace/methods_v2/loop1'));a=p.parse_args()
paths=json.loads(Path('/workspace/standard_v1/model_paths.json').read_text())
tok=AutoTokenizer.from_pretrained(paths['base'])
fit_texts=json.loads((a.fit/'fit_prompts.json').read_text())
audit_path=Path('/workspace/standard_v1/corpus.jsonl')
audit=[json.loads(line) for line in audit_path.read_text().splitlines()]
fit_ids=[tuple(tok(t,truncation=True,max_length=64)['input_ids']) for t in fit_texts]
# The toolkit's raw-document loader truncates to n*10 characters before encoding.
audit_ids=[tuple(tok(row['text'][:640],truncation=True,max_length=64)['input_ids']) for row in audit]
saved_paths=list(a.audit_root.glob('**/input_ids/*_input_ids.pt'))
assert len(saved_paths)==1, saved_paths
saved_ids=torch.load(saved_paths[0],map_location='cpu',weights_only=True).tolist()
assert [list(x) for x in audit_ids]==saved_ids, 'Reconstructed audit tokens differ from the actual toolkit tensors'
overlaps=[{'fit_index':i,'audit_indices':[j for j,x in enumerate(audit_ids) if x==ids]} for i,ids in enumerate(fit_ids) if ids in set(audit_ids)]
record={'fit_prompts':len(fit_ids),'audit_documents':len(audit_ids),'token_prefix_length':64,
    'audit_character_prefix_limit':640,'exact_token_prefix_overlaps':overlaps,
    'actual_toolkit_input_ids_sha256':hashlib.sha256(saved_paths[0].read_bytes()).hexdigest(),
    'actual_toolkit_input_ids_match':True,
    'fit_prompt_sha256':hashlib.sha256((a.fit/'fit_prompts.json').read_bytes()).hexdigest(),
    'audit_corpus_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest()}
(a.fit/'disjointness.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
