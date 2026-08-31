#!/usr/bin/env python3
"""Generate controlled KEREN training candidates from approved seed patterns.

This does NOT label generated records as gold. Generated records go to candidates/
and require validation/review before promotion.
"""
from __future__ import annotations
import argparse, json, random
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'datasets/gold/gold_v0.1_seed.jsonl'
OUT=ROOT/'datasets/candidates/generated_v0.1.jsonl'
PARAPHRASES={
 'gold_0001':['chrome kholo','Keren Chrome open kro','chrome start kar do','PC me chrome khol do','kren krom kholo'],
 'gold_0004':['PC pe VS Code kholo','computer me vscode open kro','laptop par VS Code start kar do','PC wala vscode khol do'],
 'gold_0005':['aaj mausam kaisa hai','current weather btao','aaj ka weather check kro','weather abhi kya hai']
}

def read(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',type=Path,default=OUT); args=p.parse_args(); random.seed(args.seed)
    seeds={r['id']:r for r in read(SEED)}; rows=[]; n=1
    for source_id,texts in PARAPHRASES.items():
        if source_id not in seeds: continue
        for text in texts:
            r=deepcopy(seeds[source_id]); r['id']=f'candidate_{n:04d}'; r['user_input']['text']=text; r['user_input']['normalized_text']=None
            r['metadata']['source']='deterministic_generator'; r['metadata']['approved']=False; r['metadata']['synthetic']=True; r['metadata']['training_wheels_candidate']=True; r['metadata']['dataset_split']=None
            r['metadata']['source_trace_id']=source_id; rows.append(r); n+=1
    random.shuffle(rows); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    print(f'Generated {len(rows)} review-required candidates -> {args.output}')

if __name__=='__main__': main()
