#!/usr/bin/env python3
"""Evaluate Turkish deception labels from CSV/XLSX or blank-line key:value prompts.

Examples:
  python evaluate_deception.py --input dataset.csv --output evaluation_results.csv
  OPENAI_API_KEY=... python evaluate_deception.py --input dataset.csv --real
  python evaluate_deception.py --prompts prompts.txt --template few_shot --limit 20
"""
import argparse, csv, json, os, re, sys, time, urllib.request
from pathlib import Path

TRUTHFUL = "Truthful"
DECEPTIVE = "Deceptive"
SYSTEM_PROMPT = "Classify the Turkish statement. Reply with exactly Truthful or Deceptive."
ZERO = "Aşağıdaki Türkçe cümlenin doğru mu yoksa aldatıcı mı olduğunu belirleyin. Yalnızca Truthful veya Deceptive yazın.\n\nCümle: {statement}\nKarar:"
FEW = "Aşağıdaki Türkçe cümlenin doğru mu yoksa aldatıcı mı olduğunu belirleyin. Yalnızca Truthful veya Deceptive yazın.\n\nÖrnek: Ali evde. => Truthful\nÖrnek: Ali evde değilken evde olduğunu söyledi. => Deceptive\n\nCümle: {statement}\nKarar:"

def norm_label(value):
    s=str(value or '').strip().casefold()
    if s in {'deceptive','deception','lie','lying','false','fake','1','yes','true_label'} or any(x in s for x in ('decept','lie','yalan','aldat')): return DECEPTIVE
    if s in {'truthful','truth','true','0','no','honest'} or any(x in s for x in ('truth','doğru','dogru','honest')): return TRUTHFUL
    raise ValueError(f"Unrecognized label {value!r}; use Truthful/Deceptive or configure data first")

def norm_prediction(text):
    s=str(text or '').casefold()
    # Parse the earliest explicit class token, not mere substring priority.
    m=re.search(r'\b(deceptive|deception|truthful|truth)\b|yalan|aldatıcı|doğru|dogru',s)
    if not m: raise ValueError(f"Could not parse model response: {text!r}")
    token=m.group(0)
    return DECEPTIVE if token in {'deceptive','deception','yalan','aldatıcı'} else TRUTHFUL

def canon_key(k):
    return re.sub(r'[^a-z0-9]+','_',str(k).strip().lower()).strip('_')
def get(row,*keys,default=''):
    d={canon_key(k):v for k,v in row.items()}
    for k in keys:
        if canon_key(k) in d and d[canon_key(k)] is not None: return d[canon_key(k)]
    return default

def load_records(path=None, prompts=None, sheet=0):
    if prompts:
        records=[]; cur={}
        for line in Path(prompts).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                if cur: records.append(cur); cur={}
            elif ':' in line:
                k,v=line.split(':',1); cur[canon_key(k)]=v.strip()
        if cur: records.append(cur)
        return records
    if not path: raise ValueError('Provide --input or --prompts')
    ext=Path(path).suffix.lower()
    if ext in {'.xlsx','.xls'}:
        try: import pandas as pd
        except ImportError as e: raise RuntimeError('Excel input requires pandas and openpyxl') from e
        df=pd.read_excel(path,sheet_name=sheet)
        return df.where(df.notna(),None).to_dict('records')
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def normalize_record(r):
    sentence=get(r,'sentence','sentence_tr','statement','turkish_sentence','text')
    gt=get(r,'ground_truth','mapped_label','label','target_label')
    if not str(sentence).strip(): raise ValueError(f'Missing sentence in record: {r}')
    if gt in ('',None): raise ValueError('Missing ground_truth/label in record')
    out={str(k):v for k,v in r.items()}
    out.update(sentence=str(sentence), ground_truth=norm_label(gt))
    aliases={'label_type':('label_type',),'source_column':('source_column',),'id':('id','prompt_id')}
    for dest,ks in aliases.items():
        v=get(r,*ks,default='')
        if v not in ('',None): out[dest]=v
    return out

def simulate(record):
    """Deterministic baseline, intentionally simple and clearly not an LLM."""
    text=str(record['sentence']).casefold()
    source=str(record.get('source_column','')).casefold()
    if any(x in source for x in ('l_dir','l_ind','lie','decept')): return DECEPTIVE
    return DECEPTIVE if re.search(r'(yalan|aldatıcı|aldatmak)',text) else TRUTHFUL

def query_api(prompt, model, key, endpoint, retries=3):
    if not key: raise RuntimeError('Real API mode requires --api-key or OPENAI_API_KEY')
    body=json.dumps({'model':model,'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':prompt}], 'temperature':0, 'max_tokens':12},ensure_ascii=False).encode()
    req=urllib.request.Request(endpoint,data=body,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    last=None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req,timeout=60) as res: data=json.loads(res.read().decode())
            return data['choices'][0]['message']['content'].strip()
        except Exception as exc:
            last=exc
            if attempt+1<retries: time.sleep(2**attempt)
    raise RuntimeError(f'API request failed: {last}')

def metrics(records):
    classes=[TRUTHFUL,DECEPTIVE]; n=len(records)
    cm={(a,b):0 for a in classes for b in classes}
    for r in records: cm[(r['ground_truth'],r['prediction'])]+=1
    out={'n':n,'accuracy':sum(cm[(c,c)] for c in classes)/n if n else 0.0,'confusion_matrix':{'actual_'+a:{'pred_'+b:cm[(a,b)] for b in classes} for a in classes}}
    per={}
    for c in classes:
        tp=cm[(c,c)]; fp=sum(cm[(a,c)] for a in classes if a!=c); fn=sum(cm[(c,b)] for b in classes if b!=c)
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
        per[c]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
    out['per_class']=per; out['macro_f1']=sum(v['f1'] for v in per.values())/2
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input'); ap.add_argument('--prompts'); ap.add_argument('--sheet',default=0)
    ap.add_argument('--output',default='evaluation_results.csv'); ap.add_argument('--summary',default='evaluation_summary.json')
    ap.add_argument('--real',action='store_true'); ap.add_argument('--model',default=os.getenv('LLM_MODEL','gpt-4o-mini'))
    ap.add_argument('--api-key',default=os.getenv('OPENAI_API_KEY')); ap.add_argument('--endpoint',default=os.getenv('LLM_BASE_URL','https://api.openai.com/v1/chat/completions'))
    ap.add_argument('--template',choices=['zero_shot','few_shot'],default='zero_shot'); ap.add_argument('--limit',type=int)
    a=ap.parse_args(); records=[normalize_record(r) for r in load_records(a.input,a.prompts,a.sheet)]
    if a.limit is not None: records=records[:max(0,a.limit)]
    template=FEW if a.template=='few_shot' else ZERO
    for i,r in enumerate(records,1):
        prompt=template.format(statement=r['sentence'])
        raw=query_api(prompt,a.model,a.api_key,a.endpoint) if a.real else simulate(r)
        r['raw_response']=raw
        try: r['prediction']=norm_prediction(raw)
        except ValueError:
            if a.real: raise
            r['prediction']=norm_label(raw)
        r['is_correct']=r['ground_truth']==r['prediction']
    fields=list(dict.fromkeys(k for r in records for k in r))
    with open(a.output,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(records)
    summary=metrics(records)
    Path(a.summary).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Wrote {len(records)} records to {a.output}; accuracy={summary['accuracy']:.4f}, macro_f1={summary['macro_f1']:.4f}")
if __name__=='__main__': main()
