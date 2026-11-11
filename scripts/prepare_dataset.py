#!/usr/bin/env python3
"""Normalize raw Excel/CSV data to one row per observation (standard CSV)."""
import argparse, re
from pathlib import Path
import pandas as pd

def key(s): return re.sub(r'[^a-z0-9]+','_',str(s).strip().lower()).strip('_')
def find(df,*names):
    lookup={key(c):c for c in df.columns}
    for n in names:
        if key(n) in lookup:return lookup[key(n)]
    return None
def label(v):
    s=str(v).strip().casefold()
    if any(x in s for x in ('decept','lie','yalan','aldat')): return 'Deceptive'
    if any(x in s for x in ('truth','doğru','dogru','honest')): return 'Truthful'
    if s in ('0','1'):
        # Binary Target_Label meaning must be explicit: 1 is treated as deceptive only under this option.
        return None
    return None

def prepare(df,binary_one_deceptive=False):
    idc=find(df,'id','prompt_id'); subj=find(df,'subject_en'); typ=find(df,'label_type');
    sentence=find(df,'sentence','sentence_tr','statement','turkish_sentence','text')
    gt=find(df,'ground_truth','mapped_label','label','target_label'); src=find(df,'source_column')
    records=[]
    # Wide benchmark workbook: one observation per sentence-bearing stimulus column.
    sentence_cols=[c for c in df.columns if key(c).endswith('_tr')]
    if sentence is None and sentence_cols:
        for _,row in df.iterrows():
            for c in sentence_cols:
                text=row[c]
                if pd.isna(text) or not str(text).strip(): continue
                colkey=key(c); inferred='Deceptive' if colkey.startswith(('l_','lie','decept')) else 'Truthful'
                records.append({'id':row[idc] if idc else '', 'sentence':str(text).strip(), 'ground_truth':inferred,
                    'label_type':row[typ] if typ else '', 'source_column':c, 'subject_en':row[subj] if subj else ''})
    else:
        if sentence is None: raise ValueError('No sentence column found')
        for ix,row in df.iterrows():
            text=row[sentence]
            if pd.isna(text) or not str(text).strip(): continue
            raw=row[gt] if gt else None; norm=label(raw) if raw is not None and not pd.isna(raw) else None
            if norm is None and raw is not None and str(raw).strip() in ('0','1') and binary_one_deceptive:
                norm='Deceptive' if str(raw).strip()=='1' else 'Truthful'
            if norm is None: raise ValueError(f"Cannot map label {raw!r} on row {ix}; supply semantic labels or --binary-one-deceptive")
            records.append({'id':row[idc] if idc else ix+1,'sentence':str(text).strip(),'ground_truth':norm,
                'label_type':row[typ] if typ else '', 'source_column':row[src] if src else '', 'subject_en':row[subj] if subj else ''})
    out=pd.DataFrame(records,columns=['id','sentence','ground_truth','label_type','source_column','subject_en'])
    out=out.drop_duplicates().reset_index(drop=True)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('input'); p.add_argument('-o','--output',default='standardized_dataset.csv'); p.add_argument('--sheet',default=0); p.add_argument('--binary-one-deceptive',action='store_true',help='Explicitly interpret numeric 1 as Deceptive and 0 as Truthful')
    a=p.parse_args(); path=Path(a.input)
    df=pd.read_excel(path,sheet_name=a.sheet) if path.suffix.lower() in ('.xlsx','.xls') else pd.read_csv(path,encoding='utf-8-sig')
    out=prepare(df,a.binary_one_deceptive); out.to_csv(a.output,index=False,encoding='utf-8-sig')
    print(f'Wrote {len(out)} standardized rows to {a.output}')
if __name__=='__main__': main()
