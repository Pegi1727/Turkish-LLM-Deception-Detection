#!/usr/bin/env python3
"""Compute accuracy, confusion matrix, precision, recall, and F1 by strata."""
import argparse, re
import pandas as pd

def k(x):return re.sub(r'[^a-z0-9]+','_',str(x).lower()).strip('_')
def col(df,*names):
    d={k(c):c for c in df.columns}
    for x in names:
        if k(x) in d:return d[k(x)]
    raise ValueError(f'Missing any of columns {names}; found {list(df.columns)}')
def norm(v):
    s=str(v).strip().casefold()
    if s in {'deceptive','deception','lie','false','yalan'} or any(x in s for x in ['decept','yalan','aldat']):return 'Deceptive'
    if s in {'truthful','truth','true','honest'} or any(x in s for x in ['truth','doğru','dogru']):return 'Truthful'
    raise ValueError(f'Unrecognized label {v!r}')
def summarize(g,gt,pred):
    y=g[gt].map(norm); yhat=g[pred].map(norm)
    tp=((y=='Deceptive')&(yhat=='Deceptive')).sum(); fp=((y!='Deceptive')&(yhat=='Deceptive')).sum()
    fn=((y=='Deceptive')&(yhat!='Deceptive')).sum(); tn=((y!='Deceptive')&(yhat!='Deceptive')).sum()
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    return {'n':len(g),'accuracy':(y==yhat).mean(),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('-o','--output',default='analysis_summary.csv');p.add_argument('--confusion-output',default='confusion_matrices.csv');a=p.parse_args()
    df=pd.read_csv(a.input,encoding='utf-8-sig'); gt=col(df,'ground_truth','mapped_label','label');pred=col(df,'prediction','predicted_label')
    strata=[]
    for field,options in [('label_type',('label_type',)),('source_column',('source_column',))]:
        c=next((col(df,x) for x in options if any(k(x)==k(z) for z in df.columns)),None)
        if c:
            for value,g in df.groupby(c,dropna=False):strata.append({'breakdown':field,'condition':value,**summarize(g,gt,pred)})
    rows=[{'breakdown':'overall','condition':'ALL',**summarize(df,gt,pred)}]+strata
    pd.DataFrame(rows).to_csv(a.output,index=False)
    out=[]
    groups=[('overall','ALL',df)]+[(r['breakdown'],r['condition'],df[df[col(df,field)] .fillna('').astype(str)==str(r['condition'])]) for field in ('label_type','source_column') if any(k(field)==k(c) for c in df.columns) for _,r in pd.DataFrame([x for x in strata if x['breakdown']==field]).iterrows()]
    for kind,condition,g in groups:
        y=g[gt].map(norm); predv=g[pred].map(norm)
        for actual in ('Truthful','Deceptive'):
            for prediction in ('Truthful','Deceptive'):
                out.append({'breakdown':kind,'condition':condition,'actual':actual,'prediction':prediction,'count':int(((y==actual)&(predv==prediction)).sum())})
    pd.DataFrame(out).to_csv(a.confusion_output,index=False)
    print(f'Wrote {len(rows)} metric rows and {len(out)} confusion cells')
if __name__=='__main__':main()
