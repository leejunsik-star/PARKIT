"""Select a confidence threshold on validation only; never loads test images."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import Cars,collate
from .run import load,evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--root',default='.')
    p.add_argument('--out',default='runs/train/calibration.json');p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4)
    model,ckpt=load(a.weights,a.device)
    loader=DataLoader(Cars(a.root,'val',ckpt['size']),batch_size=4,collate_fn=collate)
    metrics=evaluate(model,loader,a.device,include_records=True);records=metrics.pop('score_matches');rows=[]
    for threshold in [.25,.35,.45,.55,.65,.75,.85,.95]:
        selected=[hit for score,hit in records if score>=threshold]
        tp=sum(selected);fp=len(selected)-tp;fn=metrics['ground_truth']-tp
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        rows.append({'conf':threshold,'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-9),'false_positive':fp,'false_negative':fn})
    best=max(rows,key=lambda r:r['f1'])
    result={'split':'val','selection':'highest F1 on coarse confidence grid, NMS IoU 0.5','recommended_conf':best['conf'],'recommended_metrics':best,'grid':rows,'warning':'Validation selection only; not an independent test result.'}
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
