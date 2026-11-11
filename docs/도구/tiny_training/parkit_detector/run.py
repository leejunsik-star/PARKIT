"""Run from docs/도구/tiny_training: python -m parkit_detector.run --help"""
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import ParkitTiny,detection_loss,predict,iou
from .data import Cars,collate,draw,letterbox,tensor

@torch.no_grad()
def evaluate(model,loader,device,conf=.25,output=None,include_records=False):
    if str(device).startswith(('mps', 'cuda')):
        # Python 기반 NMS/정답 매칭은 작은 GPU 연산/동기화를 반복한다.
        # 학습 모델/optimizer는 GPU에 유지하고 검증용 복사본만 CPU에서 실행.
        import copy
        model=copy.deepcopy(model).cpu()
        device='cpu'
    model.eval();records=[];total=0;tp_fixed=fp_fixed=0;missed=[]
    for x,targets,names in loader:
        results=predict(model,x.to(device),conf=.001)
        for image,gt,(boxes,scores),name in zip(x,targets,results,names):
            gt=gt.to(device);total+=len(gt);matched=set();flags=[]
            for b,s in zip(boxes,scores):
                overlaps=iou(b,gt) if len(gt) else torch.empty(0,device=device)
                choices=overlaps.argsort(descending=True).tolist()
                match=next((j for j in choices if j not in matched and overlaps[j]>=.5),None)
                hit=match is not None
                if hit:matched.add(match)
                records.append((float(s),int(hit)));flags.append(hit)
                if s>=conf:
                    tp_fixed+=int(hit);fp_fixed+=int(not hit)
            hits=sum(int(hit) for hit,s in zip(flags,scores) if s>=conf)
            missed.append({'image':name,'false_negative':len(gt)-hits,'false_positive':sum(int(not hit) for hit,s in zip(flags,scores) if s>=conf)})
            if output:
                mask=scores>=conf;draw(image,boxes[mask].cpu(),output/name,scores[mask].cpu())
    records.sort(reverse=True);ap=0.
    if total and records:
        hits=np.array([r[1] for r in records]);tp=np.cumsum(hits);fp=np.cumsum(1-hits)
        recall=tp/total;precision=tp/(tp+fp)
        ap=float(np.mean([precision[recall>=r].max() if np.any(recall>=r) else 0 for r in np.linspace(0,1,101)]))
    result={'precision':tp_fixed/max(tp_fixed+fp_fixed,1),'recall':tp_fixed/max(total,1),'mAP50':ap,'true_positive':tp_fixed,'false_positive':fp_fixed,'false_negative':total-tp_fixed,'ground_truth':total,'confidence':conf,'per_image':missed}
    if include_records:result['score_matches']=records
    return result

def save_checkpoint(ckpt,path):
    # Use Python's file API and replace only after a complete write. A failed
    # save preserves the previous usable checkpoint instead of truncating it.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as handle:
        torch.save(ckpt,handle)
    temp.replace(path)

def load(path,device):
    ckpt=torch.load(path,map_location=device,weights_only=True)
    model=ParkitTiny(ckpt['width']).to(device);model.load_state_dict(ckpt['model']);model.eval();return model,ckpt

class Detector:
    """Adapter accepts BGR numpy webcam frames and returns original pixel boxes."""
    def __init__(self,weights,device='cpu',conf=.25):
        self.device=device;self.model,self.ckpt=load(weights,device);self.size=self.ckpt['size'];self.conf=conf
    @torch.no_grad()
    def detect(self,frame):
        from PIL import Image
        h,w=frame.shape[:2];im=Image.fromarray(np.ascontiguousarray(frame[...,::-1]))
        im,scale,pad=letterbox(im,self.size)
        boxes,scores=predict(self.model,tensor(im)[None].to(self.device),self.conf)[0]
        boxes=(boxes.cpu()-torch.tensor([*pad,*pad]))/torch.tensor([*scale,*scale])
        boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,w);boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,h)
        return [{'box':b.tolist(),'conf':float(s)} for b,s in zip(boxes,scores.cpu()) if b[2]>b[0] and b[3]>b[1]]

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','preview','train','overfit','evaluate','benchmark'])
    p.add_argument('--root',default='.');p.add_argument('--out',default='runs/parkit');p.add_argument('--weights')
    p.add_argument('--device',default='cpu');p.add_argument('--size',type=int,default=640);p.add_argument('--width',type=int,default=16)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--conf',type=float,default=.25);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.size<64 or a.size%16:p.error('--size must be >=64 and divisible by 16')
    if a.width<4:p.error('--width must be >=4')
    torch.set_num_threads(a.threads);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    model=ParkitTiny(a.width).to(a.device)
    if a.command=='inspect':
        shapes=[];macs=[0];hooks=[]
        def count(m,inp,res):macs[0]+=res.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d):hooks.append(m.register_forward_hook(count))
        model.eval()
        with torch.no_grad():shapes=[list(t.shape) for t in model(torch.zeros(1,3,a.size,a.size,device=a.device))]
        for h in hooks:h.remove()
        info={'parameters':sum(p.numel() for p in model.parameters()),'conv_GFLOPs_2_per_MAC':2*macs[0]/1e9,'output_shapes':shapes,'note':'Convolution operations only; excludes activations, BN, decode, NMS.'}
        (out/'model_info.json').write_text(json.dumps(info,indent=2));print(info);return
    if a.command=='preview':
        ds=Cars(a.root,size=a.size,augment=True)
        for i in range(min(len(ds),12)):
            x,b,n=ds[i];draw(x,b,out/n)
        print('Saved augmented training previews:',out);return
    if a.command in ['evaluate','benchmark']:
        if not a.weights:p.error('--weights is required')
        model,ckpt=load(a.weights,a.device);a.size=ckpt['size']
        if a.command=='benchmark':
            sample=Cars(a.root,'val',a.size)[0]
            x=sample[0][None].to(a.device)
            def sync():
                if a.device.startswith('cuda'):torch.cuda.synchronize()
                elif a.device=='mps':torch.mps.synchronize()
            times=[]
            with torch.no_grad():
                for i in range(110):
                    sync();t=time.perf_counter();predict(model,x,a.conf);sync()
                    if i>=10:times.append(time.perf_counter()-t)
            result={'mean_ms':float(np.mean(times)*1000),'p95_ms':float(np.percentile(times,95)*1000),'device':a.device,'threads':a.threads,'input':a.size,'image':sample[2],'scope':'first validation image tensor through model + decode + NMS; excludes image preprocessing; 10 warmups and 100 measurements'}
        else:
            ds=Cars(a.root,a.split,a.size);loader=DataLoader(ds,batch_size=a.batch,collate_fn=collate)
            result=evaluate(model,loader,a.device,a.conf,out)
        (out/(a.command+'.json')).write_text(json.dumps(result,indent=2));print(result);return
    overfit=a.command=='overfit'
    ds=Cars(a.root,size=a.size,augment=not overfit,limit=2 if overfit else None)
    loader=DataLoader(ds,batch_size=min(a.batch,len(ds)),shuffle=True,collate_fn=collate)
    val=DataLoader(ds if overfit else Cars(a.root,'val',a.size),batch_size=a.batch,collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    best=-1.;history=[]
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,y,_ in loader:
            opt.zero_grad();loss,parts=detection_loss(model(x.to(a.device)),[b.to(a.device) for b in y])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step();losses.append(float(loss.detach()))
        scheduler.step();metrics=evaluate(model,val,a.device,a.conf)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),**{k:metrics[k] for k in ['precision','recall','mAP50']}};history.append(row)
        ckpt={'model':model.state_dict(),'width':a.width,'size':a.size,'epoch':epoch+1,'seed':42,'overfit_only':overfit}
        save_checkpoint(ckpt,out/'last.pt')
        if metrics['mAP50']>best:best=metrics['mAP50'];save_checkpoint(ckpt,out/'best.pt')
        with (out/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerows(history)
        print(json.dumps(row),flush=True)
    model,_=load(out/'best.pt',a.device);final=evaluate(model,val,a.device,a.conf,out)
    (out/'validation.json').write_text(json.dumps(final,indent=2))
    # Standalone curve SVG, no plotting dependency.
    from .report import curve
    curve(history,out/'learning_curve.svg')

if __name__=='__main__':main()
