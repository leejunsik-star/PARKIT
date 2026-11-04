"""M1~M8(YOLO26n 파인튜닝)과 mymodel_M1/M5/M8(동료 자체 모델)을 split_3차 T1~T8로 같은 방식으로 채점 (docs\\29)
live_detect_winmac.py의 Detector로 불러와 실시간 프로그램과 같은 전처리로 돌린다.
mAP50은 신뢰도 0.001 이상 상자로 직접 계산(IoU 0.5, 101점 보간). 놓친 차·잘못 잡은 상자·P·R은 신뢰도 0.25.
사용법 (docs\\도구 에서): python eval_모델비교.py   → eval_모델비교_결과.json
"""
import os, sys, time, json, glob
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.argv=['x']; sys.path.insert(0,HERE)
import live_detect_winmac as L
torch.set_num_threads(4)
SPLIT=os.path.normpath(os.path.join(HERE,'..','..','dataset','split_3차'))
POS=['T1','T2','T3','T6','T8']; NEG=['T4','T5','T7']
def imgs(t):
    return [os.path.join(SPLIT,*x[2:].split('/')) for x in open(f'{SPLIT}/test_{t}.txt',encoding='utf-8').read().splitlines() if x]
def gts(img,w,h):
    lab=os.path.join(SPLIT,'labels','test',os.path.splitext(os.path.basename(img))[0]+'.txt'); o=[]
    for r in open(lab,encoding='utf-8').read().splitlines():
        p=r.split()
        if len(p)==5:
            cx,cy,bw,bh=map(float,p[1:]); o.append(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h))
    return o
def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0])); ih=max(0,min(a[3],b[3])-max(a[1],b[1])); i=iw*ih
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i; return i/u if u>0 else 0
def match(preds,g):
    """preds [(conf,xyxy)] -> list of (conf, is_tp) (greedy, conf order, IoU>=0.5, 1:1)"""
    used=set(); out=[]
    for c,p in sorted(preds,key=lambda x:-x[0]):
        best,bi=0.5,None
        for i,gg in enumerate(g):
            if i not in used:
                v=iou(p,gg)
                if v>=best: best,bi=v,i
        if bi is not None: used.add(bi)
        out.append((c,bi is not None))
    return out
def ap50(scored,ngt):
    if ngt==0: return float('nan')
    s=sorted(scored,key=lambda x:-x[0]); tp=np.cumsum([t for _,t in s]); fp=np.cumsum([not t for _,t in s])
    rec=tp/ngt; prec=tp/np.maximum(tp+fp,1e-9)
    mrec=np.concatenate([[0],rec,[1]]); mpre=np.concatenate([[1],prec,[0]])
    mpre=np.flip(np.maximum.accumulate(np.flip(mpre)))
    x=np.linspace(0,1,101); return float(np.trapezoid(np.interp(x,mrec,mpre),x))
names=[f'M{i}' for i in range(1,9)]+['mymodel_M1','mymodel_M5','mymodel_M8']
res={}
cache={}
for n in names:
    d=L.Detector('custom',f'models/{n}.pt',[])
    t0=time.time(); r={}; times=[]
    for t in POS+NEG:
        allsc=[]; ngt=0; miss=fp=fpimg=0; nimg=0; tp25=0; np25=0
        for p in imgs(t):
            im=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_COLOR); h,w=im.shape[:2]
            s=time.perf_counter(); dets=d.run(im,640,0.001) if hasattr(d,'run') else None; times.append(time.perf_counter()-s)
            pr=[(c,(x1,y1,x2,y2)) for (_,c,x1,y1,x2,y2) in dets]
            g=gts(p,w,h); ngt+=len(g); nimg+=1
            allsc+=match(pr,g)
            m25=match([x for x in pr if x[0]>=0.25],g); tp=sum(k for _,k in m25)
            miss+=len(g)-tp; fp+=len(m25)-tp; fpimg+=len(m25)>tp; tp25+=tp; np25+=len(m25)
        r[t]=dict(ngt=ngt,img=nimg,miss=miss,fp=fp,fpimg=fpimg,P=tp25/np25 if np25 else float('nan'),R=tp25/ngt if ngt else float('nan'),AP50=ap50(allsc,ngt),scored=[(float(c),bool(k)) for c,k in allsc])
        print(n,t,{k:v for k,v in r[t].items() if k!='scored'},flush=True)
    r['ms_conf0.001']=float(np.median(times)*1000)
    res[n]=r; print(n,'done',time.time()-t0,flush=True)
    json.dump(res,open(os.path.join(HERE,'eval_모델비교_결과.json'),'w'))
