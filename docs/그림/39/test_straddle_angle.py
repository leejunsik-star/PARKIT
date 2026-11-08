# 비스듬히 선 차 합성 시험(각도 찾기). 환경변수 TOL=0.1(모름 허용) 또는 TOL=0(항상 하나로 정함). 경로는 파일 맨 위에서 고친다
import sys, json, math, os
sys.path.insert(0,'/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/v5')
import numpy as np, cv2
from parkit_config import Params
from parkit_geom import SlotTable, CamGeom, project, apply_h
from parkit_judge import judge_frame, PRESENT
D='/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/v5/config/'
T=SlotTable('/home/user/parkit/docs/칸표_매트좌표_초안.json')
W,L,H=3.2,6.5,2.0
def foot(cx,cy,ang,w=W,l=L):
    a=math.radians(ang); c,s=math.cos(a),math.sin(a)
    return np.array([(cx+dx*c-dy*s,cy+dx*s+dy*c) for dx,dy in ((-w/2,-l/2),(w/2,-l/2),(w/2,l/2),(-w/2,l/2))])
def bbox(g,fp):
    P=np.vstack([np.c_[fp,np.zeros(4)],np.c_[fp,np.full(4,-H)]])
    uv,z=project(g.K,g.rvec,g.tvec,P)
    if (z<=0).any(): return None
    Wd,Hd=g.size; return [max(0,uv[:,0].min()),max(0,uv[:,1].min()),min(Wd,uv[:,0].max()),min(Hd,uv[:,1].max())]
def blocked(fp,row,thr=0.1):
    out=set()
    for sl in T:
        if sl.row!=row: continue
        inter,poly=cv2.intersectConvexConvex(fp.astype(np.float32),sl.inner.astype(np.float32))
        if inter>0 and poly is not None:
            p=poly.reshape(-1,2)
            if min(np.ptp(p[:,0]),np.ptp(p[:,1]))>=thr: out.add(sl.no)
    return out
def prep(g,angles):
    E={}
    for sl in T:
        if sl.no not in g.slot_H: continue
        for a in angles:
            b=bbox(g,foot(*sl.center,a))
            if b is None: continue
            c=apply_h(g.slot_H[sl.no],[((b[0]+b[2])/2,b[3])])[0]; E[(sl.no,a)]=(b,c)
    return E
TOL=float(os.environ.get('TOL','0.1'))
def estimate(g,E,b,row,tol=None):
    tol=TOL if tol is None else tol
    cands=[]; bw,bh=b[2]-b[0],b[3]-b[1]
    for (no,a),(eb,ec) in E.items():
        if T[no].row!=row: continue
        q=apply_h(g.slot_H[no],[((b[0]+b[2])/2,b[3])])[0]; dx=q[0]-ec[0]; dy=q[1]-ec[1]
        if abs(dx)>2.5 or abs(dy)>1.5: continue
        size=abs(math.log(bw/(eb[2]-eb[0])))+abs(math.log(bh/(eb[3]-eb[1])))
        cands.append((abs(dx)/2+abs(dy)+3*size,no,a,T[no].center[0]+dx,T[no].center[1]+dy))
    cands.sort()
    if not cands: return None,set()
    best=cands[0]; res=blocked(foot(best[3],best[4],best[2]),row)
    # 비슷하게 맞는 후보들이 다른 결과를 내면 그 차이 칸은 '모름'
    unk=set()
    for c in cands[1:]:
        if c[0]-best[0]>tol: break
        unk|=blocked(foot(c[3],c[4],c[2]),row)^res
    return res,unk
def run(camf,angles,mode):
    rc=json.load(open(D+'run.json')); params=Params(rc,None)
    cam=json.load(open(D+camf)); g=CamGeom(cam,T,(1920,1080),params); E=prep(g,angles)
    stat={}
    for row in ('1-14','28-41'):
        nos=sorted([n for n in g.monitored if T[n].row==row],key=lambda n:T[n].x0)
        for no in nos[1:-1]:
            sl=T[no]
            for ang in (0,10,20,30,45,60,75,90,-10,-20,-30,-45,-60,-75):
                for dx in (-0.8,0,0.8):
                    fp=foot(sl.center[0]+dx,sl.center[1],ang)
                    tr=blocked(fp,row)
                    if not tr<=g.monitored: continue
                    b=bbox(g,fp)
                    if b is None: continue
                    k=abs(ang); k='0' if k==0 else '90' if k==90 else '10~30' if k<=30 else '45~75'
                    st=stat.setdefault(k,[0,0,0,0])   # 경우, 맞음, 빈칸을 차있음, 차있는칸 놓침
                    if mode=='now':
                        r=judge_frame(g,params,[b+[.9]]); got={n for n,o in r.obs.items() if o==PRESENT}; unk=set()
                    else:
                        got,unk=estimate(g,E,b,row)
                    got2=got-unk
                    st[0]+=1; st[1]+=(got2==tr and not unk) ; st[2]+=bool(got2-tr); st[3]+=bool(tr-got2-unk)
    return {k:f'경우 {v[0]}, 정확히 맞음 {v[1]}, 빈칸→차있음 {v[2]}, 막힌칸 놓침 {v[3]}' for k,v in sorted(stat.items())}
Aall=list(range(-85,91,5))
for camf in ('cam_CAM-1.json','cam_CAM-2.json'):
    for mode,ang in (('now',[0,90]),('각도찾기',Aall)):
        if mode=='now' and os.environ.get('SKIPNOW'): continue
        print(camf,mode,'tol',TOL); [print('   ',k,v) for k,v in run(camf,ang,mode).items()]
