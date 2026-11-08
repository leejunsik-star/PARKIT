import json, sys, math
sys.path.insert(0, '/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/v5')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon, Rectangle
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']='WenQuanYi Zen Hei'
from parkit_config import Params
from parkit_geom import SlotTable, CamGeom, project
from parkit_judge import judge_frame, PRESENT, OCCLUDED, ABSENT
OUT='/mnt/project-files/docs/그림/39/'
D='/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/v5/config/'
run=json.load(open(D+'run.json')); params=Params(run,None)
T=SlotTable('/home/user/parkit/docs/칸표_매트좌표_초안.json')
MW,MH=T.mat_size
G={}
for n in ('CAM-1','CAM-2'):
    cam=json.load(open(D+f'cam_{n}.json')); G[n]=(CamGeom(cam,T,(1920,1080),params),cam)
CC={'CAM-1':'#1f77b4','CAM-2':'#9467bd'}
COL={PRESENT:'#2ca02c',OCCLUDED:'#ff9f1c',None:'#9e9e9e'}
LAB={PRESENT:'차 있음',OCCLUDED:'가려짐(모름)',None:'모름'}

def corners(c):
    w,l=c.get('w',3.2),c.get('l',6.5)
    if c.get('along_x'): w,l=l,w
    a=math.radians(c.get('ang',0)); co,si=math.cos(a),math.sin(a)
    out=[]
    for dx,dy in ((-w/2,-l/2),(w/2,-l/2),(w/2,l/2),(-w/2,l/2)):
        out.append((c['cx']+dx*co-dy*si, c['cy']+dx*si+dy*co))
    return np.array(out)
def car_pts3(c):
    f=corners(c); h=c.get('h',2.0)
    return np.vstack([np.hstack([f,np.zeros((4,1))]), np.hstack([f,np.full((4,1),-h)])])
def car_bbox(g,c):
    uv,z=project(g.K,g.rvec,g.tvec,car_pts3(c))
    if (z<=0).any(): return None
    W,H=g.size
    b=[max(0,uv[:,0].min()),max(0,uv[:,1].min()),min(W,uv[:,0].max()),min(H,uv[:,1].max())]
    return b
def hull(p):
    p=sorted(map(tuple,p))
    def cr(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[];up=[]
    for q in p:
        while len(lo)>=2 and cr(lo[-2],lo[-1],q)<=0: lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up)>=2 and cr(up[-2],up[-1],q)<=0: up.pop()
        up.append(q)
    return np.array(lo[:-1]+up[:-1])
def proj_poly(g,pts,z=0.0,dense=12):
    pts=np.asarray(pts,float); out=[]
    for i in range(len(pts)):
        a,b=pts[i],pts[(i+1)%len(pts)]
        for t in np.linspace(0,1,dense,endpoint=False): out.append(a+(b-a)*t)
    out=np.array(out); p3=np.hstack([out,np.full((len(out),1),z)])
    uv,zz=project(g.K,g.rvec,g.tvec,p3)
    return None if (zz<=0).any() else uv

def topview(ax,g=None,camname=None,res=None,cars=(),truth=set(),wrong=set(),show_both=False,mon_only=None):
    ax.add_patch(Rectangle((0,0),MW,MH,fc='#f4f1ea',ec='k',lw=1.2))
    for sl in T:
        fc='white'
        if res is not None and sl.no in res.obs and res.obs[sl.no]!=ABSENT: fc=COL[res.obs[sl.no]]
        elif mon_only is not None and sl.no in mon_only: fc=mon_only[sl.no]
        ax.add_patch(Polygon(sl.cell,closed=True,fc=fc,ec='#555',lw=0.6,alpha=0.85))
        ax.text(*sl.center,str(sl.no),ha='center',va='center',fontsize=6,color='#333',clip_on=True)
        if sl.no in wrong:
            ax.add_patch(Polygon(sl.cell,closed=True,fill=False,ec='red',lw=2.5))
    for c in cars:
        ax.add_patch(Polygon(corners(c),closed=True,fc=c.get('color','#d62728'),ec='k',lw=0.8,alpha=0.9))
    cams=G.items() if show_both else [(camname,G[camname])]
    for n,(gg,cam) in cams:
        p=cam['pose']['cam_pos_cm']
        # 화면에 보이는 바닥
        xs,ys=np.meshgrid(np.arange(-15,95,0.7),np.arange(-15,72,0.7))
        P=np.c_[xs.ravel(),ys.ravel(),np.zeros(xs.size)]
        uv,z=project(gg.K,gg.rvec,gg.tvec,P)
        ok=(z>0)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
        if ok.sum()>3:
            hh=hull(P[ok,:2]); ax.add_patch(Polygon(hh,closed=True,fc=CC[n],alpha=0.08,ec=CC[n],ls='--',lw=1))
        ax.plot(p[0],p[1],'^' if True else 'o',ms=11,color=CC[n],mec='k')
        ax.text(p[0],p[1]+3.2,f'{n}\n높이 {p[2]:.0f}cm',ha='center',va='top',fontsize=8,color=CC[n],fontweight='bold',clip_on=True)
        # 보는 방향
        c0=gg.cam_ground; tgt=np.array([MW/2,MH/2]); d=tgt-c0; d/=np.linalg.norm(d)
        ax.annotate('',xy=c0+d*14,xytext=c0,arrowprops=dict(arrowstyle='->',color=CC[n],lw=2))
    if res is not None:
        for b in res.boxes:
            q=b.contact_mat; ax.plot(q[0],q[1],'o',ms=6,color='yellow',mec='k',zorder=5)
    ax.set_xlim(-32,96); ax.set_ylim(70,-16); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

def camview(ax,g,res=None,cars=(),dets=(),wrong=set(),labels=True):
    W,H=g.size
    ax.add_patch(Rectangle((0,0),W,H,fc='#cfcfcf'))
    m=proj_poly(g,[(0,0),(MW,0),(MW,MH),(0,MH)],dense=60)
    if m is not None: ax.add_patch(Polygon(m,closed=True,fc='#f4f1ea',ec='k',lw=1))
    for sl in T:
        uv=proj_poly(g,sl.cell)
        if uv is None: continue
        fc='none'
        if res is not None and sl.no in res.obs and res.obs[sl.no]!=ABSENT: fc=COL[res.obs[sl.no]]
        ax.add_patch(Polygon(uv,closed=True,fc=fc,ec='#555',lw=0.6,alpha=0.6 if fc!='none' else 1))
        if sl.no in wrong: ax.add_patch(Polygon(uv,closed=True,fill=False,ec='red',lw=2.5))
        if labels:
            c,zz=project(g.K,g.rvec,g.tvec,[(sl.center[0],sl.center[1],0)])
            if 0<c[0,0]<W and 0<c[0,1]<H:
                ax.text(c[0,0],c[0,1],str(sl.no),ha='center',va='center',fontsize=7,color='#222',clip_on=True)
    # 차: 먼 것부터
    cg=g.cam_ground
    for c in sorted(cars,key=lambda c:-np.linalg.norm(np.array([c['cx'],c['cy']])-cg)):
        uv,z=project(g.K,g.rvec,g.tvec,car_pts3(c))
        if (z<=0).any(): continue
        ax.add_patch(Polygon(hull(uv),closed=True,fc=c.get('color','#d62728'),ec='k',lw=0.8))
        ax.add_patch(Polygon(uv[4:],closed=True,fc='white',alpha=0.25,ec='k',lw=0.5))
    for b in dets:
        ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='yellow',lw=1.6,ls='--'))
    if res is not None:
        for b in res.boxes:
            ax.plot(*b.contact_px,'o',ms=6,color='yellow',mec='k')
    ax.set_xlim(0,W); ax.set_ylim(H,0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

def legend(fig):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h=[Patch(fc=COL[PRESENT],label='판정: 차 있음'),Patch(fc=COL[OCCLUDED],label='판정: 가려짐(모름)'),
       Patch(fc='white',ec='red',lw=2,label='틀린 칸(빈 칸↔차 있음)'),Patch(fc='#d62728',ec='k',label='실제 차'),
       Line2D([],[],ls='--',color='yellow',lw=2,label='YOLO 상자(예상)'),
       Line2D([],[],marker='o',ls='',color='yellow',mec='k',label='바닥 접점')]
    fig.legend(handles=h,loc='lower center',ncol=6,fontsize=9,frameon=False)

def scene(fname,camname,cars,title,note,extra_dets=None,hide=(),truth_add=()):
    g,cam=G[camname]
    dets=[]
    for i,c in enumerate(cars):
        if i in hide or c.get('nodet'): continue
        b=car_bbox(g,c)
        if b: dets.append(b)
    if extra_dets is not None: dets=extra_dets(g,dets)
    res=judge_frame(g,params,[d+[0.9] for d in dets])
    truth=set()
    for c in cars:
        n=T.slot_at((c['cx'],c['cy']))
        if n and not c.get('aisle'): truth.add(n)
    truth|=set(truth_add)
    wrong=set()
    for no in g.monitored:
        o=res.obs.get(no)
        if (no in truth and o==ABSENT) or (no not in truth and o==PRESENT): wrong.add(no)
    fig,axs=plt.subplots(1,3,figsize=(20,5.8),gridspec_kw={'width_ratios':[1,1.3,1.1]})
    topview(axs[0],g,camname,res,cars,truth,wrong); axs[0].set_title('위에서 본 모습',fontsize=11)
    if camname=='CAM-1': axs[0].set_xlim(-32,81); axs[0].set_ylim(58,-3)
    else: axs[0].set_xlim(-16,81); axs[0].set_ylim(65,-3)
    camview(axs[1],g,res,cars,dets,wrong); axs[1].set_title(f'{camname} 화면 예상도 (1920×1080)',fontsize=11)
    camview(axs[2],g,res,cars,dets,wrong)
    pts=[]
    for c in cars:
        uv,z=project(g.K,g.rvec,g.tvec,car_pts3(c)); pts+=list(uv)
    for d in dets: pts+=[(d[0],d[1]),(d[2],d[3])]
    pts=np.array(pts); x0,y0=pts.min(0); x1,y1=pts.max(0)
    cx,cy=(x0+x1)/2,(y0+y1)/2; w=max(x1-x0,(y1-y0)*1.6)*1.9; h=w/1.6
    axs[2].set_xlim(cx-w/2,cx+w/2); axs[2].set_ylim(cy+h/2,cy-h/2)
    axs[2].set_title('화면 확대',fontsize=11)
    axs[1].add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fill=False,ec='#0057ff',lw=1.5))
    fig.suptitle(title,fontsize=14,fontweight='bold',y=0.99)
    fig.text(0.5,0.085,note,ha='center',fontsize=10)
    legend(fig); fig.subplots_adjust(left=0.01,right=0.99,top=0.9,bottom=0.15,wspace=0.03)
    fig.savefig(OUT+fname,dpi=110); plt.close(fig)
    print(fname,'truth',sorted(truth),'wrong',sorted(wrong),{k:v for k,v in res.obs.items() if v!=ABSENT})
    return res

if __name__=='__main__':
    # 그림 1: 배치
    fig,ax=plt.subplots(figsize=(10,7.4))
    m={}
    for n,(g,_) in G.items():
        for no in g.monitored: m[no]= '#c8b8e8' if no in m else ('#bcd6f0' if n=='CAM-1' else '#e3d3f5')
    for no in set(G['CAM-1'][0].monitored)&set(G['CAM-2'][0].monitored): m[no]='#a9dfbf'
    topview(ax,show_both=True,mon_only=m)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc='#bcd6f0',label='CAM-1만 감시'),Patch(fc='#e3d3f5',label='CAM-2만 감시'),Patch(fc='#a9dfbf',label='두 대 모두 감시'),Patch(fc='white',ec='#555',label='감시 안 함')],loc='lower right',fontsize=9)
    for y,t in ((16.1,'통로 A'),(28.7,'분리대'),(41.8,'통로 B')): ax.text(MW/2,y,t,ha='center',va='center',fontsize=9,color='#777')
    ax.set_title('그림 1. 위에서 본 배치 (11-07 보정 파일 기준, 점선 = 화면에 보이는 바닥)',fontsize=12)
    fig.tight_layout(); fig.savefig(OUT+'39_1_배치_위.png',dpi=110); plt.close(fig)
    for i,n in enumerate(G):
        g,cam=G[n]; p=cam['pose']
        fig,ax=plt.subplots(figsize=(12,7))
        camview(ax,g)
        ax.set_title(f'그림 {i+2}. {n} 화면 예상도 (빈 매트, 칸 번호)  렌즈 x {p["cam_pos_cm"][0]:.0f}, y {p["cam_pos_cm"][1]:.0f}, 높이 {p["cam_pos_cm"][2]:.0f}cm / 아래로 {p["pitch_deg"]:.0f}°',fontsize=12)
        fig.tight_layout(); fig.savefig(OUT+f'39_{i+2}_{n}_화면.png',dpi=110); plt.close(fig)
