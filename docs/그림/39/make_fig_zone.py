import sys
exec(open('/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/fig_v5.py').read().split("if __name__=='__main__':")[0])
import parkit_judge as PJ
from parkit_geom import apply_h
ORIG=PJ._assign
TF,TN=1.5,2.0
def make(mode):
    def _assign(geom, params, px):
        best=None
        if not geom.on_floor(px): return None
        for no,Hs in geom.slot_H.items():
            e=geom.expected.get(no)
            if e is None: continue
            q=apply_h(Hs,[px])[0]
            y0,y1=region_y(geom,no,mode)
            if not (y0<=q[1]<=y1): continue
            d=abs(q[0]-e[0])
            if d<=2.0 and (best is None or d<best[2]): best=(no,q,d)
        return best
    return _assign
def region_y(g,no,mode):
    sl=T[no]; m=1.0
    z0,z1=sl.y0-m, sl.y1+m
    if mode=='now': return z0,z1
    dfy=g.far_dir(sl.center)[1]
    if mode=='m1':
        e0 = m if -dfy<=0 else m*(1-min(1,-dfy))  # y0쪽 변(n=-y): nd=-dfy
        e0 = m*(1-min(1,max(0,-dfy))) if -dfy>0 else m
        e1 = m*(1-min(1,max(0,dfy))) if dfy>0 else m
        return sl.y0-e0, sl.y1+e1
    e=g.expected[no]
    if dfy<0: lo,hi=e[1]-TF,e[1]+TN   # 먼 쪽이 -y
    else: lo,hi=e[1]-TN,e[1]+TF
    return max(z0,lo),min(z1,hi)
def region_poly(g,no,mode):
    e=g.expected[no]; y0,y1=region_y(g,no,mode)
    return np.array([(e[0]-2,y0),(e[0]+2,y0),(e[0]+2,y1),(e[0]-2,y1)])
MODES=[('now','지금\n칸 깊이 앞뒤로 1cm 늘림'),('m1','방법 1 (제안 그대로)\n카메라에서 먼 쪽 변은 늘리지 않음'),('m2',f'방법 2 (예상 접점 기준)\n예상 접점에서 먼 쪽 {TF}cm, 가까운 쪽 {TN}cm까지만')]
RC={0:'#1f77b4',1:'#e377c2',2:'#17becf'}
def figure(camname,slots,cars,xlim,ylim,fname,title,note):
    g,cam=G[camname]
    dets=[car_bbox(g,c) for c in cars]
    fig,axs=plt.subplots(2,3,figsize=(20,11.5),gridspec_kw={'height_ratios':[1,1]})
    for j,(mode,lab) in enumerate(MODES):
        PJ._assign = ORIG if mode=='now' else make(mode)
        res=judge_frame(g,params,[d+[.9] for d in dets])
        truth={T.slot_at((c['cx'],c['cy'])) for c in cars if not c.get('aisle')}
        wrong={no for no in g.monitored if res.obs.get(no)!=(PRESENT if no in truth else ABSENT)}
        print(camname,mode,{n:(res.obs.get(n),res.why.get(n)) for n in wrong})
        unk_ok={n for n in wrong if res.obs.get(n)==OCCLUDED and n not in truth and 'P5' in (res.why.get(n) or '')}
        wrong=wrong-unk_ok
        # 위 줄: 도면
        ax=axs[0,j]
        pass
        topview(ax,g,camname,res,cars,truth,wrong)
        for k,no in enumerate(slots):
            ax.add_patch(Polygon(region_poly(g,no,mode),closed=True,fill=True,fc=RC[k],alpha=0.18,ec=RC[k],lw=2.2,ls='--'))
            e=g.expected[no]; ax.plot(e[0],e[1],'x',ms=10,mew=2.5,color=RC[k])
        for b in res.boxes:
            q=b.contact_mat; ax.plot(q[0],q[1],'o',ms=9,color='yellow',mec='k',zorder=6)
        ax.set_xlim(*xlim); ax.set_ylim(ylim[1],ylim[0]); ax.set_title(lab,fontsize=12)
        fp=sorted(n for n in wrong if res.obs.get(n)==PRESENT and n not in truth)
        miss=sorted(n for n in wrong if n in truth)
        unk=sorted(unk_ok); wrong=wrong-unk_ok
        t=[]
        if fp: t.append('빈 칸을 차 있음으로: '+', '.join(map(str,fp))+'번')
        if miss: t.append('차 있는 칸을 놓침: '+', '.join(map(str,miss))+'번')
        if unk: t.append('통로 차에 가려 모름: '+', '.join(map(str,unk))+'번 (오판 아님)')
        ok=not fp and not miss
        oth=sorted(wrong-set(fp)-set(miss))
        if oth: t.append('걸침 판정 못 해 모름: '+', '.join(f"{n}번" for n in oth))
        ax.text(0.02,0.03,'\n'.join(t) if t else '모두 맞음',transform=ax.transAxes,fontsize=12,color='red' if not ok else 'green',fontweight='bold',bbox=dict(fc='white',ec='none',alpha=0.85))
        # 아래 줄: 카메라
        ax=axs[1,j]
        camview(ax,g,res,cars,dets,wrong)
        pts=[]
        for k,no in enumerate(slots):
            uv=proj_poly(g,region_poly(g,no,mode),dense=20)
            if uv is not None:
                ax.add_patch(Polygon(uv,closed=True,fc=RC[k],alpha=0.18,ec=RC[k],lw=2.2,ls='--')); pts+=list(uv)
            e=g.expected[no]; c,_=project(g.K,g.rvec,g.tvec,[(e[0],e[1],0)]); ax.plot(c[0,0],c[0,1],'x',ms=10,mew=2.5,color=RC[k])
        for c in cars:
            uv,_=project(g.K,g.rvec,g.tvec,car_pts3(c)); pts+=list(uv)
        P=np.array(pts); x0,y0=P.min(0); x1,y1=P.max(0); cx,cy=(x0+x1)/2,(y0+y1)/2
        w=max(x1-x0,(y1-y0)*1.6)*1.25; h=w/1.6
        ax.set_xlim(cx-w/2,cx+w/2); ax.set_ylim(cy+h/2,cy-h/2)
        ax.set_title(f'{camname} 화면 (확대)',fontsize=11)
    PJ._assign=ORIG
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h=[Patch(fc='white',ec='#555',label='칸(cell)')]+[Patch(fc=RC[k],alpha=0.3,ec=RC[k],ls='--',label=f'{no}번 칸으로 인정하는 접점 범위') for k,no in enumerate(slots)]+\
      [Line2D([],[],marker='x',ls='',color='k',mew=2,label='예상 접점(P2b)'),Line2D([],[],marker='o',ls='',color='yellow',mec='k',label='실제 바닥 접점'),
       Patch(fc=COL[PRESENT],label='판정: 차 있음'),Patch(fc=COL[OCCLUDED],label='판정: 가려짐(모름)'),Patch(fc='white',ec='red',lw=2,label='판정이 틀린 칸'),Line2D([],[],ls='--',color='yellow',lw=2,label='YOLO 상자')]
    fig.legend(handles=h,loc='lower center',ncol=6,fontsize=10,frameon=False)
    fig.suptitle(title,fontsize=15,fontweight='bold',y=0.995)
    fig.text(0.5,0.065,note,ha='center',fontsize=11)
    fig.subplots_adjust(left=0.01,right=0.99,top=0.92,bottom=0.12,wspace=0.04,hspace=0.12)
    fig.savefig(OUT+fname,dpi=100); plt.close(fig); print(fname)
def S(no,**k):
    c=dict(cx=T[no].center[0],cy=T[no].center[1]); c.update(k); return c
figure('CAM-2',[11,12,13],[dict(cx=20,cy=18.5,along_x=True,aisle=True,color='#ff7f0e'),S(13)],
       (8,34),(12,30),'39_Z1_판정범위_CAM-2.png',
       'CAM-2(대각선): 통로 차(주황)와 13번에 바르게 선 차(빨강)',
       '통로 차의 접점(노란 점)은 칸 선을 살짝 넘은 칸 안쪽에 찍힌다. 지금과 방법 1은 칸 전체 깊이를 받아 주므로 빈 12번을 차 있음으로 틀린다.\n방법 2는 예상 접점(×) 근처만 받아 주는데, 바르게 선 차의 접점은 × 바로 옆이고 통로 차의 접점은 멀리 있어 걸러진다. 걸러진 통로 차는 12번 앞을 가리므로 12번은 \'모름\'이 된다(맞는 동작).\nCAM-2는 매트 왼쪽 아래에 있으므로 이 줄에서 카메라 쪽은 아래(카스토퍼 쪽), 먼 쪽은 위(통로 A 쪽)다. 그래서 바르게 선 차의 접점(×)도 칸 아래 끝에 찍힌다.')
figure('CAM-1',[9,10,11],[dict(cx=30,cy=18.5,along_x=True,aisle=True,color='#ff7f0e'),S(11)],
       (18,40),(12,30),'38_Z2_판정범위_CAM-1.png',
       'CAM-1(측면): 통로 차(주황)와 11번에 바르게 선 차(빨강)',
       'CAM-1은 매트 오른쪽 옆에서 줄을 따라 보므로 칸의 위·아래 변이 둘 다 "먼 쪽"이 아니다. 그래서 방법 1은 지금과 같다.\n방법 2는 예상 접점의 깊이(y)와 비교하므로 측면에서도 통로 차를 거른다.')
