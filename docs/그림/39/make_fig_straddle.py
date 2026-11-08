exec(open('/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/fig_v5.py').read().split("if __name__=='__main__':")[0])
from matplotlib.patches import FancyArrowPatch
# ---------- 그림 S1: 기준(0.1)의 뜻 ----------
def slots_draw(ax, nos, yl):
    for no in nos:
        sl=T[no]
        ax.add_patch(Rectangle((sl.x0,sl.y0),sl.x1-sl.x0,sl.y1-sl.y0,fc='#e9e4d8',ec='#333',lw=1.0))   # 칸 전체(cell)
        ax.add_patch(Rectangle((sl.ix0,sl.y0+0.2),sl.ix1-sl.ix0,sl.y1-sl.y0-0.4,fc='white',ec='#888',lw=0.8,ls=':'))  # 빈 공간(inner)
        ax.text(sl.center[0],sl.y1-0.5,f'{no}번',ha='center',va='bottom',fontsize=11,fontweight='bold')
fig,axs=plt.subplots(1,3,figsize=(20,7))
nos=[5,4,3]; sl4=T[4]; cy=sl4.center[1]
cases=[('① 4번 가운데에 바르게 선 차',[(sl4.center[0],3.2,6.5)],'옆 칸 빈 공간에 안 들어감\n→ 4번만 차 있음 (기준 0.1, 1.0 모두)'),
       ('② 3번 쪽으로 1cm 비켜 선 차',[(sl4.center[0]+1,3.2,6.5)],'칸 선을 넘어 3번 빈 공간에 0.14cm 들어감\n→ 같은 크기 차는 3번에 선을 넘지 않고는 못 선다\n→ 기준 0.1: 3·4번 차 있음 (hun 기준으로 맞음) / 1.0: 4번만'),
       ('③ 4번에 가로로 선 차',[(sl4.center[0],6.5,3.2)],'x 길이 6.5cm라 3·5번 빈 공간에 0.8cm씩 들어감\n→ 3·4·5번 모두 막힘\n→ 지금 프로그램은 차 폭을 3.2cm로 보므로 4번만')]
for ax,(title,cars,note) in zip(axs,cases):
    slots_draw(ax,nos,None)
    for cx,w,l in cars:
        ax.add_patch(Rectangle((cx-w/2,cy-l/2),w,l,fc='#d62728',ec='k',lw=1.2,alpha=0.75))
        # 프로그램이 보는 범위 (고정 폭 3.2)
        ax.annotate('',xy=(cx-1.6,cy-l/2-0.9),xytext=(cx+1.6,cy-l/2-0.9),arrowprops=dict(arrowstyle='<->',color='#1f77b4',lw=2))
        ax.text(cx,cy-l/2-1.2,'프로그램이 보는 차 폭 3.2cm',ha='center',va='bottom',color='#1f77b4',fontsize=10)
        if w>4:
            ax.annotate('',xy=(cx-w/2,cy+l/2+0.7),xytext=(cx+w/2,cy+l/2+0.7),arrowprops=dict(arrowstyle='<->',color='#d62728',lw=2))
            ax.text(cx,cy+l/2+1.0,'실제 차 x 길이 6.5cm',ha='center',va='top',color='#d62728',fontsize=10)
        # 침범 표시
        for sl in (T[3],T[5]):
            ov=min(cx+w/2,sl.ix1)-max(cx-w/2,sl.ix0)
            if ov>0:
                x0=max(cx-w/2,sl.ix0)
                ax.add_patch(Rectangle((x0,cy-l/2),ov,l,fc='none',ec='#ff7f0e',lw=2.5,hatch='///'))
                ax.text(x0+ov/2,cy,f'{ov:.2f}cm\n침범',ha='center',va='center',fontsize=10,color='#b35900',fontweight='bold',bbox=dict(fc='white',ec='none',alpha=0.8))
    ax.set_xlim(T[5].x0-0.3,T[3].x1+0.3); ax.set_ylim(sl4.y1+3.0,sl4.y0-3.0); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.text(0.5,-0.04,note,transform=ax.transAxes,ha='center',va='top',fontsize=11)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc='#e9e4d8',ec='#333',label='칸 전체(cell, 폭 4.03cm). 칸 사이 회색 띠 = 칸 선(약 0.9cm)'),Patch(fc='white',ec='#888',ls=':',label='빈 공간(inner, 폭 약 3.1cm)'),
                    Patch(fc='#d62728',alpha=0.75,label='실제 차'),Patch(fc='none',ec='#ff7f0e',hatch='///',label='옆 칸 빈 공간 침범')],loc='lower center',ncol=4,fontsize=11,frameon=False)
fig.suptitle('그림 S1. 걸침 기준 straddle_min_cm 0.1의 뜻: "칸 선을 넘어 옆 칸 빈 공간에 0.1cm라도 들어가면 막힘" (위에서 본 4번 칸 주변)',fontsize=14,fontweight='bold')
fig.subplots_adjust(left=0.01,right=0.99,top=0.88,bottom=0.25,wspace=0.05)
fig.savefig(OUT+'39_S1_걸침기준.png',dpi=100); plt.close(fig); print('S1')

# ---------- 그림 S2: 방향 추정 ----------
import json as _j
exec(open('/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/st4.py').read().split("def run3")[0].split("exec(open('st.py')")[0])
from parkit_geom import apply_h, car_box
Wc,Lc=3.2,6.5
def bx(g,x0,x1,y0,y1): return car_box(g.K,g.rvec,g.tvec,x0,x1,y0,y1,2.0,g.size)
def prep(g):
    E={}
    for sl in T:
        if sl.no not in g.slot_H: continue
        cx,cy=sl.center
        for o,(hx,hy) in (('n',(Wc/2,Lc/2)),('s',(Lc/2,Wc/2))):
            b=bx(g,cx-hx,cx+hx,cy-hy,cy+hy)
            if b is None: continue
            c=apply_h(g.slot_H[sl.no],[((b[0]+b[2])/2,b[3])])[0]; E[(sl.no,o)]=(b,c,hx)
    return E
def estimate(g,E,b,row):
    best=None; bw,bh=b[2]-b[0],b[3]-b[1]
    for (no,o),(eb,ec,hx) in E.items():
        if T[no].row!=row: continue
        q=apply_h(g.slot_H[no],[((b[0]+b[2])/2,b[3])])[0]; dx=q[0]-ec[0]; dy=q[1]-ec[1]
        if abs(dx)>2.5 or abs(dy)>1.5: continue
        ew,eh=eb[2]-eb[0],eb[3]-eb[1]; size=abs(math.log(bw/ew))+abs(math.log(bh/eh))
        cost=abs(dx)/2+abs(dy)+3*size
        if best is None or cost<best[0]: best=(cost,no,o,T[no].center[0]+dx,hx)
    return best
for camname,slot in (('CAM-2',9),('CAM-1',9)):
    g,cam=G[camname]; E=prep(g)
    car=dict(cx=T[slot].center[0],cy=T[slot].center[1],w=6.5,l=3.2)
    b=car_bbox(g,car); res=judge_frame(g,params,[b+[.9]])
    est=estimate(g,E,b,T[slot].row); _,s,o,cx,hx=est
    got={sl.no for sl in T if sl.row==T[slot].row and min(cx+hx,sl.ix1)-max(cx-hx,sl.ix0)>=0.1}
    truth={slot,T[slot].left,T[slot].right}
    class R: pass
    r2=R(); r2.obs={n:(PRESENT if n in got else ABSENT) for n in g.monitored}; r2.boxes=res.boxes
    fig,axs=plt.subplots(2,2,figsize=(18,12))
    for j,(rr,lab) in enumerate(((res,'지금: 차를 늘 "바르게, 폭 3.2cm"로 봄'),(r2,f'방향 추정: "{ "가로" if o=="s" else "바르게"}"로 고르고 x 길이 {2*hx:.1f}cm로 봄'))):
        wrong={n for n in g.monitored if (n in truth)!=(rr.obs.get(n)==PRESENT)}
        ax=axs[0,j]; topview(ax,g,camname,rr,[car],truth,wrong)
        x0=T[slot].x0; ax.set_xlim(x0-10,x0+14); ax.set_ylim(T[slot].y1+4,T[slot].y0-4)
        ax.set_title(lab,fontsize=13,fontweight='bold')
        pr=sorted(n for n in g.monitored if rr.obs.get(n)==PRESENT)
        ax.text(0.02,0.03,f'차 있음: {", ".join(map(str,pr))}번  (정답 {", ".join(map(str,sorted(truth)))}번)',transform=ax.transAxes,fontsize=12,fontweight='bold',color='green' if not wrong else 'red',bbox=dict(fc='white',ec='none',alpha=0.85))
        ax=axs[1,j]; camview(ax,g,rr,[car],[b],wrong)
        if j==1:
            for (oo,colr,nm) in (('n','#1f77b4','바르게 선 차 예상 상자'),('s','#e377c2','가로로 선 차 예상 상자')):
                eb=E[(slot,oo)][0]
                ax.add_patch(Rectangle((eb[0],eb[1]),eb[2]-eb[0],eb[3]-eb[1],fill=False,ec=colr,lw=2.5,ls='--',label=nm))
                ax.plot((eb[0]+eb[2])/2,eb[3],'X',ms=14,color=colr,mec='k',label=nm.replace('상자','접점'))
            ax.legend(loc='upper left',fontsize=10)
        P=np.array([[b[0],b[1]],[b[2],b[3]]]); c=P.mean(0); w=(b[2]-b[0])*3.2; h=w/1.6
        ax.set_xlim(c[0]-w/2,c[0]+w/2); ax.set_ylim(c[1]+h/2,c[1]-h/2)
        ax.set_title(f'{camname} 화면 (확대). 노란 점선 = YOLO 상자' + ('' if j==0 else ', 파랑·분홍 = 이 칸의 두 예상 상자'),fontsize=11)
    fig.suptitle(f'그림 S2-{camname}. {slot}번에 가로로 선 차 (3칸을 막음): 지금 방식 vs 방향 추정',fontsize=15,fontweight='bold')
    fig.text(0.5,0.035,'방향 추정은 YOLO 상자의 폭·높이와 접점(노란 점)을 이 칸의 두 예상(바르게 = 파랑, 가로 = 분홍)과 비교해 더 맞는 쪽을 고른다. 카메라마다 상자 크기 차이가 클 때도, 접점 위치 차이가 클 때도 있다.\n가로로 고르면 차 x 길이를 6.5cm로 써서 옆 칸 침범을 다시 잰다.\n'
             '지금 방식은 접점이 한쪽 끝으로 쏠려 칸을 옆 칸으로 고르거나, 폭 3.2cm로 계산해 옆 칸 침범을 못 본다.',ha='center',fontsize=12)
    fig.subplots_adjust(left=0.01,right=0.99,top=0.93,bottom=0.09,wspace=0.04,hspace=0.12)
    fig.savefig(OUT+f'39_S2_방향추정_{camname}.png',dpi=95); plt.close(fig); print(camname,slot,'est',s,o,sorted(got),'now',sorted(n for n,v in res.obs.items() if v==PRESENT))
