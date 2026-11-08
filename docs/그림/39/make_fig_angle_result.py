# 그림 S5: 비스듬히 선 차 합성 시험 결과 (두 카메라가 모두 감시하는 칸에서)
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'test_straddle_angle.py')).read().split("Aall=list(range(-85")[0])
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'); plt.rcParams['font.family']='WenQuanYi Zen Hei'
OUT='/mnt/project-files/docs/그림/39/'
Aall=list(range(-85,91,5))
rc=json.load(open(D+'run.json')); params=Params(rc,None)
GG=[]
for cf in ('cam_CAM-1.json','cam_CAM-2.json'):
    g=CamGeom(json.load(open(D+cf)),T,(1920,1080),params); GG.append((g,prep(g,Aall)))
both=GG[0][0].monitored&GG[1][0].monitored
def now(g,b):
    r=judge_frame(g,params,[b+[.9]]); return {n for n,o in r.obs.items() if o==PRESENT},set()
def merge(r):   # 카메라 합치기(P9와 같은 생각): 한 카메라라도 확실히 차있음이면 차있음, 둘 다 모름이면 모름
    (a1,u1),(a2,u2)=r; unk=u1&u2; return ((a1-u1)|(a2-u2))-unk,unk
ROWS=['CAM-1 지금','CAM-1 각도 찾기','CAM-2 지금','CAM-2 각도 찾기','두 카메라 합침 지금','두 카메라 합침 각도 찾기']
GR=['0°','10~30°','45~75°','90°']
stat={(r,k):[0,0,0,0] for r in ROWS for k in GR}   # 맞음, 빈칸→차있음, 놓침, 모름만
for row in ('1-14','28-41'):
    nos=sorted([n for n in both if T[n].row==row],key=lambda n:T[n].x0)
    for no in nos[1:-1]:
        sl=T[no]
        for ang in (0,10,20,30,45,60,75,90,-10,-20,-30,-45,-60,-75):
            for dx in (-0.8,0,0.8):
                fp=foot(sl.center[0]+dx,sl.center[1],ang); tr=blocked(fp,row)
                if not tr<=both: continue
                bs=[bbox(g,fp) for g,_ in GG]
                if None in bs: continue
                k=abs(ang); k='0°' if k==0 else '90°' if k==90 else '10~30°' if k<=30 else '45~75°'
                rn=[now(g,b) for (g,_),b in zip(GG,bs)]
                ra=[(lambda x:(x[0] or set(),x[1]))(estimate(g,E,b,row)) for (g,E),b in zip(GG,bs)]
                res=dict(zip(ROWS,[rn[0],ra[0],rn[1],ra[1],merge(rn),merge(ra)]))
                for name,(got,unk) in res.items():
                    got2=got-unk; st=stat[(name,k)]
                    if got2==tr and not unk: st[0]+=1
                    elif got2-tr: st[1]+=1
                    elif tr-got2-unk: st[2]+=1
                    else: st[3]+=1
for (r,k),v in stat.items(): print(r,k,v)
LAB=['맞음','빈 칸을 차 있음으로','막힌 칸을 놓침','모름(틀리진 않음)']
COL=['#0ca30c','#ec835a','#d03b3b','#b5b3a8']
fig,axs=plt.subplots(1,4,figsize=(18,6.2),sharey=True)
for ax,k in zip(axs,GR):
    for i,r in enumerate(ROWS):
        v=stat[(r,k)]; n=sum(v); left=0
        for j,c in enumerate(v):
            if c==0: continue
            w=c/n*100
            ax.barh(i,w-0.6 if w>1 else w,left=left,color=COL[j],height=0.62)
            if w>=9: ax.text(left+w/2,i,str(c),ha='center',va='center',color='white',fontsize=10.5)
            left+=w
        ax.text(101,i,f'/ {n}',va='center',fontsize=10,color='#555')
    ax.set_xlim(0,112); ax.set_title(f'차 각도 {k}',fontsize=13)
    ax.set_xticks([0,50,100]); ax.set_xticklabels(['0%','50%','100%'],color='#666')
    for s in ('top','right','left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7'); ax.tick_params(axis='y',length=0)
    ax.axhline(1.5,color='#e1e0d9',lw=1); ax.axhline(3.5,color='#e1e0d9',lw=1)
axs[0].set_yticks(range(len(ROWS))); axs[0].set_yticklabels(ROWS,fontsize=11.5); axs[0].invert_yaxis()
fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in COL],LAB,loc='upper center',ncol=4,fontsize=12,frameon=False,bbox_to_anchor=(0.55,0.9))
fig.suptitle('그림 S5. 비스듬히 선 차 합성 시험: 막대 안 숫자는 경우의 수(cm·픽셀 아님), 막대 길이는 비율',fontsize=15,fontweight='bold')
fig.text(0.5,0.02,f'두 카메라가 모두 감시하는 {len(both)}칸(양 끝 칸 제외)에서, 칸마다 각도 14가지 × 좌우 위치 3가지로 차를 놓았다. 각도 찾기는 5° 간격. '
 '한 경우에 틀림과 모름이 같이 있으면 틀림으로 셌다.',ha='center',fontsize=11.5)
fig.subplots_adjust(left=0.13,right=0.99,top=0.78,bottom=0.12,wspace=0.06)
fig.savefig(OUT+'39_S5_비스듬히_시험결과.png',dpi=95); print('ok')
