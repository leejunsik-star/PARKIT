exec(open('/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/fig_st.py').read().split("# ---------- 그림 S1")[0])
exec(open('/tmp/claude-0/-home-user-parkit/354fbfa9-ad95-5b45-8b60-1224343afb3b/scratchpad/fig_st.py').read().split("# ---------- 그림 S1: 기준(0.1)의 뜻 ----------")[1].split("fig,axs=plt.subplots(1,3")[0])
sl4=T[4]; cy=sl4.center[1]; nos=[5,4,3]
fig,axs=plt.subplots(1,3,figsize=(21,7.5))
sets=[]
a=[]
for w in (2.8,3.2,3.6):
    for dx in (-1,-0.5,0,0.5,1): a.append((sl4.center[0]+dx,w,6.5))
sets.append(('① 바르게 선 차: 15가지',a,'차 폭 2.8 / 3.2 / 3.6cm × 칸 가운데에서 좌우로 -1, -0.5, 0, +0.5, +1cm\n칸 하나에 15가지. CAM-1은 이런 칸 24개 → 360가지'))
b=[(sl4.ix1+0.45+dx,6.5,3.2) for dx in np.arange(-2.0,2.01,0.5)]
sets.append(('② 가로로 선 차, 두 칸 사이: 9가지',b,'차 가운데를 4번·3번 사이 칸 선에서 -2 ~ +2cm, 0.5cm 간격\n칸 하나에 9가지. CAM-1 24칸 → 216가지'))
c=[(sl4.center[0]+dx,6.5,3.2) for dx in (-0.6,-0.3,0,0.3,0.6)]
sets.append(('③ 가로로 선 차, 칸 가운데: 5가지',c,'차 가운데를 4번 칸 가운데에서 -0.6 ~ +0.6cm\n세 칸(3·4·5번)을 모두 막는다. 칸 하나에 5가지. CAM-1 24칸 → 120가지'))
cols=plt.cm.viridis(np.linspace(0,0.9,15))
for ax,(title,cars,note) in zip(axs,sets):
    slots_draw(ax,[6,5,4,3,2],None)
    for k,(cx,w,l) in enumerate(cars):
        ax.add_patch(Rectangle((cx-w/2,cy-l/2),w,l,fill=False,ec=cols[k%15],lw=1.6,alpha=0.9))
        ax.plot(cx,cy,'o',ms=3,color=cols[k%15])
    ax.set_xlim(T[6].x0+1.5,T[2].x1-1.5); ax.set_ylim(sl4.y1+1.5,sl4.y0-1.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title,fontsize=13,fontweight='bold'); ax.text(0.5,-0.04,note,transform=ax.transAxes,ha='center',va='top',fontsize=11)
fig.suptitle('그림 S3. 걸침 표에서 시험한 차 위치 (4번 칸 예시, 위에서 본 모습, 단위 cm). 같은 시험을 감시하는 1~14번·28~41번 줄의 칸마다 반복했다',fontsize=14,fontweight='bold')
fig.text(0.5,0.04,'정답 = 실제 차가 빈 공간(inner)에 0.1cm 이상 들어간 칸 모두.  "맞음" = 프로그램이 "차 있음"으로 낸 칸들이 정답과 정확히 같을 때(하나라도 더하거나 빠지면 틀림).\n표의 "310 / 360" = 360가지 위치 중 310가지에서 맞음. cm나 픽셀이 아니라 "경우의 수"다.',ha='center',fontsize=12)
fig.subplots_adjust(left=0.01,right=0.99,top=0.88,bottom=0.2,wspace=0.05)
fig.savefig(OUT+'39_S3_걸침시험_경우.png',dpi=100); print('S3')
