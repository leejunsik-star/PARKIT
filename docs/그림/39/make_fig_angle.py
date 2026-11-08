import os  # 그림 S4: +20°/-20° 차의 YOLO 상자 비교
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'make_fig_base.py')).read().split("if __name__=='__main__':")[0])
fig,axs=plt.subplots(2,2,figsize=(17,11))
for j,camname in enumerate(('CAM-1','CAM-2')):
    g,cam=G[camname]; sl=T[9]
    cars=[dict(cx=sl.center[0],cy=sl.center[1],ang=20,color='#d62728'),dict(cx=sl.center[0],cy=sl.center[1],ang=-20,color='#1f77b4')]
    ax=axs[0,j]; topview(ax,g,camname,None,[])
    for c in cars: ax.add_patch(Polygon(corners(c),closed=True,fc=c['color'],alpha=0.45,ec='k'))
    ax.set_xlim(sl.x0-9,sl.x1+9); ax.set_ylim(sl.y1+3,sl.y0-3)
    ax.set_title(f'위에서 본 모습: 9번에 +20°(빨강) / -20°(파랑)로 선 차',fontsize=12)
    ax=axs[1,j]; camview(ax,g,None,[])
    bs=[]
    for c in cars:
        uv,_=project(g.K,g.rvec,g.tvec,car_pts3(c))
        ax.add_patch(Polygon(hull(uv),closed=True,fc=c['color'],alpha=0.35,ec=c['color'],lw=1.5))
        b=car_bbox(g,c); bs.append(b)
        ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec=c['color'],lw=2.5,ls='--'))
    b=np.array(bs); x0,y0=b[:,:2].min(0); x1,y1=b[:,2:].max(0); cx,cy=(x0+x1)/2,(y0+y1)/2; w=(x1-x0)*2.6; h=w/1.6
    ax.set_xlim(cx-w/2,cx+w/2); ax.set_ylim(cy+h/2,cy-h/2)
    d=np.abs(b[0]-b[1]); ax.set_title(f'{camname} 화면: 두 차의 YOLO 상자(점선) 차이 왼 {d[0]:.0f} / 위 {d[1]:.0f} / 오른 {d[2]:.0f} / 아래 {d[3]:.0f} 픽셀',fontsize=12)
fig.suptitle('그림 S4. 비스듬히 선 차의 어려움: 반대로 같은 각도만큼 돌린 두 차가 화면에서는 거의 같은 상자가 된다',fontsize=15,fontweight='bold')
fig.text(0.5,0.03,'YOLO 상자는 숫자 4개(왼·위·오른·아래)뿐이라 차의 위치·각도·크기를 다 알 수 없다. 특히 줄을 따라 낮게 보는 CAM-1에서 두 상자가 거의 같다.\n그래서 각도 찾기로도 구별이 안 되는 경우는 "모름"으로 두거나, 다른 카메라나 다른 탐지 방식(외곽선)의 도움이 필요하다.',ha='center',fontsize=12)
fig.subplots_adjust(left=0.01,right=0.99,top=0.92,bottom=0.09,wspace=0.04,hspace=0.12)
fig.savefig(OUT+'39_S4_비스듬히선차.png',dpi=95); print('ok')
