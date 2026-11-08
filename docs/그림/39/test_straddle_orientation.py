exec(open('st.py').read().split("for camf in")[0])
W,L=3.2,6.5
def prep(g):
    E={}
    for sl in T:
        if sl.no not in g.slot_H: continue
        cx,cy=sl.center
        for o,(hx,hy) in (('n',(W/2,L/2)),('s',(L/2,W/2))):
            b=box(g,cx-hx,cx+hx,cy-hy,cy+hy)
            if b is None: continue
            # 차를 x로 1cm 옮겼을 때 접점 x가 얼마나 움직이는지는 1로 본다(호모그래피가 바닥 점이라)
            c=apply_h(g.slot_H[sl.no],[((b[0]+b[2])/2,b[3])])[0]
            E[(sl.no,o)]=(b,c,hx)
    return E
def estimate(g,E,b,row):
    best=None
    bw,bh=b[2]-b[0],b[3]-b[1]
    for (no,o),(eb,ec,hx) in E.items():
        if T[no].row!=row: continue
        q=apply_h(g.slot_H[no],[((b[0]+b[2])/2,b[3])])[0]
        dx=q[0]-ec[0]; dy=q[1]-ec[1]
        if abs(dx)>2.5 or abs(dy)>1.5: continue
        ew,eh=eb[2]-eb[0],eb[3]-eb[1]
        size=abs(math.log(bw/ew))+abs(math.log(bh/eh))
        cost=abs(dx)/2+abs(dy)+3*size
        if best is None or cost<best[0]: best=(cost,no,o,T[no].center[0]+dx,hx)
    return best
def run3(camf,thr=0.1):
    rc=json.load(open(D+'run.json')); params=Params(rc,None)
    cam=json.load(open(D+camf)); g=CamGeom(cam,T,(1920,1080),params); E=prep(g)
    res={'normal':[0,0],'side2':[0,0],'side3':[0,0]}; ori=[0,0]
    for row in ('1-14','28-41'):
        nos=sorted([n for n in g.monitored if T[n].row==row],key=lambda n:T[n].x0)
        for no in nos[1:-1]:
            sl=T[no]; cy=sl.center[1]; cases=[]
            for w in (2.8,3.2,3.6):
                for dx in (-1,-0.5,0,0.5,1):
                    cx=sl.center[0]+dx; cases.append(('normal',cx-w/2,cx+w/2,cy-3.25,cy+3.25))
            for dx in np.arange(-2.0,2.01,0.5):
                cx=sl.ix1+0.45+dx; cases.append(('side2',cx-3.25,cx+3.25,cy-1.6,cy+1.6))
            for dx in (-0.6,-0.3,0,0.3,0.6):
                cx=sl.center[0]+dx; cases.append(('side3',cx-3.25,cx+3.25,cy-1.6,cy+1.6))
            for kind,x0,x1,y0,y1 in cases:
                b=box(g,x0,x1,y0,y1)
                if b is None: continue
                tr=truth_slots(x0,x1,row)
                if not tr<=g.monitored: continue
                est=estimate(g,E,b,row)
                got=set()
                if est:
                    _,s,o,cx,hx=est
                    ori[0]+=1; ori[1]+= (o=='n')==(kind=='normal')
                    got={sl2.no for sl2 in T if sl2.row==row and min(cx+hx,sl2.ix1)-max(cx-hx,sl2.ix0)>=thr}
                res[kind][0]+=1; res[kind][1]+=(got==tr)
    return {k:f'{v[1]}/{v[0]}' for k,v in res.items()}, f'방향 맞춤 {ori[1]}/{ori[0]}'
for camf in ('cam_CAM-1.json','cam_CAM-2.json'): print(camf, run3(camf))
