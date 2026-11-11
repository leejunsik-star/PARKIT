"""점유/추적을 변경하지 않고 읽기만 하는 통로 기반 내비게이션."""
import heapq
import json
import math
from pathlib import Path


def distance(a, b):
    return math.dist(a, b)


def project(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return t, [a[0]+t*dx, a[1]+t*dy]


def crosses(p, q, poly):
    # cell의 외접 사각형 내부 통과를 보수적으로 금지한다. 경계 접촉은 허용.
    lo = [min(v[k] for v in poly)+1e-6 for k in (0, 1)]
    hi = [max(v[k] for v in poly)-1e-6 for k in (0, 1)]
    left, right = 0., 1.
    for k in (0, 1):
        d = q[k]-p[k]
        if abs(d)<1e-10:
            if not lo[k]<p[k]<hi[k]: return False
        else:
            a, b = sorted(((lo[k]-p[k])/d, (hi[k]-p[k])/d))
            left, right = max(left,a), min(right,b)
    return left < right and right > 0 and left < 1


class Navigation:
    def __init__(self, layout, config_path=None):
        path = Path(config_path) if config_path else Path(__file__).parent/'config/navigation_layout.json'
        self.error = None
        self.layout = layout
        self.slots = {s['no']:s for s in layout.get('slots', [])}
        self.config = {}
        try:
            self.config = json.loads(path.read_text(encoding='utf-8'))
            c = self.config
            mat = layout['mat']
            if any(abs(a-b)>.01 for a,b in zip(c['mat'], [mat['width_cm'],mat['height_cm']])):
                raise ValueError('내비 설정의 매트 크기가 도면과 다릅니다')
            nodes = c['nodes']
            for p in nodes.values():
                if len(p)!=2 or not all(math.isfinite(v) for v in p): raise ValueError('잘못된 통로 좌표')
                if not (0<=p[0]<=mat['width_cm'] and 0<=p[1]<=mat['height_cm']): raise ValueError('통로가 매트 밖입니다')
            for edge in c['edges']:
                a,b = (nodes[k] for k in edge[:2])
                if distance(a,b)<1e-6 or not self.clear(a,b): raise ValueError('통로가 주차칸을 가로지르거나 길이가 0입니다')
            if not c['edges']: raise ValueError('통로가 없습니다')
            for k in ('max_snap_cm','arrival_cm','prediction_s','reselect_s'):
                if not math.isfinite(c[k]) or c[k]<=0: raise ValueError('내비 시간/거리 설정이 잘못되었습니다')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error = str(exc)

    def clear(self, a, b, ignore=None):
        return not any(s['no']!=ignore and crosses(a,b,s['cell']) for s in self.slots.values())

    def info(self):
        return {**self.config, 'available': self.error is None, 'error':self.error}

    def entry(self, slot):
        s=self.slots[slot]
        return [sum(p[0] for p in s['inner'])/len(s['inner']), s['entrance_y']]

    def route(self, position, destination, source_slot=None):
        if self.error: raise ValueError(self.error)
        if destination not in self.slots: raise ValueError('존재하지 않는 주차칸입니다')
        target=self.entry(destination)
        original=list(position)
        start=original
        prefix=[original]
        if source_slot in self.slots and crosses(original,original,self.slots[source_slot]['cell']):
            start=self.entry(source_slot)
            if not self.clear(original,start,source_slot): raise ValueError('차량의 주차칸 진출 경로를 확인하세요')
            prefix.append(start)
        nodes={k:list(v) for k,v in self.config['nodes'].items()}
        edges=self.config['edges']
        attachments=[]
        for name,p in [('start',start),('target',target)]:
            candidates=[]
            for i,e in enumerate(edges):
                t,q=project(p,nodes[e[0]],nodes[e[1]])
                if self.clear(p,q): candidates.append((distance(p,q),i,t,q))
            if not candidates: raise ValueError('통로에 연결할 수 없습니다')
            d,i,t,q=min(candidates)
            if d>self.config['max_snap_cm']: raise ValueError('차량 또는 목적지가 설정된 통로에서 너무 멉니다')
            nodes[name]=q
            attachments.append((i,t,name))
        graph={k:[] for k in nodes}
        for i,e in enumerate(edges):
            chain=sorted([(0.,e[0]),(1.,e[1])]+[(t,k) for j,t,k in attachments if j==i])
            for (_,a),(_,b) in zip(chain,chain[1:]):
                w=distance(nodes[a],nodes[b]);graph[a].append((b,w))
                if len(e)<3 or e[2]!='oneway': graph[b].append((a,w))
        queue=[(0.,'start')];cost={'start':0.};prev={}
        while queue:
            d,a=heapq.heappop(queue)
            if d!=cost[a]: continue
            if a=='target': break
            for b,w in graph[a]:
                if d+w<cost.get(b,float('inf')):
                    cost[b]=d+w;prev[b]=a;heapq.heappush(queue,(d+w,b))
        if 'target' not in cost: raise ValueError('목적지로 연결된 통로가 없습니다')
        chain=['target']
        while chain[-1]!='start': chain.append(prev[chain[-1]])
        path=prefix+[nodes[k] for k in reversed(chain)]+[target]
        clean=[]
        for p in path:
            if not clean or distance(clean[-1],p)>1e-6: clean.append(p)
        return {'points':clean,'distance_cm':sum(distance(a,b) for a,b in zip(clean,clean[1:])),
                'destination':destination,'arrival_cm':self.config['arrival_cm']}
