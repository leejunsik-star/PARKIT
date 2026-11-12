"""사각 구역의 영속 상태와 계수선 통과 판정."""
import json
import math
import sqlite3
import threading
import time
import uuid


class ZoneCounts:
    def __init__(self, path, mat):
        self.mat = mat
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.db.execute('CREATE TABLE IF NOT EXISTS zones (id TEXT PRIMARY KEY, data TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, data TEXT)')
        self.db.commit()
        self.tracks = {}
        self.samples = {}
        self.last_crossings = {}
        self.session = None

    def _zones(self):
        return [json.loads(row[0]) for row in self.db.execute('SELECT data FROM zones ORDER BY id')]

    def snapshot(self):
        with self.lock:
            return self._zones()

    def history(self):
        with self.lock:
            rows = self.db.execute('SELECT data FROM events ORDER BY rowid DESC LIMIT 30')
            return [json.loads(row[0]) for row in rows]

    def _save(self, zone):
        zone['updated_at'] = time.time()
        self.db.execute('INSERT OR REPLACE INTO zones VALUES (?,?)', (zone['zone_id'], json.dumps(zone)))

    def _event(self, event_id, zone, count, kind, actor, vehicle=None):
        event = dict(event_id=event_id, timestamp=time.time(), zone_id=zone['zone_id'],
                     type=kind, before=zone['vehicle_count'], after=count, actor=actor, global_id=vehicle)
        self.db.execute('INSERT INTO events VALUES (?,?)', (event_id, json.dumps(event)))
        zone['vehicle_count'] = count
        self._save(zone)

    def point(self, value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError('지도 좌표 두 개가 필요합니다.')
        for x, limit in zip(value, (self.mat['width_cm'], self.mat['height_cm'])):
            if type(x) not in (int, float) or not 0 <= x <= limit:
                raise ValueError('지도 범위 안의 좌표를 선택하세요.')
        return value

    def create(self, name, polygon, gates, count):
        name = name.strip() if isinstance(name, str) else ''
        if not name or len(name) > 60 or not 3 <= len(polygon) <= 100 or not 1 <= len(gates) <= 20:
            raise ValueError('이름, 구역 꼭짓점 3개 이상, 계수선 1개 이상이 필요합니다.')
        polygon = [self.point(p) for p in polygon]
        if abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(polygon, polygon[1:]+polygon[:1]))) < 1:
            raise ValueError('구역에 면적이 필요합니다.')
        clean = []
        for number, gate in enumerate(gates, start=1):
            a,b,inside = [self.point(p) for p in gate]
            if math.dist(a,b) < 1:
                raise ValueError(f'{number}번 계수선의 양 끝을 1cm 이상 떨어뜨리세요.')
            if not contains(polygon, inside):
                raise ValueError(f'{number}번 계수선의 세 번째 점을 구역 안으로 옮기세요.')
            if abs(signed(a,b,inside)) < .8:
                raise ValueError(f'{number}번 계수선의 세 번째 점을 선에서 0.8cm 이상 떨어뜨리세요.')
            clean.append([a,b,inside])
        require_count(count)
        zone = dict(zone_id=uuid.uuid4().hex, zone_name=name, polygon=polygon, gates=clean,
                    vehicle_count=0, count_status='CONFIRMED')
        with self.lock, self.db:
            if any(z['zone_name'] == name for z in self._zones()):
                raise ValueError('같은 이름의 구역이 있습니다.')
            self._event(uuid.uuid4().hex, zone, count, 'MANUAL_SET', 'administrator')
        return zone

    def delete(self, zone_id=None):
        with self.lock, self.db:
            if zone_id is None:
                self.db.execute('DELETE FROM zones')
            else:
                self.db.execute('DELETE FROM zones WHERE id=?', (zone_id,))
            self.tracks = {k:v for k,v in self.tracks.items() if zone_id is not None and k[0] != zone_id}
            self.last_crossings = {k:v for k,v in self.last_crossings.items()
                                   if zone_id is not None and k[0] != zone_id}

    def set_blind_mode(self, zone_id, enabled):
        if type(enabled) is not bool:
            raise ValueError('구역 감시 모드를 선택하세요.')
        with self.lock, self.db:
            zone = next((z for z in self._zones() if z['zone_id'] == zone_id), None)
            if zone is None:
                raise ValueError('구역을 찾을 수 없습니다.')
            zone['force_blind'] = enabled
            self._save(zone)
            return zone

    def adjust(self, zone_id, action, request_id, value=None, target=None):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError('요청 ID가 필요합니다.')
        key = 'manual:' + request_id
        with self.lock, self.db:
            if self.db.execute('SELECT 1 FROM events WHERE id=?', (key,)).fetchone():
                return self._zones()
            zones = {z['zone_id']: z for z in self._zones()}
            if zone_id not in zones:
                raise ValueError('구역을 찾을 수 없습니다.')
            zone = zones[zone_id]
            count = zone['vehicle_count']
            if action == 'set':
                require_count(value)
                count = value
                zone['count_status'] = 'CONFIRMED'
            elif action == 'add':
                count += 1
            elif action in ('remove', 'transfer'):
                if count == 0:
                    raise ValueError('0대인 구역에서 차량을 뺄 수 없습니다.')
                count -= 1
            else:
                raise ValueError('지원하지 않는 보정입니다.')
            if action == 'transfer':
                if target == zone_id or target not in zones:
                    raise ValueError('다른 도착 구역을 선택하세요.')
                dest = zones[target]
                self._event('manual-target:'+request_id, dest, dest['vehicle_count']+1, 'MANUAL_TRANSFER', 'administrator')
            self._event(key, zone, count, 'MANUAL_'+action.upper(), 'administrator')
            # 보정 전에 시작한 횡단은 폐기하고 새 관측에서 기준면을 다시 확정한다.
            self.tracks = {k:v for k,v in self.tracks.items() if k[0] not in (zone_id,target)}
            self.last_crossings = {k:v for k,v in self.last_crossings.items() if k[0] not in (zone_id,target)}
            return self._zones()

    def observe(self, snap):
        with self.lock, self.db:
            zones = self._zones()
            if snap.get('session_id') != self.session:
                self.session = snap.get('session_id')
                self.tracks.clear()
                self.samples.clear()
                self.last_crossings.clear()
            if snap.get('engine_status') != 'running':
                return
            vehicles = {v['id']:v for v in snap.get('vehicles', []) if v['status'] != 'REMOVED'}
            fresh = {}
            for key, v in vehicles.items():
                stamp = v.get('last_seen', -1)
                if stamp > self.samples.get(key, -math.inf):
                    self.samples[key] = stamp
                    if reliable(v):
                        fresh[key] = v
            self.samples = {k:t for k,t in self.samples.items() if k in vehicles}
            self.last_crossings = {k:v for k,v in self.last_crossings.items() if k[1] in vehicles}
            for zone in zones:
                self._observe_zone(zone, vehicles, fresh)

    def _review(self, zone):
        if zone['count_status'] != 'REVIEW':
            zone['count_status'] = 'REVIEW'
            self._save(zone)

    def _observe_zone(self, zone, vehicles, fresh):
        zid = zone['zone_id']
        for key, track in list(self.tracks.items()):
            if key[0] != zid:
                continue
            vehicle = vehicles.get(key[2])
            if not reliable(vehicle):
                if track['near'] or track['pending']:
                    self._review(zone)
                del self.tracks[key]
        counted = set()
        for gid, v in fresh.items():
            p = (v['observed_x_cm'], v['observed_y_cm'])
            t = v['last_seen']
            for index, (a,b,inside) in enumerate(zone['gates']):
                d = signed(a,b,p) * math.copysign(1,signed(a,b,inside))
                near = abs(d) <= 2 and gate_projection(a,b,p)
                side = 1 if d >= .8 else -1 if d <= -.8 else 0
                key = (zid,index,gid)
                track = self.tracks.get(key)
                if track and (t-track['time'] > .8 or math.dist(track['point'],p) > 8):
                    if near or track['near'] or track['pending'] or contains(zone['polygon'],p) != contains(zone['polygon'],track['point']):
                        self._review(zone)
                    track = None
                if track is None:
                    self.tracks[key] = dict(side=side, point=p, time=t, near=near, pending=False, hits=0)
                    continue
                old = track['point']
                da, db = signed(a,b,old), signed(a,b,p)
                if da != db and da*db <= 0:
                    u = da/(da-db)
                    crossing = tuple(x+u*(y-x) for x,y in zip(old,p))
                    if gate_projection(a,b,crossing) and d*track['side'] <= 0:
                        track['pending'] = True
                if not track['side'] and side:
                    track.update(side=side, pending=False, hits=0)
                elif side and side != track['side']:
                    track['hits'] += 1
                    if track['pending'] and track['hits'] >= 2:
                        delta = side
                        inside_zone = contains(zone['polygon'], p)
                        if inside_zone != (side == 1):
                            self._review(zone)
                        elif gid not in counted and self.last_crossings.get((zid,gid)) != side:
                            count = zone['vehicle_count'] + delta
                            if count < 0:
                                self._review(zone)
                            else:
                                self._event(uuid.uuid4().hex, zone, count,
                                            'AUTO_ENTER' if delta == 1 else 'AUTO_EXIT', 'system', gid)
                            counted.add(gid)
                            self.last_crossings[zid,gid] = side
                        track.update(side=side, pending=False, hits=0)
                    elif not track['pending'] and track['hits'] >= 2:
                        if contains(zone['polygon'],p) != contains(zone['polygon'],track['point']):
                            self._review(zone)
                        track.update(side=side, hits=0)
                elif side == track['side']:
                    track.update(pending=False, hits=0)
                elif side == 0:
                    track['hits'] = 0
                track.update(point=p,time=t,near=near)


def require_count(value):
    if type(value) is not int or value < 0:
        raise ValueError('0 이상의 정수를 입력하세요.')


def reliable(v):
    return (v and v['status'] == 'TRACKED' and not v.get('approx') and not v.get('related')
            and v.get('age_s', 99) <= .8 and all(math.isfinite(v.get(k, math.nan))
            for k in ('observed_x_cm','observed_y_cm')))


def signed(a,b,p):
    return ((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])) / math.dist(a,b)


def gate_projection(a,b,p):
    return 0 <= sum((q-x)*(y-x) for x,y,q in zip(a,b,p)) <= math.dist(a,b)**2


def contains(poly,p):
    x,y = p
    return sum((a[1]>y)!=(b[1]>y) and x < a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1])
               for a,b in zip(poly,poly[1:]+poly[:1])) % 2 == 1


def slots_in_zone(layout, polygon):
    for slot in layout['slots']:
        cell = slot['inner']
        center = [sum(p[i] for p in cell)/len(cell) for i in (0, 1)]
        if contains(polygon, center):
            yield slot


def infer_blind_occupancy(layout, snapshot, zones, coverage=None):
    """구역의 잔여 대수로 감시 밖 칸의 표시만 추정한다."""
    if snapshot.get('engine_status') != 'running' or snapshot.get('publication_age_s', 0) >= 2:
        return {}, []
    active_cameras = {c['id'] for c in snapshot.get('cameras', [])
                      if c['status'] == 'ok' and c['valid']}
    if not active_cameras:
        return {}, []

    statuses = {slot['no']: slot['status'] for slot in snapshot['slots']}
    estimates = {}
    groups = []
    for zone in zones:
        polygon = zone['polygon']
        members = list(slots_in_zone(layout, polygon))
        blind = [s['no'] for s in members if statuses.get(s['no']) == 'STATIC_BLIND']
        if not blind or zone['count_status'] != 'CONFIRMED':
            continue
        if any(statuses.get(s['no']) not in ('EMPTY', 'OCCUPIED', 'STATIC_BLIND') for s in members):
            continue

        if coverage is not None and any(
            not active_cameras.intersection(coverage.get(s['no'], ()))
            for s in members if s['no'] not in blind
        ):
            continue

        # 확실히 추적 중인 통로 차량은 구역 대수에서 제외한다.
        aisle_ids = set()
        uncertain_aisle = False
        for vehicle in snapshot['vehicles']:
            if vehicle['status'] in ('LOST', 'REMOVED'):
                continue
            point = (vehicle['observed_x_cm'], vehicle['observed_y_cm'])
            if not contains(polygon, point) or any(contains(s['cell'], point) for s in members):
                continue
            if vehicle.get('kind') != 'aisle' or not reliable(vehicle) or not vehicle.get('id'):
                uncertain_aisle = True
                break
            aisle_ids.add(vehicle['id'])
        if uncertain_aisle:
            continue

        occupied = sum(statuses.get(s['no']) == 'OCCUPIED' for s in members)
        remaining = zone['vehicle_count'] - occupied - len(aisle_ids)
        if not 0 <= remaining <= len(blind):
            continue
        if remaining == 0:
            label = 'INFERRED_EMPTY'
        elif remaining == len(blind):
            label = 'INFERRED_OCCUPIED'
        else:
            groups.append({
                'zone_id': zone.get('zone_id'),
                'zone_name': zone.get('zone_name'),
                'occupied': remaining,
                'capacity': len(blind),
                'slots': blind,
            })
            label = 'POSSIBLY_EMPTY'
        for no in blind:
            estimates.setdefault(no, set()).add(label)
    resolved = {no: next(iter(labels)) for no, labels in estimates.items() if len(labels) == 1}
    groups = [group for group in groups
              if all(resolved.get(no) == 'POSSIBLY_EMPTY' for no in group['slots'])]
    return resolved, groups


def infer_blind_slots(layout, snapshot, zones, coverage=None):
    return infer_blind_occupancy(layout, snapshot, zones, coverage)[0]
