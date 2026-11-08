// Keep connection, interpolation and selection rules independently testable.
// Shortest angular path avoids spinning 358 degrees when crossing north.
export function interpolateHeading(from, to, fraction) {
  return from + (((to - from + 540) % 360) - 180) * fraction;
}
export class Positions {
  constructor() { this.items = new Map(); }
  at(id, now) {
    const item = this.items.get(id);
    if (!item) return null;
    const u = Math.max(0, Math.min(1, (now - item.start) / 350));
    return {x: item.from.x + (item.to.x-item.from.x)*u, y: item.from.y + (item.to.y-item.from.y)*u};
  }
  headingAt(id, now) {
    const item = this.items.get(id);
    if (!item) return 0;
    const u = Math.max(0, Math.min(1, (now-item.start)/350));
    return interpolateHeading(item.headingFrom, item.headingTo, u);
  }
  update(vehicles, now) {
    const keep = new Set();
    for (const v of vehicles) {
      if (v.status === 'REMOVED') continue;
      keep.add(v.id);
      const old = this.items.get(v.id);
      if (old?.lastSeen === v.last_seen) continue; // heartbeat isn't a position update
      const to = {x: v.x_cm, y: v.y_cm};
      const headingFrom = old ? this.headingAt(v.id, now) : (Number.isFinite(v.motion_heading) ? v.motion_heading : 0);
      const headingTo = Number.isFinite(v.motion_heading) ? v.motion_heading : headingFrom;
      this.items.set(v.id, {headingFrom: ((headingFrom % 360)+360)%360, headingTo, from: this.at(v.id, now) || to, to, start: now, lastSeen: v.last_seen});
    }
    for (const id of this.items.keys()) if (!keep.has(id)) this.items.delete(id);
  }
}

export function vehicleState(v, snapshot, receivedAt, now, connected) {
  if (!v || v.status === 'REMOVED') return 'REMOVED';
  if (!connected) return 'DISCONNECTED';
  const elapsed = Math.max(0, (now - receivedAt)/1000);
  const publicationAge = (snapshot.publication_age_s || 0) + elapsed;
  if (publicationAge > 2) return 'STALE';
  if (snapshot.engine_status !== 'running') return 'PAUSED';
  const age = v.age_s + (snapshot.clock === 'wall' ? publicationAge : 0);
  if (age > (snapshot.tracking_config?.remove_after_s || 4)) return 'REMOVED';
  if (age > (snapshot.tracking_config?.lost_after_s || 1.2)) return 'LOST';
  return v.status;
}

// v4 "내 차만 보기": 지도에 그릴 차량 목록. 끝난(REMOVED) 차는 늘 뺀다.
// 옵션을 켜도 내 차가 목록에 없으면(추적 끝, ID 바뀜) 다시 고를 수 있게 모든 차를 돌려주고 mineMissing을 알린다.
export function visibleVehicles(vehicles, selected, onlyMine) {
  const alive = (vehicles || []).filter(v => v.status !== 'REMOVED');
  if (!onlyMine || !selected) return {list: alive, mineMissing: false};
  const mine = alive.filter(v => v.id === selected);
  return mine.length ? {list: mine, mineMissing: false} : {list: alive, mineMissing: true};
}

// v6 "보일 차량" (docs/42). 서버는 모든 차를 보내고 여기서 거른다. 통로 차(kind aisle)는 늘 보인다.
//   show          전체 차량
//   hide_occupied 주차 확정 차량 숨기기: 차가 선 칸이 OCCUPIED(판정 끝)면 숨김
//   aisle_only    주차칸 차량 모두 숨기기: 칸에 넣은 차는 판정과 상관없이 숨김
export const PARKING_FILTERS = [
  ['show', '전체 차량'],
  ['hide_occupied', '주차 확정 차량 숨기기'],
  ['aisle_only', '주차칸 차량 모두 숨기기'],
];
export function parkedHidden(v, mode, slotStatus) {
  if (!v || mode === 'show' || !mode || v.kind === 'aisle') return false;
  if (mode === 'aisle_only') return true;
  return v.slot != null && slotStatus(v.slot) === 'OCCUPIED';
}
export function shownByParking(vehicles, mode, slots) {
  const status = new Map((slots || []).map(s => [s.no, s.status]));
  const list = [], hidden = [];
  for (const v of vehicles || []) (parkedHidden(v, mode, no => status.get(no)) ? hidden : list).push(v);
  return {list, hidden};
}
// 처음 값: 이 브라우저가 기억한 값 > 서버 설정(run.json) > 전체 차량
export function initialParkingFilter(saved, serverDefault) {
  const ok = m => PARKING_FILTERS.some(([k]) => k === m);
  return ok(saved) ? saved : ok(serverDefault) ? serverDefault : 'show';
}

export function followBox(position, mat) {
  const w = mat.width_cm * .55, h = mat.height_cm * .55;
  return [position.x - w/2, position.y - h/2, w, h];
}

export function connectParking({url, onState, onConnection, WebSocketImpl=WebSocket,
                                schedule=setTimeout, cancel=clearTimeout, clock=()=>performance.now()}) {
  let socket, timer, watchdog, stopped=false, attempts=0, lastMessage=0;
  function open() {
    if (stopped) return;
    onConnection(false);
    socket = new WebSocketImpl(url);
    const current = socket;
    lastMessage = clock();
    const watch = () => {
      if (stopped || socket !== current) return;
      if (clock()-lastMessage > 5000) { current.close(); return; }
      watchdog = schedule(watch, 1000);
    };
    watchdog = schedule(watch, 1000);
    current.onmessage = event => {
      if (stopped || current !== socket) return;
      try {
        const data=JSON.parse(event.data);
        if (!Array.isArray(data.vehicles) || !Array.isArray(data.slots)) return;
        lastMessage=clock(); attempts=0;
        onState(data); onConnection(true);
      } catch { /* ignore malformed packets, watchdog still detects silence */ }
    };
    current.onerror = () => current.close();
    current.onclose = () => {
      if (stopped || current !== socket) return;
      cancel(watchdog); onConnection(false);
      timer = schedule(open, Math.min(5000, 500*2**attempts++));
    };
  }
  open();
  return () => { stopped=true; cancel(timer); cancel(watchdog); socket?.close(); };
}

// v6 위치 대략 (docs/43): 상자 아래가 다른 차에 가렸거나 화면 옆 끝에 잘려 접점을 대략으로 정한 차
export const APPROX_LABELS = {cut: '다른 차에 가려 위치 대략', edge: '화면 끝에 걸려 위치 대략'};
export function approxLabel(v) { return (v && APPROX_LABELS[v.approx]) || ''; }
