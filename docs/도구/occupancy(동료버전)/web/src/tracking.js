// Keep connection, interpolation and selection rules independently testable.
export class Positions {
  constructor() { this.items = new Map(); }
  at(id, now) {
    const item = this.items.get(id);
    if (!item) return null;
    const u = Math.max(0, Math.min(1, (now - item.start) / 350));
    return {x: item.from.x + (item.to.x-item.from.x)*u, y: item.from.y + (item.to.y-item.from.y)*u};
  }
  update(vehicles, now) {
    const keep = new Set();
    for (const v of vehicles) {
      if (v.status === 'REMOVED') continue;
      keep.add(v.id);
      const old = this.items.get(v.id);
      if (old?.lastSeen === v.last_seen) continue; // heartbeat isn't a position update
      const to = {x: v.x_cm, y: v.y_cm};
      this.items.set(v.id, {from: this.at(v.id, now) || to, to, start: now, lastSeen: v.last_seen});
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
