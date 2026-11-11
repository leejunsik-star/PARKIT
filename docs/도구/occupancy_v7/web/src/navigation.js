// 안내 상태는 차량 ID와 목적지를 분리해 관리한다. 새 ID 선택은 사용자가 한다.
export function navigationState({selected, selectedSession, destination, snapshot, connected, elapsed=0, config={}}) {
  if(destination==null) return 'IDLE';
  if(!selected || selectedSession!==snapshot?.session_id) return 'RESELECT';
  if(!connected || !snapshot || snapshot.engine_status!=='running' || (snapshot.publication_age_s||0)+elapsed>2) return 'PAUSED';
  const v=snapshot.vehicles.find(v=>v.id===selected);
  if(!v || v.status==='REMOVED') return 'RESELECT';
  const age=v.age_s+(snapshot.clock==='wall'?(snapshot.publication_age_s||0)+elapsed:0);
  if(age>=(config.reselect_s||3)) return 'RESELECT';
  if(v.status!=='TRACKED'||v.approx||(v.related||[]).length||age>(config.prediction_s||.5)) return 'PAUSED';
  if(snapshot.slots.find(s=>s.no===destination)?.status!=='EMPTY') return 'BLOCKED';
  return age>.15?'ESTIMATING':'ACTIVE';
}
export function zoomView(view, factor, point, mat) {
  const width=Math.max(mat.width_cm*.05,Math.min(mat.width_cm,view[2]*factor));
  const ratio=width/view[2];
  return clampView([point.x-(point.x-view[0])*ratio,point.y-(point.y-view[1])*ratio,width,view[3]*ratio],mat);
}
export function clampView(v,mat) {
  return [Math.max(-v[2]*.2,Math.min(mat.width_cm-v[2]*.8,v[0])),
    Math.max(-v[3]*.2,Math.min(mat.height_cm-v[3]*.8,v[1])),v[2],v[3]];
}
export function predictedPosition(v, age, horizon=.5) {
  const dt=Math.max(0,Math.min(age,horizon));
  return {x:v.x_cm+(v.velocity_cm_s?.[0]||0)*dt,y:v.y_cm+(v.velocity_cm_s?.[1]||0)*dt};
}
