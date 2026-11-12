import React, {useEffect, useRef, useState} from 'react';

const emptyDraft = () => ({name:'', count:'0', polygon:[], gates:[], gate:[], mode:'polygon'});
const pointList = points => points.filter(Boolean).map(p => p.join(',')).join(' ');
const statusText = zone => zone.count_status === 'CONFIRMED' ? '대수 확인됨' : '대수 확인 필요';
const eventNames = {
  AUTO_ENTER:'자동 진입', AUTO_EXIT:'자동 진출', MANUAL_ADD:'수동 추가',
  MANUAL_REMOVE:'수동 제거', MANUAL_SET:'대수 직접 설정', MANUAL_TRANSFER:'구역 이동',
};

export function useBlindZones(snapshot) {
  const zones = snapshot?.zones;
  const [draft, setDraft] = useState(null);
  const [selectedPoint, selectPoint] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const revision = zones?.map(zone => zone.updated_at).join(',');

  async function request(path, method='GET', payload) {
    const response = await fetch('/api/admin/zones'+path, {
      method, headers:{'Content-Type':'application/json'},
      body:payload === undefined ? undefined : JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '관리자 요청에 실패했습니다.');
    return data;
  }

  useEffect(() => {
    if (revision === undefined) return;
    let active = true;
    request('/events').then(data => { if (active) setHistory(data); })
      .catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [revision]);

  async function change(path, payload, done, method='POST') {
    setBusy(true);
    setError('');
    try {
      await request(path, method, payload);
      done?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!draft || !selectedPoint) return;
    const remove = event => {
      if (event.key !== 'Backspace' || event.target.closest('input, textarea, select, [contenteditable]')) return;
      event.preventDefault();
      const {kind, index, gateIndex} = selectedPoint;
      setDraft(d => {
        if (!d) return d;
        if (kind === 'gates') {
          const gates = d.gates.map((gate,i) => i === gateIndex
            ? gate.map((p,j) => j === index ? null : p) : gate);
          return {...d, gates, mode:'gate'};
        }
        return {...d, [kind]: d[kind].map((p,i) => i === index ? null : p)};
      });
      selectPoint(null);
    };
    window.addEventListener('keydown', remove);
    return () => window.removeEventListener('keydown', remove);
  }, [draft, selectedPoint]);

  function capture(event) {
    const matrix = event.currentTarget.ownerSVGElement.getScreenCTM();
    if (!matrix) return;
    const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    const p = [position.x, position.y].map(x => Number(x.toFixed(2)));
    selectPoint(null);
    setDraft(d => {
      if (!d) return d;
      const missingVertex = d.polygon.indexOf(null);
      if (missingVertex >= 0) {
        return {...d,polygon:d.polygon.map((value,i) => i === missingVertex ? p : value)};
      }
      if (d.mode === 'polygon') return {...d, polygon:[...d.polygon,p]};
      const incomplete = d.gates.findIndex(gate => gate.includes(null));
      if (incomplete >= 0) {
        const gates = d.gates.map((gate,i) => {
          if (i !== incomplete) return gate;
          const missing = gate.indexOf(null);
          return gate.map((value,j) => j === missing ? p : value);
        });
        return {...d,gates};
      }
      const missingGatePoint = d.gate.indexOf(null);
      const gate = missingGatePoint < 0 ? [...d.gate,p] :
        d.gate.map((value,i) => i === missingGatePoint ? p : value);
      return gate.length === 3 ? {...d,gates:[...d.gates,gate],gate:[]} : {...d,gate};
    });
  }

  return {zones,draft,setDraft,selectedPoint,selectPoint,history,error,busy,change,capture};
}

export function ZoneOverlay({admin, mat, connected}) {
  const drag = useRef(null);
  if (!admin.zones) return null;
  const {zones,draft,setDraft,capture,selectedPoint,selectPoint} = admin;
  function handle(p, kind, index, gateIndex) {
    if (!p) return null;
    const selected = selectedPoint?.kind === kind && selectedPoint.index === index &&
      selectedPoint.gateIndex === gateIndex;
    const move = event => {
      const start = drag.current;
      if (start?.pointerId !== event.pointerId) return;
      event.stopPropagation();
      if (Math.hypot(event.clientX-start.x, event.clientY-start.y) < 3 && !start.moved) return;
      const matrix = event.currentTarget.ownerSVGElement.getScreenCTM();
      if (!matrix) return;
      start.moved = true;
      const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      const origin = new DOMPoint(start.x,start.y).matrixTransform(matrix.inverse());
      const clamp = (value,max) => Math.max(0, Math.min(max, Number(value.toFixed(2))));
      const next = [clamp(start.point[0]+position.x-origin.x, mat.width_cm),
        clamp(start.point[1]+position.y-origin.y, mat.height_cm)];
      const replace = points => points.map((point,i) => i === index ? next : point);
      setDraft(d => {
        if (!d) return d;
        if (kind === 'gates') {
          return {...d,gates:d.gates.map((gate,i) => i === gateIndex ? replace(gate) : gate)};
        }
        return {...d,[kind]:replace(d[kind])};
      });
    };
    const finish = event => {
      if (drag.current?.pointerId !== event.pointerId) return;
      event.stopPropagation();
      drag.current = null;
    };
    return <circle key={kind+':'+gateIndex+':'+index} cx={p[0]} cy={p[1]} r={selected ? '.8' : '.6'}
      fill={selected ? '#f59e0b' : kind === 'polygon' ? '#6d28d9' : '#be185d'}
      stroke="white" strokeWidth=".18" style={{cursor:'grab'}}
      onPointerDown={event => {
        if (event.button !== 0 || drag.current) return;
        event.stopPropagation();
        selectPoint({kind,index,gateIndex});
        drag.current = {pointerId:event.pointerId,x:event.clientX,y:event.clientY,point:p,moved:false};
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={move} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish}/>;
  }
  return <>
    <g pointerEvents="none">
      {zones.map(z => {
        const [x,y] = [0,1].map(k => z.polygon.reduce((sum,p) => sum+p[k],0)/z.polygon.length);
        return <g key={z.zone_id}>
          <polygon className="blind-boundary" points={pointList(z.polygon)}/>
          {z.gates.map(([a,b],i) => <line key={i} className="blind-gate" x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}/>)}
          <text className="blind-label" x={x} y={y-1}>{z.zone_name} · {z.vehicle_count}대</text>
          <text className="blind-label" x={x} y={y+1.3}>{connected ? statusText(z) : '연결 끊김'}</text>
        </g>;
      })}
      {draft && <g>
        <polygon className="blind-boundary" points={pointList(draft.polygon)}/>
        {draft.gates.map(([a,b],i) => a && b && <line key={i} className="blind-gate" x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}/>)}
      </g>}
    </g>
    {draft && <rect width={mat.width_cm} height={mat.height_cm} fill="transparent"
      className="blind-draw-target" onClick={capture}/>}
    {draft && <g>
      {draft.polygon.map((p,i) => handle(p,'polygon',i))}
      {draft.gates.flatMap((gate,i) => gate.map((p,j) => handle(p,'gates',j,i)))}
      {draft.gate.map((p,i) => handle(p,'gate',i))}
    </g>}
  </>;
}

function draftIssue(draft) {
  if (!draft) return '';
  if (draft.polygon.length < 3) return '구역 꼭짓점을 세 개 이상 찍으세요.';
  if (draft.polygon.includes(null)) return '지운 꼭짓점의 새 위치를 찍으세요.';
  const incomplete = draft.gates.findIndex(gate => gate.includes(null));
  if (incomplete >= 0) return `${incomplete + 1}번 계수선에서 지운 점을 다시 찍으세요.`;
  if (draft.gate.length > 0) {
    return `새 계수선에 점이 ${draft.gate.length}/3개 있습니다. 남은 점을 찍거나 현재 선을 취소하세요.`;
  }
  if (!draft.gates.length) return '계수선 양 끝과 구역 안쪽 점을 순서대로 찍으세요.';
  if (!draft.name.trim()) return '구역 이름을 입력하세요.';
  if (draft.count === '' || !Number.isInteger(Number(draft.count)) || Number(draft.count) < 0) {
    return '초기 대수를 0 이상의 정수로 입력하세요.';
  }
  return '';
}

export function ZonePanel({admin}) {
  const {zones,draft,setDraft,selectPoint,history,error,busy,change} = admin;
  const [values,setValues] = useState({});
  const [targets,setTargets] = useState({});
  if (!zones || !['127.0.0.1','localhost','[::1]'].includes(location.hostname)) return null;

  const issue = draftIssue(draft);

  const removeZones = zone => {
    const message = zone ? zone.zone_name+' 구역을 삭제할까요?' : '저장한 사각지대 구역을 모두 삭제할까요?';
    if (window.confirm(message)) change(zone ? '/'+zone.zone_id : '', undefined, undefined, 'DELETE');
  };

  const adjust = (zone, action, extra={}) => {
    if (action !== 'set' && !window.confirm('자동 기록에 아직 반영되지 않은 이동인가요?')) return;
    change('/'+zone.zone_id+'/adjust', {action,request_id:crypto.randomUUID(),...extra});
  };

  return <section className="blind-admin">
    <h2>사각지대 관리자</h2>
    {!draft && error && <p role="alert">{error}</p>}
      <p>손으로 옮긴 차량은 최근 자동 통과 기록을 확인한 뒤 보정하세요.</p>
      {zones.length > 0 && <button disabled={busy} onClick={() => removeZones()}>전체 구역 삭제</button>}
      {zones.map(z => <div key={z.zone_id} className="blind-zone-controls">
        <strong>{z.zone_name} · {z.vehicle_count}대 · {statusText(z)}</strong>
        <div>
          <label><input type="checkbox" checked={Boolean(z.force_blind)} disabled={busy}
            onChange={e => change('/'+z.zone_id+'/blind-mode', {enabled:e.target.checked}, undefined, 'PUT')}/>
            구역 전체를 사각지대로 지정</label>
          <button disabled={busy} onClick={() => removeZones(z)}>구역 삭제</button>
          <button disabled={busy} onClick={() => adjust(z,'add')}>+1</button>
          <button disabled={busy || !z.vehicle_count} onClick={() => adjust(z,'remove')}>−1</button>
          <form onSubmit={e => {e.preventDefault();adjust(z,'set',{value:Number(values[z.zone_id])});}}>
            <input aria-label={z.zone_name+' 실제 대수'} type="number" min="0" step="1" required
              value={values[z.zone_id] ?? ''} onChange={e => setValues(v => ({...v,[z.zone_id]:e.target.value}))}/>
            <button disabled={busy}>대수 직접 설정</button>
          </form>
          {zones.length > 1 && <form onSubmit={e => {e.preventDefault();adjust(z,'transfer',{target:targets[z.zone_id]});}}>
            <select aria-label={z.zone_name+' 이동할 구역'} required value={targets[z.zone_id] || ''}
              onChange={e => setTargets(v => ({...v,[z.zone_id]:e.target.value}))}>
              <option value="">도착 구역</option>
              {zones.filter(other => other.zone_id !== z.zone_id).map(other =>
                <option key={other.zone_id} value={other.zone_id}>{other.zone_name}</option>)}
            </select>
            <button disabled={busy || !z.vehicle_count}>1대 이동</button>
          </form>}
        </div>
      </div>)}
      {!draft ? <button onClick={() => {selectPoint(null);setDraft(emptyDraft());}}>지도에서 구역 추가</button> :
        <form onSubmit={e => {
          e.preventDefault();
          if (issue || busy) return;
          change('',{name:draft.name,count:Number(draft.count),polygon:draft.polygon,gates:draft.gates},() => setDraft(null));
        }}>
          <label>구역 이름 <input value={draft.name} onChange={e => setDraft(d => ({...d,name:e.target.value}))} maxLength="60" required/></label>
          <label>초기 대수 <input type="number" min="0" step="1" required value={draft.count}
            onChange={e => setDraft(d => ({...d,count:e.target.value}))}/></label>
          <p>{draft.mode === 'polygon' ? '지도에서 경계 꼭짓점을 순서대로 클릭하세요.' :
            '출입구의 계수선 양 끝을 누른 다음, 진입 방향을 나타내는 구역 안쪽 점을 누르세요.'}</p>
          <button type="button" disabled={draft.polygon.length < 3 || draft.polygon.includes(null) || draft.mode === 'gate'}
            onClick={() => setDraft(d => ({...d,mode:'gate'}))}>경계 확정 · 계수선 찍기</button>
          <p>점을 끌어 옮기거나 선택한 뒤 Backspace로 지울 수 있습니다. 빠진 점은 지도에서 다시 찍으세요.</p>
          <span> 꼭짓점 {draft.polygon.length}개 · 계수선 {draft.gates.length}개 · 현재 선 {draft.gate.length}/3 </span>
          {draft.gate.length > 0 && <button type="button" onClick={() => {
            selectPoint(null);
            setDraft(d => ({...d,gate:[]}));
          }}>현재 계수선 취소</button>}
          <button type="button" onClick={() => setDraft(null)}>취소</button>
          <button disabled={busy || Boolean(issue)}>구역 저장</button>
          {issue && <p role="status">{issue}</p>}
          {error && <p role="alert">{error}</p>}
        </form>}
      <h3>최근 변경</h3>
      <ul>{history.map(event => <li key={event.event_id}>
        {new Date(event.timestamp*1000).toLocaleTimeString()} · {zones.find(z => z.zone_id === event.zone_id)?.zone_name || event.zone_id} · {eventNames[event.type]} · {event.before} → {event.after}대
        {event.global_id && ' · 차량 '+event.global_id}
      </li>)}</ul>
  </section>;
}
