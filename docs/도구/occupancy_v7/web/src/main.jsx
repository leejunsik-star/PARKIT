import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Positions, vehicleState, connectParking, visibleVehicles,
  PARKING_FILTERS, shownByParking, initialParkingFilter, parkedHidden, approxLabel, myCarState} from './tracking.js';
import './style.css';
import AdminApp from './AdminApp.jsx';
import BlindGroupSummary from './BlindGroupSummary.jsx';
import MapViewport from './MapViewport.jsx';
import useNavigation from './useNavigation.js';
import {predictedPosition} from './navigation.js';

const LABELS = {EMPTY:'빈자리', OCCUPIED:'주차 중', DYNAMIC_BLIND:'가림 / 확인 중', STATIC_BLIND:'감시 밖'};
const ESTIMATE_LABELS = {INFERRED_EMPTY:'빈자리 추정', INFERRED_OCCUPIED:'점유 추정'};
const TRACK_LABELS = {TRACKED:'추적 중', CANDIDATE:'확인 중인 후보', UNCERTAIN:'ID 연결 불확실', LOST:'일시적으로 놓침',
  REMOVED:'추적 종료 · 차량을 다시 선택하세요', DISCONNECTED:'연결 끊김 · 마지막 위치', STALE:'업데이트 지연 · 마지막 위치', PAUSED:'분석 정지 · 마지막 위치'};
const points = coords => coords.map(p=>p.join(',')).join(' ');
const FILTER_KEY = 'parkit.parkingFilter';   // v6: 이 브라우저가 기억하는 "보일 차량" (docs/42)
const loadFilter = () => { try { return localStorage.getItem(FILTER_KEY); } catch { return null; } };
const saveFilter = v => { try { localStorage.setItem(FILTER_KEY, v); } catch { /* 저장 못 해도 화면은 그대로 */ } };

function App() {
  const [layout,setLayout]=useState(null), [snapshot,setSnapshot]=useState(null), [connected,setConnected]=useState(false);
  const [layoutError,setLayoutError]=useState(''), [selected,setSelected]=useState(null);
  const [selectedSession,setSelectedSession]=useState(null);
  const [now,setNow]=useState(performance.now());
  const [onlyMine,setOnlyMine]=useState(false);   // v4: 내 차만 보기
  const [savedFilter,setSavedFilter]=useState(loadFilter);   // v6: 보일 차량 (없으면 서버 설정)
  const [guess,setGuess]=useState(null);                     // v7: 자동으로 이어 붙인 내 차 {from, to} (docs/44)
  const [rejected,setRejected]=useState(()=>new Set());     // v7: [아니에요]로 뺀 후보
  const receipt=useRef(0), positions=useRef(new Positions()), anchor=useRef(null), session=useRef(null);
  useEffect(()=>{
    let alive=true, retry;
    const controller=new AbortController();
    async function load(){
      try {
        const response=await fetch('/api/layout', {signal:controller.signal});
        if(!response.ok) throw new Error(`HTTP ${response.status}`);
        const data=await response.json();
        if(alive){setLayout(data);setLayoutError('');}
      } catch(e) {
        if(alive){setLayoutError('주차장 정보를 기다리는 중');retry=setTimeout(load,2000);}
      }
    }
    load();
    const stop=connectParking({url:`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/parking`,
      onConnection:setConnected, onState:data=>{
        const t=performance.now();
        if(session.current!==data.session_id){positions.current=new Positions();session.current=data.session_id;}
        positions.current.update(data.vehicles,t);
        receipt.current=t;setSnapshot(data);
      }});
    let raf;
    const animate=t=>{setNow(t);raf=requestAnimationFrame(animate);};
    raf=requestAnimationFrame(animate);
    return ()=>{alive=false;controller.abort();clearTimeout(retry);stop();cancelAnimationFrame(raf);};
  },[]);
  const allVehicles=snapshot?.vehicles||[];
  const vehicle=allVehicles.find(v=>v.id===selected);
  const state=selected?vehicleState(vehicle,snapshot||{},receipt.current,now,connected):null;
  // 후보가 하나여도 사용자가 직접 확인한다. 안내 목적지는 ID 선택과 별도로 유지한다.
  const mc=myCarState(selected,snapshot,rejected);
  const candSet=new Set(['mixed','pick','auto'].includes(mc.mode)?mc.candidates:[]);
  const pickMine=id=>{setSelected(id);setSelectedSession(snapshot?.session_id);setGuess(null);setRejected(new Set());anchor.current=positions.current.at(id,now);};
  const nav=useNavigation({selected,selectedSession,snapshot,connected,elapsed:Math.max(0,(now-receipt.current)/1000)});
  const guessed=guess&&selected===guess.to;
  const parkFilter=initialParkingFilter(savedFilter,snapshot?.tracking_config?.parked_vehicles);
  const byParking=shownByParking(allVehicles,parkFilter,snapshot?.slots);
  if(nav.destination!=null&&vehicle&&!byParking.list.some(v=>v.id===selected)) {
    byParking.list.push(vehicle);byParking.hidden=byParking.hidden.filter(v=>v.id!==selected);
  }
  const slotStatus=no=>(snapshot?.slots||[]).find(s=>s.no===no)?.status;
  const mineHidden=nav.destination==null&&parkedHidden(vehicle,parkFilter,slotStatus);   // 내 차가 "보일 차량" 설정으로 숨겨짐
  const shown0=visibleVehicles(byParking.list,selected,onlyMine);
  const shown=onlyMine&&candSet.size?{list:byParking.list.filter(v=>v.id===selected||candSet.has(v.id)),mineMissing:false}:shown0;
  const mineOnly=onlyMine&&selected&&!shown.mineMissing;   // 지금 내 차만 그리는 중
  const hiddenCount=byParking.hidden.filter(v=>v.status!=='REMOVED').length;
  const filterLabel=Object.fromEntries(PARKING_FILTERS)[parkFilter];
  const chooseFilter=k=>{setSavedFilter(k);saveFilter(k);};
  const live=connected && snapshot && snapshot.publication_age_s+(now-receipt.current)/1000<2 && snapshot.engine_status==='running';
  const slots=new Map((snapshot?.slots||[]).map(s=>[s.no,s]));
  const slotDisplay = no => {
    const estimate = live ? snapshot?.blind_slot_estimates?.[no] : null;
    return (estimate !== 'POSSIBLY_EMPTY' && estimate) || slots.get(no)?.status || 'STATIC_BLIND';
  };
  const counts=Object.fromEntries(Object.keys(LABELS).map(k=>[k,(snapshot?.slots||[]).filter(s=>s.status===k).length]));
  const position=vehicle && positions.current.at(vehicle.id,now);
  if(position && (!nav.destination || nav.active)) anchor.current=position;
  const mat=layout?.mat;
  const choose=v=>pickMine(v.id);
  const carNo=id=>id.split('-').at(-1);
  const mcNotice=!selected?'':guessed?'내 차를 놓쳐서 가장 그럴듯한 차로 이어 붙였어요(추정). 왼쪽에서 맞는지 알려 주세요.'
    :mc.mode==='mixed'?'내 차가 근처 차와 엇갈렸어요. 주황색 "내 차 후보" 중 내 차를 확인하세요.'
    :mc.mode==='pick'?`내 차를 잠시 놓쳤어요. 주황색 "내 차 후보" ${mc.candidates.length}대 중 내 차를 골라 주세요.`
    :mc.mode==='lost'&&state!=='TRACKED'?'내 차를 놓쳤어요. 마지막으로 본 곳을 표시합니다. 차를 다시 눌러 선택하세요.':'';
  return <main>
    <header><div><span className="brand">PARKIT<span className="brand-dot">●</span></span><span className="header-sub">실시간 주차장</span></div>
      <span className={`connection ${live?'online':''}`} role="status">{live?'● 실시간 연결':connected?'○ 분석 대기 / 정지':'○ 연결 끊김 · 재연결 중'}</span></header>
    <div className="workspace">
      <aside>
        <div className="eyebrow">PARKING OVERVIEW</div><h1>내 차를<br/>한눈에.</h1>
        <p className="intro">지도에서 차량을 선택하고<br/>주차장 안의 이동을 확인하세요.</p>
        <div className="availability"><span>빈 주차면</span><strong>{snapshot?counts.EMPTY:'—'}<small> / {layout?.slots.length||'—'}</small></strong><span className="caption">{live?'현재 점유 판정':'마지막 수신 판정 · 현재 상태 확인 필요'}</span></div>
        <section className="selection"><div className="eyebrow">MY VEHICLE</div>
          {selected ? <><h2>차량 {selected.split('-').at(-1)}</h2><div className="tracking-status" role="status">{TRACK_LABELS[state]}{approxLabel(vehicle)?` · ${approxLabel(vehicle)}`:''}</div>
          <code className="caption">{selected}</code>
          <p className="caption">{vehicle?`마지막 관측: ${vehicle.last_seen.toFixed(1)}초 (분석 시계)`:'이 ID는 현재 목록에 없습니다.'}</p>
          {guessed&&<div className="guess-note">내 차(추정) · 다른 차와 엇갈린 뒤 이어 붙였어요. 맞나요?
            <div className="guess-btns"><button onClick={()=>{setGuess(null);setRejected(new Set());}}>맞아요</button>
            <button onClick={()=>{setRejected(r=>new Set([...r,guess.to]));setSelected(guess.from);setGuess(null);}}>아니에요</button></div></div>}
          {(['mixed','pick','auto'].includes(mc.mode))&&<div className="guess-note">{mc.mode==='mixed'?'다른 차와 엇갈려 내 차가 헷갈릴 수 있어요.':'내 차를 잠시 놓쳤어요. 근처에 내 차일 수 있는 차가 있어요.'} 내 차를 눌러 주세요.
            <div className="guess-btns">{mc.candidates.map(id=><button key={id} onClick={()=>pickMine(id)}>차량 {carNo(id)}{id===selected?' (지금 선택)':''}</button>)}</div></div>}
          {mineHidden&&<p className="caption parked-note">{vehicle.slot!=null?`${vehicle.slot}번 칸에 `:''}주차되어 지도에서 숨겼습니다 ({filterLabel}).</p>}
          <button aria-pressed={onlyMine} onClick={()=>setOnlyMine(!onlyMine)}>{onlyMine?'다른 차량도 보기':'내 차만 보기'}</button>
          <button onClick={()=>{nav.cancel();setSelected(null);setOnlyMine(false);anchor.current=null;}}>선택 해제</button></> : <><h2>차량을 선택하세요</h2><p>회색 차량 아이콘을 누르면<br/>내 차량으로 선택할 수 있어요.</p></>}
        </section>
        <section className="navigation-panel"><div className="eyebrow">주차 안내</div>
          <h2>{nav.destination!=null?`${nav.destination}번 자리로 안내`:'빈자리를 선택하세요'}</h2>
          <p role="status">{nav.message}</p>
          {nav.active&&nav.route&&<p>남은 경로 <strong>{nav.route.distance_cm.toFixed(1)} cm</strong></p>}
          {nav.destination!=null&&<button onClick={nav.cancel}>안내 취소</button>}
          <p className="caption">목적지는 예약되지 않습니다. 경로는 설정된 통로를 따르며 다른 차량·장애물 회피와 주차 조향은 제공하지 않습니다.</p>
        </section>
        <section className="parking-filter"><div className="eyebrow">보일 차량</div>
          <div className="choices" role="radiogroup" aria-label="보일 차량">
            {PARKING_FILTERS.map(([k,label])=><button key={k} role="radio" aria-checked={parkFilter===k} className={parkFilter===k?'chosen-filter':''} onClick={()=>chooseFilter(k)}>{label}</button>)}
          </div>
          <p className="caption">{parkFilter==='show'?'주차된 차도 모두 보여 줍니다.':parkFilter==='hide_occupied'?'주차가 확정된 칸의 차는 숨깁니다. 칸에 들어가는 중인 차는 확정될 때까지 보입니다.':'칸 안에 있는 차는 모두 숨기고 통로 차만 보여 줍니다.'}{hiddenCount?` 지금 ${hiddenCount}대 숨김.`:''}</p>
        </section>
        <section className="cameras"><div className="eyebrow">CAMERAS</div>{snapshot?.cameras.map(c=><div key={c.id}><span>{c.id}</span><span>{live&&c.status==='ok'&&c.valid?'관측 중':'관측 지연 / 중단'}</span></div>)}</section>
        <p className="footnote">지도 위치는 탐지 상자의 접점을 이용한 근사치입니다. 차량 ID는 위치 기반 추정이며, 교차·가림 때 연결이 끊길 수 있습니다.</p>
      </aside>
      <section className="map-panel">
        <div className="map-title"><div><span className="eyebrow">LIVE MAP</span><h2>{mineOnly?'내 차량만':'전체 주차장'}</h2></div><div className="north">↑ N <span>지도 위쪽 고정</span></div></div>
        {layout?<MapViewport mat={mat}>
          <rect width={mat.width_cm} height={mat.height_cm} fill="#edf0f2"/>
          {layout.slots.map(s=><g key={s.no} role="button" tabIndex={0} aria-label={`${s.no}번 ${ESTIMATE_LABELS[slotDisplay(s.no)] || LABELS[slotDisplay(s.no)]}`} aria-pressed={nav.destination===s.no}
            onClick={()=>{if(live)nav.choose(s.no);}} onKeyDown={e=>{if((e.key==='Enter'||e.key===' ')&&live){e.preventDefault();nav.choose(s.no);}}}
            className={nav.destination===s.no?'destination-slot':'selectable-slot'}><polygon points={points(s.cell)} className="slot-cell"/><polygon points={points(s.inner)} className={`slot ${slotDisplay(s.no)}`} opacity={live?1:.5}/><text x={s.cell.reduce((a,p)=>a+p[0],0)/s.cell.length} y={s.cell.reduce((a,p)=>a+p[1],0)/s.cell.length} className="slot-number">{s.no}</text></g>)}
          <defs><marker id="nav-arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#216bea"/></marker></defs>
          {nav.route&&<polyline markerMid="url(#nav-arrow)" markerEnd="url(#nav-arrow)" className={`navigation-route ${nav.active?'':'paused'}`} points={points(nav.route.points)} fill="none"/>}
          {nav.route&&<g className="destination-marker" transform={`translate(${nav.route.points.at(-1).join(' ')})`}><circle r="1.2"/><text y="-2">목적지 {nav.destination}</text></g>}
          {shown.list.map(v=>{
            let p=positions.current.at(v.id,now);if(!p)return null;
            if(v.id===selected&&nav.destination){
              if(!nav.active&&anchor.current)p=anchor.current;
              else if(nav.state==='ESTIMATING')p=predictedPosition(v,v.age_s+(snapshot.clock==='wall'?(snapshot.publication_age_s||0)+(now-receipt.current)/1000:0),nav.config?.prediction_s);
            }
            const st=vehicleState(v,snapshot,receipt.current,now,connected), approx=approxLabel(v), uncertain=st!=='TRACKED'||!!approx;   // v6: 위치 대략도 점선 (docs/43)
            const cand=candSet.has(v.id);   // v7: 내 차 후보 (docs/44)
            return <g key={v.id} transform={`translate(${p.x} ${p.y})`} role="button" tabIndex={0} aria-label={`차량 ${v.id.split('-').at(-1)} ${TRACK_LABELS[st]}${cand?' 내 차 후보':''}`} aria-pressed={v.id===selected} onClick={()=>choose(v)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(v);}}} className={`car ${v.id===selected?'selected':''} ${uncertain?'uncertain':''} ${cand?'candidate':''}`}>
              {uncertain&&<circle r={Math.min(6,Math.max(2.6,v.uncertainty_cm))} className="uncertainty"/>}
              <g transform={`rotate(${positions.current.headingAt(v.id,now)})`}>
              <rect x="-2.6" y="-4" width="5.2" height="8" fill="transparent"/>
              <rect className="car-body" x="-1.6" y="-3.25" width="3.2" height="6.5" rx=".7"/>
              <rect className="glass" x="-1.15" y="-1.8" width="2.3" height="1.1" rx=".25"/>
              <rect className="glass" x="-1.1" y="1.4" width="2.2" height=".8" rx=".2"/>
              <path d="M-1.1 -2.7h.6 M.5 -2.7h.6" className="lights"/>
              </g>
              <text y=".9" className="vehicle-id">{v.id.split('-').at(-1)}</text>
              {cand&&<text y="-4.4" className="cand-mark">?</text>}
              <title>{v.id} · {TRACK_LABELS[st]}{cand?' · 내 차 후보':''}{approx?` · ${approx}`:''} · {v.motion_heading == null ? '이동 방향 미확인' : '이동 방향 기준 회전 (차체 방향 아님)'}</title>
            </g>;
          })}
          {selected&&mc.mode==='lost'&&anchor.current&&<g className="last-seen" transform={`translate(${anchor.current.x} ${anchor.current.y})`}><circle r="3.2"/><text y="-3.8">마지막 위치</text></g>}
        </MapViewport>:<div className="loading">{layoutError||'주차장 정보를 불러오는 중…'}</div>}
        {snapshot?.mode==='synthetic'&&<div className="notice">합성 데모 · 실제 CCTV 탐지 결과가 아닙니다.</div>}
        {(!live||mcNotice||state==='LOST'||state==='UNCERTAIN'||state==='REMOVED')&&<div className="notice" role="status">{!live?'실시간 위치를 확인할 수 없습니다. 마지막 수신 정보를 표시합니다.':mcNotice||TRACK_LABELS[state]}</div>}
        {mineHidden?<div className="notice" role="status">내 차량이 {vehicle.slot!=null?`${vehicle.slot}번 칸에 `:''}주차되어 지도에서 숨겼습니다. 보려면 왼쪽 "보일 차량"에서 전체 차량을 고르세요.</div>
          :onlyMine&&selected&&shown.mineMissing&&<div className="notice" role="status">내 차량을 찾을 수 없어 모든 차량을 표시합니다. 내 차량을 다시 선택하세요.</div>}
        {live && <BlindGroupSummary groups={snapshot.blind_group_estimates}/>}
        <div className="legend">{Object.entries(Object.keys(snapshot?.blind_slot_estimates || {}).length ? {...LABELS,...ESTIMATE_LABELS} : LABELS).map(([k,v])=><span key={k}><i className={k}/>{v}</span>)}<span><i className="chosen"/>선택 차량</span><span><i className="approx-key"/>점선 = 위치 대략</span>{candSet.size>0&&<span><i className="cand-key"/>내 차 후보</span>}{mineOnly&&<span>다른 차량 숨김</span>}{parkFilter!=='show'&&<span>{filterLabel}{hiddenCount?` · ${hiddenCount}대`:''}</span>}</div>
        <div className="map-footer"><span>{mat?`${mat.width_cm} × ${mat.height_cm} cm`:'매트 좌표계'} · 아이콘은 이동 방향으로 회전합니다. 정지 시 마지막 방향 유지 (차체 방향 아님).</span><span>{snapshot?`갱신 ${new Date(snapshot.updated_at*1000).toLocaleTimeString('ko-KR')}`:'대기 중'}</span></div>
      </section>
    </div>
  </main>;
}
createRoot(document.getElementById('root')).render(location.pathname.replace(/\/$/, '') === '/admin' ? <AdminApp/> : <App/>);
