import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Positions, vehicleState, followBox, connectParking} from './tracking.js';
import './style.css';

const LABELS = {EMPTY:'빈자리', OCCUPIED:'주차 중', DYNAMIC_BLIND:'가림 / 확인 중', STATIC_BLIND:'감시 밖'};
const TRACK_LABELS = {TRACKED:'추적 중', CANDIDATE:'확인 중인 후보', UNCERTAIN:'ID 연결 불확실', LOST:'일시적으로 놓침',
  REMOVED:'추적 종료 · 차량을 다시 선택하세요', DISCONNECTED:'연결 끊김 · 마지막 위치', STALE:'업데이트 지연 · 마지막 위치', PAUSED:'분석 정지 · 마지막 위치'};
const points = coords => coords.map(p=>p.join(',')).join(' ');

function App() {
  const [layout,setLayout]=useState(null), [snapshot,setSnapshot]=useState(null), [connected,setConnected]=useState(false);
  const [layoutError,setLayoutError]=useState(''), [selected,setSelected]=useState(null), [follow,setFollow]=useState(false);
  const [now,setNow]=useState(performance.now());
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
  const live=connected && snapshot && snapshot.publication_age_s+(now-receipt.current)/1000<2 && snapshot.engine_status==='running';
  const slots=new Map((snapshot?.slots||[]).map(s=>[s.no,s]));
  const counts=Object.fromEntries(Object.keys(LABELS).map(k=>[k,(snapshot?.slots||[]).filter(s=>s.status===k).length]));
  const position=vehicle && positions.current.at(vehicle.id,now);
  if(position) anchor.current=position;
  const mat=layout?.mat;
  const view=mat ? (follow && anchor.current ? followBox(anchor.current,mat) : [0,0,mat.width_cm,mat.height_cm]) : [0,0,1,1];
  const choose=v=>{setSelected(v.id);setFollow(false);anchor.current=positions.current.at(v.id,now);};
  return <main>
    <header><div><span className="brand">PARKIT<span className="brand-dot">●</span></span><span className="header-sub">실시간 주차장</span></div>
      <span className={`connection ${live?'online':''}`} role="status">{live?'● 실시간 연결':connected?'○ 분석 대기 / 정지':'○ 연결 끊김 · 재연결 중'}</span></header>
    <div className="workspace">
      <aside>
        <div className="eyebrow">PARKING OVERVIEW</div><h1>내 차를<br/>한눈에.</h1>
        <p className="intro">지도에서 차량을 선택하고<br/>주차장 안의 이동을 확인하세요.</p>
        <div className="availability"><span>빈 주차면</span><strong>{snapshot?counts.EMPTY:'—'}<small> / {layout?.slots.length||'—'}</small></strong><span className="caption">{live?'현재 점유 판정':'마지막 수신 판정 · 현재 상태 확인 필요'}</span></div>
        <section className="selection"><div className="eyebrow">MY VEHICLE</div>
          {selected ? <><h2>차량 {selected.split('-').at(-1)}</h2><div className="tracking-status" role="status">{TRACK_LABELS[state]}</div>
          <code className="caption">{selected}</code>
          <p className="caption">{vehicle?`마지막 관측: ${vehicle.last_seen.toFixed(1)}초 (분석 시계)`:'이 ID는 현재 목록에 없습니다.'}</p>
          <button className="primary" disabled={!position||state==='REMOVED'} onClick={()=>setFollow(!follow)}>{follow?'전체 주차장 보기':'내 차량 따라가기'}</button>
          <button onClick={()=>{setSelected(null);setFollow(false);anchor.current=null;}}>선택 해제</button></> : <><h2>차량을 선택하세요</h2><p>회색 차량 아이콘을 누르면<br/>내 차량으로 선택할 수 있어요.</p></>}
        </section>
        <section className="cameras"><div className="eyebrow">CAMERAS</div>{snapshot?.cameras.map(c=><div key={c.id}><span>{c.id}</span><span>{live&&c.status==='ok'&&c.valid?'관측 중':'관측 지연 / 중단'}</span></div>)}</section>
        <p className="footnote">지도 위치는 탐지 상자의 접점을 이용한 근사치입니다. 차량 ID는 위치 기반 추정이며, 교차·가림 때 연결이 끊길 수 있습니다.</p>
      </aside>
      <section className="map-panel">
        {snapshot?.mode==='synthetic'&&<div className="notice">합성 데모 · 실제 CCTV 탐지 결과가 아닙니다.</div>}
        <div className="map-title"><div><span className="eyebrow">LIVE MAP</span><h2>{follow?'내 차량 중심':'전체 주차장'}</h2></div><div className="north">↑ N <span>지도 위쪽 고정</span></div></div>
        {follow&&<button className="back" onClick={()=>setFollow(false)}>← 전체 주차장 보기</button>}
        {(!live||state==='LOST'||state==='UNCERTAIN'||state==='REMOVED')&&<div className="notice" role="status">{!live?'실시간 위치를 확인할 수 없습니다. 마지막 수신 정보를 표시합니다.':TRACK_LABELS[state]}</div>}
        {layout?<svg className="parking-map" viewBox={view.join(' ')} preserveAspectRatio="xMidYMid meet" aria-label="차량을 선택할 수 있는 주차장 평면도">
          <rect width={mat.width_cm} height={mat.height_cm} fill="#edf0f2"/>
          {layout.slots.map(s=><g key={s.no}><polygon points={points(s.cell)} className="slot-cell"/><polygon points={points(s.inner)} className={`slot ${slots.get(s.no)?.status||'STATIC_BLIND'}`} opacity={live?1:.5}/><text x={s.cell.reduce((a,p)=>a+p[0],0)/s.cell.length} y={s.cell.reduce((a,p)=>a+p[1],0)/s.cell.length} className="slot-number">{s.no}</text></g>)}
          {allVehicles.filter(v=>v.status!=='REMOVED').map(v=>{
            const p=positions.current.at(v.id,now);if(!p)return null;
            const st=vehicleState(v,snapshot,receipt.current,now,connected), uncertain=st!=='TRACKED';
            return <g key={v.id} transform={`translate(${p.x} ${p.y})`} role="button" tabIndex={0} aria-label={`차량 ${v.id.split('-').at(-1)} ${TRACK_LABELS[st]}`} aria-pressed={v.id===selected} onClick={()=>choose(v)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(v);}}} className={`car ${v.id===selected?'selected':''} ${uncertain?'uncertain':''}`}>
              {uncertain&&<circle r={Math.min(6,Math.max(2.6,v.uncertainty_cm))} className="uncertainty"/>}
              <rect x="-2.6" y="-4" width="5.2" height="8" fill="transparent"/>
              <rect className="car-body" x="-1.6" y="-3.25" width="3.2" height="6.5" rx=".7"/>
              <rect className="glass" x="-1.15" y="-1.8" width="2.3" height="1.1" rx=".25"/>
              <rect className="glass" x="-1.1" y="1.4" width="2.2" height=".8" rx=".2"/>
              <path d="M-1.1 -2.7h.6 M.5 -2.7h.6" className="lights"/>
              <text y=".9" className="vehicle-id">{v.id.split('-').at(-1)}</text>
              <title>{v.id} · {TRACK_LABELS[st]} · 차체 방향 미추정</title>
            </g>;
          })}
        </svg>:<div className="loading">{layoutError||'주차장 정보를 불러오는 중…'}</div>}
        <div className="legend">{Object.entries(LABELS).map(([k,v])=><span key={k}><i className={k}/>{v}</span>)}<span><i className="chosen"/>선택 차량</span></div>
        <div className="map-footer"><span>{mat?`${mat.width_cm} × ${mat.height_cm} cm`:'매트 좌표계'} · 아이콘 방향은 실제 차체 방향이 아닙니다.</span><span>{snapshot?`갱신 ${new Date(snapshot.updated_at*1000).toLocaleTimeString('ko-KR')}`:'대기 중'}</span></div>
      </section>
    </div>
  </main>;
}
createRoot(document.getElementById('root')).render(<App/>);
