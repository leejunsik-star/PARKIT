import React, {useEffect, useState} from 'react';
import {connectParking} from './tracking.js';
import BlindGroupSummary from './BlindGroupSummary.jsx';
import MapViewport from './MapViewport.jsx';
import {useBlindZones, ZoneOverlay, ZonePanel} from './BlindZones.jsx';

const labels = {
  EMPTY:'빈자리', OCCUPIED:'주차 중', DYNAMIC_BLIND:'가림 / 확인 중',
  STATIC_BLIND:'감시 밖', INFERRED_EMPTY:'빈자리 추정', INFERRED_OCCUPIED:'점유 추정',
};
const points = polygon => polygon.map(point => point.join(',')).join(' ');

export default function AdminApp() {
  const [layout, setLayout] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [connected, setConnected] = useState(false);
  const [layoutError, setLayoutError] = useState('');
  const admin = useBlindZones(snapshot);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/layout', {signal:controller.signal})
      .then(response => {
        if (!response.ok) throw new Error('주차장 도면을 불러올 수 없습니다.');
        return response.json();
      })
      .then(setLayout)
      .catch(error => { if (error.name !== 'AbortError') setLayoutError(error.message); });
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const stop = connectParking({url:protocol+'//'+location.host+'/ws/parking',
      onConnection:setConnected, onState:setSnapshot});
    return () => { controller.abort(); stop(); };
  }, []);

  const statuses = new Map(snapshot?.slots?.map(slot => [slot.no,slot.status]));
  const mat = layout?.mat;
  const live = connected && snapshot?.engine_status === 'running' && snapshot.publication_age_s < 2;
  return <main>
    <header>
      <span className="brand">PARKIT 관리자</span>
      <span role="status" className={live ? 'connection online' : 'connection'}>
        {live ? '실시간 연결' : '판정 연결 대기'}
      </span>
    </header>
    <section className="map-panel admin-map-panel">
      <h1>주차칸과 사각지대 관리</h1>
      {layoutError && <p role="alert">{layoutError}</p>}
      {mat && <MapViewport mat={mat}>
        <rect width={mat.width_cm} height={mat.height_cm} fill="#edf0f2"/>
        {layout.slots.map(slot => {
          const estimate = live && snapshot.blind_slot_estimates?.[slot.no];
          const status = (estimate !== 'POSSIBLY_EMPTY' && estimate) ||
            statuses.get(slot.no) || 'STATIC_BLIND';
          const [x,y] = [0,1].map(axis => slot.inner.reduce((sum,p) => sum+p[axis],0)/slot.inner.length);
          return <g key={slot.no}>
            <title>{slot.no}번 · {labels[status]}</title>
            <polygon points={points(slot.cell)} className="slot-cell"/>
            <polygon points={points(slot.inner)} className={'slot '+status}/>
            <text x={x} y={y} className="slot-number">{slot.no}</text>
          </g>;
        })}
        <ZoneOverlay admin={admin} mat={mat} connected={Boolean(live)}/>
      </MapViewport>}
      {live && <BlindGroupSummary groups={snapshot.blind_group_estimates}/>}
      <div className="legend">{Object.entries(labels).map(([status,label]) =>
        <span key={status}><i className={status}/>{label}</span>)}</div>
      {mat && <ZonePanel admin={admin}/>}
    </section>
  </main>;
}
