import React,{useEffect,useRef,useState} from 'react';
import {clampView,zoomView} from './navigation.js';
export default function MapViewport({mat,children}) {
  const full=[0,0,mat.width_cm,mat.height_cm];
  const [manual,setManual]=useState(full);
  const svg=useRef(null),pointers=useRef(new Map()),gesture=useRef(null),dragged=useRef(false);
  const view=manual;
  const current=useRef(view);current.current=view;
  const point=(x,y,matrix)=>new DOMPoint(x,y).matrixTransform(matrix);
  const resetGesture=()=>{
    const ps=[...pointers.current.values()];if(!ps.length){gesture.current=null;return;}
    gesture.current={ps,view:[...current.current],inverse:svg.current.getScreenCTM().inverse()};
  };
  const change=(v)=>{setManual(clampView(v,mat));};
  useEffect(()=>{
    const el=svg.current;
    const wheel=e=>{e.preventDefault();const p=point(e.clientX,e.clientY,el.getScreenCTM().inverse());
      change(zoomView(current.current,Math.exp(Math.max(-.5,Math.min(.5,e.deltaY*.002))),p,mat));};
    el.addEventListener('wheel',wheel,{passive:false});return()=>el.removeEventListener('wheel',wheel);
  },[mat]);
  const move=e=>{
    if(!pointers.current.has(e.pointerId)||!gesture.current)return;
    pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});
    const g=gesture.current,ps=[...pointers.current.values()];
    const midpoint=a=>a.length>1?{x:(a[0].x+a[1].x)/2,y:(a[0].y+a[1].y)/2}:a[0];
    const a=midpoint(g.ps),b=midpoint(ps);
    if(!dragged.current&&ps.length===1&&Math.hypot(b.x-a.x,b.y-a.y)<6)return;
    dragged.current=true;svg.current.setPointerCapture(e.pointerId);
    let ratio=1;
    if(ps.length>1&&g.ps.length>1){const dist=a=>Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y);
      ratio=dist(g.ps)/Math.max(1,dist(ps));}
    const start=point(a.x,a.y,g.inverse),end=point(b.x,b.y,g.inverse);
    const z=zoomView(g.view,ratio,start,mat),r=z[2]/g.view[2];
    change([start.x-(end.x-g.view[0])*r,start.y-(end.y-g.view[1])*r,z[2],z[3]]);
  };
  const up=e=>{pointers.current.delete(e.pointerId);if(svg.current.hasPointerCapture(e.pointerId))svg.current.releasePointerCapture(e.pointerId);resetGesture();};
  const zoom=f=>{setManual(v=>zoomView(v,f,{x:v[0]+v[2]/2,y:v[1]+v[3]/2},mat));};
  return <>
    <div className="map-controls"><button onClick={()=>zoom(.8)} aria-label="지도 확대">＋</button><button onClick={()=>zoom(1.25)} aria-label="지도 축소">−</button>
      <button onClick={()=>{setManual(full);}}>전체 보기</button><span>끌어서 이동 · 휠 / 두 손가락으로 확대</span></div>
    <svg ref={svg} className="parking-map" viewBox={view.join(' ')} preserveAspectRatio="xMidYMid meet" aria-label="차량 및 빈자리를 선택하는 주차장 지도"
      onPointerDown={e=>{if(e.button!==0&&e.pointerType==='mouse')return;if(!pointers.current.size)dragged.current=false;
        pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});resetGesture();}}
      onPointerMove={move} onPointerUp={up} onPointerCancel={e=>{dragged.current=true;up(e);}}
      onClickCapture={e=>{if(dragged.current&&e.detail!==0){e.preventDefault();e.stopPropagation();}}}>
      {children}
    </svg>
  </>;
}
