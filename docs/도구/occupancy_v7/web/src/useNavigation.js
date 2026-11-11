import {useEffect,useState,useRef} from 'react';
import {navigationState} from './navigation.js';
export default function useNavigation({selected,selectedSession,snapshot,connected,elapsed}) {
  const [destination,setDestination]=useState(null),[route,setRoute]=useState(null),[error,setError]=useState('');
  const [config,setConfig]=useState(null);
  const generation=useRef(0);
  useEffect(()=>{let stopped=false,timer;const ctrl=new AbortController();
    const load=async()=>{try{const r=await fetch('/api/navigation',{signal:ctrl.signal});if(!r.ok)throw Error();const c=await r.json();if(!stopped)setConfig(c);}
      catch{if(!stopped)timer=setTimeout(load,2000);}};load();return()=>{stopped=true;ctrl.abort();clearTimeout(timer);};},[]);
  const state=navigationState({selected,selectedSession,destination,snapshot,connected,elapsed,config:config||{}});
  const v=snapshot?.vehicles.find(v=>v.id===selected);
  const canRoute=(state==='ACTIVE'||state==='ESTIMATING')&&config?.available;
  useEffect(()=>{
    const token=++generation.current;
    if(!canRoute)return;
    const ctrl=new AbortController();
    const args=new URLSearchParams({vehicle_id:selected,destination,session_id:selectedSession});
    fetch(`/api/route?${args}`,{signal:ctrl.signal}).then(async r=>{const data=await r.json();if(!r.ok)throw Error(data.detail||'경로를 계산할 수 없습니다');return data;})
      .then(data=>{if(token===generation.current){setRoute(data);setError('');}})
      .catch(e=>{if(e.name!=='AbortError'&&token===generation.current){setRoute(null);setError(e.message);}});
    return()=>{++generation.current;ctrl.abort();};
  },[canRoute,selected,selectedSession,destination,v?.last_seen,config]);
  const choose=no=>{
    if(!selected){setError('먼저 내 차량을 선택하세요');return;}
    if(!connected||snapshot?.slots.find(s=>s.no===no)?.status!=='EMPTY'){setError('현재 빈자리로 확인된 칸을 선택하세요');return;}
    setDestination(no);setRoute(null);setError('');
  };
  const cancel=()=>{setDestination(null);setRoute(null);setError('');};
  const validRoute=route?.vehicle_id===selected&&route?.session_id===selectedSession&&route?.destination===destination?route:null;
  const arrived=canRoute&&state==='ACTIVE'&&validRoute&&validRoute.last_seen===v?.last_seen&&validRoute.distance_cm<=validRoute.arrival_cm;
  const labels={IDLE:'내 차량을 선택한 뒤 초록색 빈자리를 누르세요',ACTIVE:'안내 중',ESTIMATING:'위치 추정 중',PAUSED:'위치 확인 중 · 안내 일시정지',RESELECT:'내 차량을 다시 선택하세요 · 목적지는 유지됩니다',BLOCKED:'목적지가 빈자리로 확인되지 않습니다 · 다른 자리를 선택하세요'};
  const message=!config?'통로 설정 불러오는 중':!config.available?`통로 설정 확인 필요: ${config.error}`:(['PAUSED','RESELECT','BLOCKED'].includes(state)?labels[state]:error)|| (arrived?'주차칸 진입점 도착 · 주차 완료 판정은 아닙니다':labels[state]);
  return {destination,route:validRoute,state,message,choose,cancel,config,arrived,active:canRoute};
}
