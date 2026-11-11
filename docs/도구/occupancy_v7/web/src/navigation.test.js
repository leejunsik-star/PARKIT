import {test} from 'node:test';
import assert from 'node:assert/strict';
import {navigationState,zoomView,clampView,predictedPosition} from './navigation.js';
const base=()=>({selected:'A',selectedSession:'s',destination:1,connected:true,elapsed:0,
  snapshot:{session_id:'s',engine_status:'running',clock:'wall',publication_age_s:0,vehicles:[{id:'A',status:'TRACKED',age_s:0}],slots:[{no:1,status:'EMPTY'}]}});
test('안내/짧은 누락/일시정지/재선택; 목적지 유지',()=>{
 const a=base();assert.equal(navigationState(a),'ACTIVE');
 a.elapsed=.2;assert.equal(navigationState(a),'ESTIMATING');
 a.elapsed=.7;assert.equal(navigationState(a),'PAUSED');
 a.elapsed=0;a.snapshot.vehicles[0].age_s=3.1;assert.equal(navigationState(a),'RESELECT');assert.equal(a.destination,1);
 a.snapshot.vehicles=[{id:'B',status:'TRACKED',age_s:0}];assert.equal(navigationState(a),'RESELECT');assert.equal(a.selected,'A');
 a.selected='B';assert.equal(navigationState(a),'ACTIVE');
});
test('연결 중단·재시작·불확실성·목적지 변경',()=>{
 const a=base();a.connected=false;assert.equal(navigationState(a),'PAUSED');a.connected=true;
 a.snapshot.session_id='new';assert.equal(navigationState(a),'RESELECT');a.selectedSession='new';
 a.snapshot.vehicles[0].related=['B'];assert.equal(navigationState(a),'PAUSED');a.snapshot.vehicles[0].related=[];
 a.snapshot.slots[0].status='DYNAMIC_BLIND';assert.equal(navigationState(a),'BLOCKED');
 a.snapshot.slots[0].status='OCCUPIED';assert.equal(navigationState(a),'BLOCKED');
 a.destination=null;assert.equal(navigationState(a),'IDLE');
});
test('예측은 0.5초로 제한',()=>{assert.deepEqual(predictedPosition({x_cm:10,y_cm:20,velocity_cm_s:[4,-2]},5),{x:12,y:19});});
test('지도 확대 중심·최소/최대 배율·지도 밖 이동 제한',()=>{
 const mat={width_cm:100,height_cm:60};assert.deepEqual(zoomView([0,0,100,60],.5,{x:50,y:30},mat),[25,15,50,30]);
 assert.equal(zoomView([0,0,100,60],.0001,{x:50,y:30},mat)[2],5);
 assert.equal(zoomView([0,0,100,60],10,{x:50,y:30},mat)[2],100);
 assert.deepEqual(clampView([999,-999,50,30],mat),[60,-6,50,30]);
});
