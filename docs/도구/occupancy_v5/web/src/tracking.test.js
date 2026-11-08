import test from 'node:test';
import assert from 'node:assert/strict';
import {Positions, vehicleState, followBox, connectParking, visibleVehicles} from './tracking.js';

test('interpolates, never extrapolates, heartbeat does not restart movement',()=>{
 const p=new Positions();p.update([{id:'a',x_cm:0,y_cm:0,last_seen:0}],0);
 p.update([{id:'a',x_cm:10,y_cm:4,last_seen:1}],1000);
 assert.deepEqual(p.at('a',1175),{x:5,y:2});
 p.update([{id:'a',x_cm:10,y_cm:4,last_seen:1}],1200);
 assert.deepEqual(p.at('a',1350),{x:10,y:4});
 assert.deepEqual(p.at('a',10000),{x:10,y:4});
});
test('lost selected ID never resolves to another car; removed selected ID stays missing',()=>{
 const state={clock:'wall',engine_status:'running',publication_age_s:0,tracking_config:{lost_after_s:1,remove_after_s:4}};
 const v={id:'a',status:'TRACKED',age_s:2};
 assert.equal(vehicleState(v,state,0,100,true),'LOST');
 assert.equal(vehicleState(undefined,state,0,100,true),'REMOVED');
 assert.equal(vehicleState(v,state,0,100,false),'DISCONNECTED');
 assert.equal(vehicleState(v,state,0,3000,true),'STALE');
 assert.equal(vehicleState({...v,status:'REMOVED'},state,0,100,true),'REMOVED');
 assert.equal(vehicleState(v,{...state,engine_status:'paused'},0,100,true),'PAUSED');
});
test('follow keeps the selected car at viewport center, including mat edge',()=>{
 const mat={width_cm:78.3,height_cm:55.4};
 for(const p of [{x:1,y:2},{x:40,y:30},{x:78,y:55}]){
  const [x,y,w,h]=followBox(p,mat);assert.ok(Math.abs((p.x-x)/w-.5)<1e-10);assert.ok(Math.abs((p.y-y)/h-.5)<1e-10);
 }
});
test('WebSocket close reconnects and immediately accepts complete new state; cleanup stops retry',()=>{
 const sockets=[],tasks=new Map(),connections=[],states=[];let next=0;
 class FakeWS{constructor(){sockets.push(this);}close(){this.onclose?.();}}
 const schedule=(fn,ms)=>{const id=++next;tasks.set(id,{fn,ms});return id;};
 const stop=connectParking({url:'ws://test',WebSocketImpl:FakeWS,schedule,cancel:id=>tasks.delete(id),clock:()=>0,onConnection:v=>connections.push(v),onState:v=>states.push(v)});
 sockets[0].onmessage({data:JSON.stringify({vehicles:[{id:'a'}],slots:[]})});
 assert.equal(connections.at(-1),true);sockets[0].close();assert.equal(connections.at(-1),false);
 const [id,retry]=[...tasks].find(([,t])=>t.ms===500);tasks.delete(id);retry.fn();assert.equal(sockets.length,2);
 sockets[1].onmessage({data:JSON.stringify({vehicles:[{id:'b'}],slots:[]})});assert.equal(states.at(-1).vehicles[0].id,'b');
 stop();assert.equal(tasks.size,0);
});
test('silent connection watchdog closes and schedules reconnect',()=>{
 let now=0,closed=false;const tasks=new Map();let next=0;
 class FakeWS{close(){closed=true;this.onclose?.();}}
 const stop=connectParking({url:'ws://test',WebSocketImpl:FakeWS,schedule:(fn,ms)=>{tasks.set(++next,{fn,ms});return next;},cancel:id=>tasks.delete(id),clock:()=>now,onConnection:()=>{},onState:()=>{}});
 const [id,job]=tasks.entries().next().value;tasks.delete(id);now=6000;job.fn();assert.ok(closed);assert.ok([...tasks.values()].some(x=>x.ms===500));stop();
});

test('heading uses shortest turn and preserves orientation on heartbeat or unknown heading',()=>{
 const p=new Positions();
 p.update([{id:'a',x_cm:0,y_cm:0,last_seen:0,motion_heading:350}],0);
 p.update([{id:'a',x_cm:1,y_cm:0,last_seen:1,motion_heading:10}],1000);
 assert.equal(p.headingAt('a',1175),360);
 p.update([{id:'a',x_cm:1,y_cm:0,last_seen:1,motion_heading:10}],1200);
 assert.equal(p.headingAt('a',1350),370);
 p.update([{id:'a',x_cm:1,y_cm:0,last_seen:2,motion_heading:null}],1400);
 assert.equal(p.headingAt('a',1750),10);
 p.update([{id:'a',status:'REMOVED'}],1800);
 assert.equal(p.headingAt('a',1800),0);
});

// v4 내 차만 보기
test('only-mine shows just the selected vehicle, falls back to all when it is gone',()=>{
 const vs=[{id:'s-0001',status:'TRACKED'},{id:'s-0002',status:'LOST'},{id:'s-0003',status:'REMOVED'}];
 assert.deepEqual(visibleVehicles(vs,null,false).list.map(v=>v.id),['s-0001','s-0002']);
 assert.deepEqual(visibleVehicles(vs,'s-0002',false).list.map(v=>v.id),['s-0001','s-0002']);
 assert.deepEqual(visibleVehicles(vs,null,true),{list:vs.slice(0,2),mineMissing:false});
 assert.deepEqual(visibleVehicles(vs,'s-0002',true),{list:[vs[1]],mineMissing:false});
 assert.deepEqual(visibleVehicles(vs,'s-0003',true),{list:vs.slice(0,2),mineMissing:true});
 assert.deepEqual(visibleVehicles(vs,'s-9999',true),{list:vs.slice(0,2),mineMissing:true});
 assert.deepEqual(visibleVehicles(undefined,'s-0001',true),{list:[],mineMissing:true});
});
