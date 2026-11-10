"""3~4대 혼합 시점·보정·ID 통합 합성 검사. 실제 장비 검사가 아님."""
import itertools,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from calibrate import CalibModel,camera_names
from camera_contact import contact_mode
from parkit_config import DEFAULT_RUN,load_run,Params
from parkit_geom import CamGeom,SlotTable,build_camera_config
from parkit_judge import judge_frame,PRESENT
from tracking_bridge import TrackingBridge
from vehicle_tracker import VehicleTracker,Observation
from parkit_synth import cam1,cam2,pick_refs

class MultiCameraTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=load_run(DEFAULT_RUN);cls.table=SlotTable(cls.cfg['slot_table']);cls.params=Params({},cls.table)
 def top(self,name='TOP'):
  return dict(name=name,source=0,calib_size=[800,600],contact_mode='center',homography=[[.1,0,0],[0,.1,0],[0,0,1]],monitored=list(self.table.slots),ref_points=[])
 def test_four_names_validation(self):
  self.assertTrue({'CAM-1','CAM-2','CAM-3','CAM-4'}<=set(camera_names({})))
  self.assertEqual(contact_mode({}),'ground')
  with self.assertRaises(ValueError):contact_mode({'contact_mode':'bad'})
 def test_recent_v2_calibration_selected_once(self):
  import json
  from occupancy import calibrated_cameras
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'v2').mkdir()
   for path,date,mode in [(root/'cam_CAM-1.json','2026-11-01 10:00:00','ground'),(root/'v2/cam_CAM-1.json','2026-11-10 10:00:00','center')]:
    path.write_text(json.dumps(dict(self.top(),date=date,contact_mode=mode)))
   rows=calibrated_cameras(folder)
   self.assertEqual(len(rows),1);self.assertEqual(rows[0][2]['contact_mode'],'center')
 def test_center_occupancy_no_shift(self):
  g=CamGeom(self.top(),self.table,(800,600),self.params);x,y=self.table[4].center*10
  r=judge_frame(g,self.params,[(x-6,y-10,x+6,y+10,.9)]);b=r.boxes[0]
  np.testing.assert_allclose(b.contact_px,[x,y]);np.testing.assert_allclose(b.contact_mat,self.table[4].center)
  self.assertEqual(r.obs[4],PRESENT);b.kind='aisle'
  np.testing.assert_allclose(TrackingBridge(self.table,contact_shift_cm=2).shifted_contact(b,g),b.contact_mat)
 def test_edges_overlap_rejected(self):
  g=CamGeom(self.top(),self.table,(800,600),self.params)
  for dets in [[(0,0,20,20,.9)],[(100,100,130,130,.9),(110,110,140,140,.9)]]:
   b=TrackingBridge(self.table);b.submit('P',0,judge_frame(g,self.params,dets),g);b.publish(0,{}, {'P':'ok'})
   self.assertEqual(b.store.read()['vehicles'],[])
 def test_mode_save_load(self):
  from calibrate_v2 import CalibModelV2
  vc=cam1();refs=vc.ref_clicks(self.table,pick_refs(self.table,vc),noise_px=0,seed=1)
  for cls in [CalibModel,CalibModelV2]:
   with tempfile.TemporaryDirectory() as folder:
    m=cls('PHONE',0,self.cfg,self.table,self.params,vc.size);m.path=str(Path(folder)/'phone.json');m.refs=refs;m.contact_mode='center';m.recompute()
    with patch('calibrate.save_image'),patch('calibrate_v2.save_image'),patch.object(m,'draw'):
     m.save(np.zeros((vc.size[1],vc.size[0],3),np.uint8))
    n=cls('PHONE',0,self.cfg,self.table,self.params,vc.size);n.path=m.path;n.load();n.recompute()
    self.assertEqual(n.geom.contact_mode,'center')
    self.assertEqual(cls('WEB',1,self.cfg,self.table,self.params,vc.size).contact_mode,'ground')
 def test_three_four_views_permutations(self):
  for count in [3,4]:
   for order in itertools.permutations(range(count)):
    tr=VehicleTracker(session_id='multi')
    for step in range(6):
     t=step*.2
     for k in order:tr.update(str(k),t,[Observation(str(k),t,20+step*.5+[-.7,.7,-.2,.2][k],20)])
    self.assertEqual(len(tr.tracks),1,order);self.assertEqual(tr.snapshot(1.)[0]['status'],'TRACKED')
    self.assertEqual(len(next(iter(tr.tracks.values())).sources),count)
 def test_two_cars_four_views(self):
  tr=VehicleTracker(session_id='multi')
  for step in range(8):
   for k in range(4):
    t=step*.2+k*.01;obs=[Observation(str(k),t,20+step*.5,20),Observation(str(k),t,30+step*.5,20)]
    tr.update(str(k),t,obs[::-1] if k%2 else obs)
  self.assertEqual(len(tr.tracks),2);self.assertTrue(all(len(v.sources)==4 for v in tr.tracks.values()))
 def test_four_camera_crossing_and_ambiguous_link(self):
  tr=VehicleTracker(session_id='cross')
  for step in range(11):
   t=step*.2
   for cam in ['A','B','C','D']:
    tr.update(cam,t,[Observation(cam,t,10+step,20),Observation(cam,t,20-step,20)])
  self.assertEqual(len(tr.tracks),2)
  self.assertGreater(tr.tracks['cross-0001'].raw[0],tr.tracks['cross-0002'].raw[0])
  tr=VehicleTracker(session_id='ambiguous')
  tr.update('A',0,[Observation('A',0,20,20),Observation('A',0,22,20)])
  tr.update('B',0,[Observation('B',0,21,20)])
  self.assertEqual(len(tr.tracks),2)
  self.assertTrue(all('B' not in v.sources for v in tr.tracks.values()))
 def test_dropout_stale(self):
  tr=VehicleTracker(session_id='multi')
  for step in range(12):
   for k in ([0,1,2,3] if step<4 else [2,3]):
    t=step*.2+k*.01;tr.update(str(k),t,[Observation(str(k),t,20+t,20)])
  self.assertEqual(len(tr.tracks),1);self.assertFalse(tr.update('late',0,[Observation('late',0,20,20)],now=3))
 def test_mixed_views_parked_car(self):
  bridge=TrackingBridge(self.table,contact_shift_cm=1.6,normalize_slot_centers=True);views=[]
  for k in range(2):
   g=CamGeom(self.top('P'+str(k)),self.table,(800,600),self.params);x,y=self.table[34].center*10
   views.append(('P'+str(k),g,(x-8,y-12,x+8,y+12,.9)))
  for k,vc in enumerate([cam1(),cam2()]):
   refs=vc.ref_clicks(self.table,pick_refs(self.table,vc),noise_px=0,seed=1)
   cal=build_camera_config('W'+str(k),k,vc.size,refs,self.table,self.params,50)
   views.append(('W'+str(k),CamGeom(cal,self.table,vc.size,self.params),vc.car_in_slot(self.table[34])+(.9,)))
  for step in range(6):
   for name,g,box in views:bridge.submit(name,step*.2,judge_frame(g,self.params,[box]),g)
   bridge.publish(step*.2,{}, {v[0]:'ok' for v in views})
  self.assertEqual(len(bridge.tracker.tracks),1)
  self.assertEqual(len(next(iter(bridge.tracker.tracks.values())).sources),4)

class FourCameraEngineTests(unittest.TestCase):
 setUpClass = classmethod(MultiCameraTests.setUpClass.__func__)
 top = MultiCameraTests.top
 # 기존 시나리오를 중복 실행하지 않고 엔진 통합 검사만 별도로 선언한다.
 def test_engine_four_inputs_center_modes_and_one_vehicle(self):
  from types import SimpleNamespace
  from parkit_config import save_json
  from occupancy import Engine
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);entries=[]
   for k in range(4):
    cal=self.top('CAM-'+str(k+1));cal['source']='unused.mp4'
    save_json(str(root/f'cam{k}.json'),cal);entries.append({'config':f'cam{k}.json'})
   save_json(str(root/'run.json'),{'slot_table':self.table.path,'cameras':entries,'log':{'dir':'logs'}})
   args=SimpleNamespace(run=str(root/'run.json'),preset=None,source=None,log_frames=False,every=None,speed=0,model=None,conf=None,quiet=True,detector='gt',tracking=True,tracking_config=None,web=False,duration=0)
   engine=Engine(args,headless=True)
   try:
    x,y=self.table[4].center*10;engine.detector=lambda *_:[(x-6,y-10,x+6,y+10,.9)]
    for c in engine.cams:c.geom=CamGeom(c.cal,self.table,(800,600),engine.params);c.size=(800,600)
    frame=np.zeros((600,800,3),np.uint8)
    for step in range(9):
     engine.t=step*.5
     for c in engine.cams:engine.analyze(c,frame,step,observed_at=engine.t)
     engine.finish_tick()
    state=engine.vehicle_tracking.store.read()
    self.assertEqual(len(state['cameras']),4);self.assertEqual(len(state['vehicles']),1)
    self.assertEqual(len(state['vehicles'][0]['observations']),4)
    self.assertEqual(next(s for s in state['slots'] if s['no']==4)['status'],'OCCUPIED')
   finally:engine.detector=None;engine.close()

if __name__=='__main__':unittest.main()
