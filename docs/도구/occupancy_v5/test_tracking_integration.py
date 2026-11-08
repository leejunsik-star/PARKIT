import asyncio
import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from fastapi.testclient import TestClient
from occupancy import Engine
from parking_api import create_app
from parkit_config import DEFAULT_RUN, Params, load_run, save_json
from parkit_geom import CamGeom, SlotTable, build_camera_config
from parkit_synth import cam1, pick_refs
from tracking_bridge import StateStore, TrackingBridge


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.table = SlotTable(load_run(DEFAULT_RUN)['slot_table'])

    def test_layout_and_full_snapshot_on_every_connection(self):
        store = StateStore({'mat':{'width_cm':78.3},'slots':[]})
        app = create_app(store)
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/layout').json()['mat']['width_cm'],78.3)
            for i in range(3):
                store.publish({'vehicles':[{'id':str(i)}],'slots':[], 'sequence':i})
                with client.websocket_connect('/ws/parking') as ws:
                    self.assertEqual(ws.receive_json()['sequence'],i)
                self.assertEqual(client.get('/api/state').json()['sequence'],i)

    def test_store_isolation_and_bounded_latest_snapshot(self):
        store=StateStore({})
        def writer():
            for n in range(1000):
                store.publish({'sequence':n,'vehicles':[{'id':str(n)}]})
        with concurrent.futures.ThreadPoolExecutor() as pool:
            f=pool.submit(writer)
            for _ in range(100):
                s=store.read()
                if s['vehicles']:
                    self.assertEqual(str(s['sequence']),s['vehicles'][0]['id'])
                    s['vehicles'].clear()
            f.result()
        self.assertEqual(store.read()['vehicles'],[{'id':'999'}])
        self.assertEqual(store.read()['sequence'],999)

    def test_idle_network_client_does_not_block_producer(self):
        store=StateStore({})
        with TestClient(create_app(store)) as client:
            with client.websocket_connect('/ws/parking') as slow:
                start=time.monotonic()
                for n in range(3000):
                    store.publish({'sequence':n,'vehicles':[],'slots':[]})
                self.assertLess(time.monotonic()-start,1)
                with client.websocket_connect('/ws/parking') as fresh:
                    self.assertEqual(fresh.receive_json()['sequence'],2999)

    def test_engine_results_slots_and_vehicle_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            params=Params({},self.table)
            vc=cam1()
            refs=vc.ref_clicks(self.table,pick_refs(self.table,vc),noise_px=0,seed=1)
            cal=build_camera_config('A','unused.mp4',vc.size,refs,self.table,params,50)
            path=Path(directory)
            save_json(str(path/'cam.json'),cal)
            save_json(str(path/'run.json'),{'slot_table':self.table.path,'cameras':[{'config':'cam.json'}],
                                         'log':{'dir':'logs'}})
            args=SimpleNamespace(run=str(path/'run.json'),preset=None,source=None,log_frames=False,
                                 every=None,speed=0,model=None,conf=None,quiet=True,detector='gt',tracking=True,
                                 tracking_config=None,web=False,duration=0)
            e=Engine(args,headless=True)
            try:
                c=e.cams[0];c.geom=CamGeom(cal,self.table,vc.size,e.params);c.size=vc.size
                box=vc.car_in_slot(self.table[4])+(0.95,)
                e.detector=lambda *_:[box]
                frame=np.zeros((vc.size[1],vc.size[0],3),dtype=np.uint8)
                for i in range(9):
                    e.t=i*.5;e.analyze(c,frame,i,observed_at=e.t);e.finish_tick()
                state=e.vehicle_tracking.store.read()
                self.assertEqual(len(state['vehicles']),1)
                self.assertEqual(state['vehicles'][0]['status'],'TRACKED')
                self.assertEqual(next(s for s in state['slots'] if s['no']==4)['status'],'OCCUPIED')
                self.assertEqual(state['vehicles'][0]['observations'][0]['observed_at'],4.)
                self.assertTrue(state['cameras'][0]['valid'])
                self.assertEqual(len(state['slots']),len(self.table.slots))
                self.assertEqual(len(e.vehicle_tracking.store.layout['slots']),len(self.table.slots))
            finally:
                e.detector=None;e.close()

    def test_cut_contact_not_used_and_pending_times_are_sorted(self):
        bridge=TrackingBridge(self.table)
        def box(x,cut=None):
            return SimpleNamespace(contact_mat=(x,15),conf=.9,slot=None,kind='aisle',cut_by=cut,edge=False)
        bridge.submit('B',.2,SimpleNamespace(boxes=[box(10.2)]))
        bridge.submit('A',.1,SimpleNamespace(boxes=[box(10)]))
        bridge.publish(.2,{}, {'A':'ok','B':'ok'})
        self.assertEqual(len(bridge.store.read()['vehicles']),1)
        bridge.submit('A',.3,SimpleNamespace(boxes=[box(30,0)]))
        bridge.publish(.3,{}, {'A':'ok','B':'ok'})
        self.assertEqual(len(bridge.store.read()['vehicles']),1)
        self.assertEqual(bridge.rejected_contacts,1)


if __name__=='__main__':
    unittest.main(verbosity=2)
