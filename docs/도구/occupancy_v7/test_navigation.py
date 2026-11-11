import copy
import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from navigation import Navigation, crosses
from parkit_geom import SlotTable
from parkit_config import load_run, DEFAULT_RUN
from tracking_bridge import StateStore, layout_from_table
from parking_api import create_app


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.layout=layout_from_table(SlotTable(load_run(DEFAULT_RUN)['slot_table']))
        self.nav=Navigation(self.layout)
        self.assertIsNone(self.nav.error)

    def test_route_around_slots_and_to_entry(self):
        r=self.nav.route([60,14.6],28)
        self.assertIn([3,14.6],r['points'])  # 위->아래 통로는 왼쪽 연결길을 이용
        self.assertEqual(r['points'][-1],self.nav.entry(28))
        for a,b in zip(r['points'],r['points'][1:]): self.assertTrue(self.nav.clear(a,b))
        self.assertEqual(self.nav.route([60,14.6],1)['points'][0],[60,14.6])

    def test_all_slots_reachable(self):
        for no in self.nav.slots:
            r=self.nav.route([20,14.6],no)
            self.assertTrue(r['points']);self.assertGreater(r['distance_cm'],0)

    def test_off_aisle_and_wrong_slot(self):
        with self.assertRaises(ValueError):self.nav.route([100,100],1)
        with self.assertRaises(ValueError):self.nav.route([20,14.6],999)
        # 다른 주차칸 내부에서 통로를 가로질러 연결하지 않는다.
        with self.assertRaises(ValueError):self.nav.route([63,23],28)
        r=self.nav.route([63,23],28,1)
        self.assertEqual(r['points'][1],self.nav.entry(1))

    def test_disconnected_and_oneway(self):
        c=copy.deepcopy(self.nav.config)
        c['edges']=[c['edges'][0],c['edges'][2]]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'nav.json';p.write_text(json.dumps(c));nav=Navigation(self.layout,p)
            with self.assertRaises(ValueError):nav.route([60,14.6],28)
            c['edges']=[['upper_left','upper_right','oneway']];p.write_text(json.dumps(c));nav=Navigation(self.layout,p)
            self.assertGreater(nav.route([20,14.6],1)['distance_cm'],0)
            with self.assertRaises(ValueError):nav.route([60,14.6],14)

    def test_invalid_graph_does_not_break_api(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('{}')
            self.assertFalse(Navigation(self.layout,p).info()['available'])
        with TestClient(create_app(StateStore({}))) as client:
            self.assertEqual(client.get('/api/state').status_code,200)
            self.assertFalse(client.get('/api/navigation').json()['available'])

    def test_api_requires_live_empty_confirmed_same_session(self):
        store=StateStore(self.layout)
        snap={'session_id':'test','sequence':1,'clock':'wall','engine_status':'running',
              'vehicles':[{'id':'car1','status':'TRACKED','x_cm':20,'y_cm':14.6,'age_s':0,'last_seen':1}],
              'slots':[{'no':1,'status':'EMPTY'}]}
        def call(client):
            store.publish(copy.deepcopy(snap))
            return client.get('/api/route',params={'vehicle_id':'car1','destination':1,'session_id':'test'})
        with TestClient(create_app(store)) as client:
            self.assertEqual(call(client).status_code,200)
            snap['slots'][0]['status']='OCCUPIED';self.assertEqual(call(client).status_code,409)
            snap['slots'][0]['status']='EMPTY'
            for status in ['LOST','UNCERTAIN','CANDIDATE','REMOVED']:
                snap['vehicles'][0]['status']=status;self.assertEqual(call(client).status_code,409)
            snap['vehicles'][0]['status']='TRACKED';snap['vehicles'][0]['age_s']=1
            self.assertEqual(call(client).status_code,409)
            snap['vehicles'][0]['age_s']=0;snap['session_id']='restarted'
            self.assertEqual(call(client).status_code,409)

if __name__=='__main__':unittest.main()
