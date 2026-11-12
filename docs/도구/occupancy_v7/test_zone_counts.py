import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from occupancy2 import ZoneStore
from parking_api import create_app
from tracking_bridge import StateStore
from zone_counts import ZoneCounts, infer_blind_occupancy, infer_blind_slots


class ZoneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'counts.db'
        self.mat = dict(width_cm=78.3, height_cm=55.4)
        self.counts = ZoneCounts(self.path, self.mat)
        self.addCleanup(self.counts.db.close)
        self.a = self.counts.create('A', [[10,10],[20,10],[20,20],[10,20]],
                                    [[[10,10],[10,20],[13,15]]], 3)['zone_id']
        self.b = self.counts.create('B', [[30,10],[40,10],[40,20],[30,20]],
                                    [[[30,10],[30,20],[33,15]]], 2)['zone_id']
        self.clock = 0

    def zone(self, zid=None):
        return next(z for z in self.counts.snapshot() if z['zone_id'] == (zid or self.a))

    def step(self, x=None, *, status='TRACKED', y=15, dt=.1, session='test'):
        self.clock += dt
        vehicles = [] if x is None else [dict(id='car', status=status, age_s=0, last_seen=self.clock,
                                              observed_x_cm=x, observed_y_cm=y)]
        snap = dict(session_id=session, engine_status='running', vehicles=vehicles)
        self.counts.observe(snap)
        self.counts.observe(snap)  # 반복 전송된 관측은 한 번만 처리
        return snap

    def test_manual_transfer_atomic_and_persistent(self):
        self.counts.adjust(self.a, 'remove', 'remove')
        self.counts.adjust(self.a, 'add', 'add')
        self.counts.adjust(self.a, 'transfer', 'move', target=self.b)
        self.counts.adjust(self.a, 'transfer', 'move', target=self.b)
        self.assertEqual((self.zone()['vehicle_count'], self.zone(self.b)['vehicle_count']), (2,3))
        self.counts.adjust(self.a, 'set', 'set', value=0)
        with self.assertRaises(ValueError):
            self.counts.adjust(self.a, 'transfer', 'bad', target=self.b)
        other = ZoneCounts(self.path, self.mat)
        try:
            self.assertEqual(other.snapshot(), self.counts.snapshot())
            self.assertEqual(other.history(), self.counts.history())
        finally:
            other.db.close()

    def test_auto_and_manual_share_count(self):
        for x in [8,9.7,10.2,11,12,13]:
            self.step(x)
        self.assertEqual(self.zone()['vehicle_count'], 4)
        self.counts.adjust(self.a, 'remove', 'hand')
        for x in [12,11,10.3,9,8,7]:
            self.step(x)
        self.assertEqual(self.zone()['vehicle_count'], 2)
        self.assertEqual([e['type'] for e in self.counts.history()[:3]],
                         ['AUTO_EXIT','MANUAL_REMOVE','AUTO_ENTER'])

    def test_reverse_and_loss_near_line(self):
        for x in [8,9.8,10.1,9.7,8]:
            self.step(x)
        self.assertEqual(self.zone()['vehicle_count'], 3)
        self.step(9.9)
        self.step()
        self.assertEqual(self.zone()['count_status'], 'REVIEW')
        self.counts.adjust(self.a, 'set', 'confirm', value=3)
        self.assertEqual(self.zone()['count_status'], 'CONFIRMED')

    def test_stale_jump_and_session_reset(self):
        self.step(8)
        self.step(15, dt=2)
        self.step(16)
        self.assertEqual(self.zone()['vehicle_count'], 3)
        self.assertEqual(self.zone()['count_status'], 'REVIEW')
        self.step(8, session='new')
        self.step(7, session='new')
        self.assertEqual(self.zone()['vehicle_count'], 3)

    def test_manual_discards_crossing_in_progress(self):
        self.step(8)
        self.step(10.3)
        self.counts.adjust(self.a, 'add', 'hand')
        self.step(11)
        self.step(12)
        self.assertEqual(self.zone()['vehicle_count'], 4)
        self.assertFalse(any(e['type'] == 'AUTO_ENTER' for e in self.counts.history()))

    def test_concurrent_manual_updates(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda i: self.counts.adjust(self.a,'add',str(i)), range(20)))
        self.assertEqual(self.zone()['vehicle_count'], 23)

    def test_admin_can_mask_camera_empty_slots_in_covered_zone(self):
        cells = [[[x,11],[x+1,11],[x+1,12],[x,12]] for x in range(11,17)]
        layout = {'mat':self.mat, 'slots':[dict(no=no, inner=cell, cell=cell)
                                           for no, cell in zip(range(21,27), cells)]}
        base = StateStore(layout)
        base.publish(dict(engine_status='running', cameras=[dict(id='CAM-3',status='ok',valid=True)],
                          vehicles=[], slots=[dict(no=no,status='EMPTY') for no in range(21,27)]))
        store = ZoneStore(base,self.counts)
        self.counts.adjust(self.a,'set','five',value=5)
        with TestClient(create_app(store),client=('127.0.0.1',50000),base_url='http://127.0.0.1') as client:
            url = '/api/admin/zones/'+self.a+'/blind-mode'
            self.assertEqual(client.put(url,json={'enabled':True}).status_code,200)
            state = client.get('/api/state').json()
            self.assertEqual({s['status'] for s in state['slots']}, {'STATIC_BLIND'})
            group = state['blind_group_estimates'][0]
            self.assertEqual((group['occupied'],group['capacity']), (5,6))
            self.assertEqual(client.put(url,json={'enabled':False}).status_code,200)
            self.assertEqual({s['status'] for s in client.get('/api/state').json()['slots']}, {'EMPTY'})

    def test_admin_delete_one_and_all_preserves_history(self):
        store = ZoneStore(StateStore({'mat':self.mat,'slots':[]}), self.counts)
        with TestClient(create_app(store), client=('127.0.0.1',50000),base_url='http://127.0.0.1') as client:
            url = '/api/admin/zones'
            self.assertEqual(client.delete(url, headers={'Origin':'https://example.com'}).status_code, 403)
            self.assertEqual(client.delete(url+'/'+self.a).status_code, 200)
            self.assertEqual([z['zone_id'] for z in client.get('/api/state').json()['zones']], [self.b])
            self.assertEqual(client.delete(url).status_code, 200)
            self.assertEqual(client.get('/api/state').json()['zones'], [])
            self.assertEqual(len(self.counts.history()), 2)

    def test_api_auth_and_websocket_refresh(self):
        base = StateStore({'mat':self.mat,'slots':[]})
        store = ZoneStore(base, self.counts)
        with TestClient(create_app(base)) as client:
            self.assertNotIn('zones', client.get('/api/state').json())
            self.assertEqual(client.get('/api/admin/zones/events').status_code,404)
        with TestClient(create_app(store),client=('192.168.0.12',50000)) as client:
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(client.get('/admin').status_code,403)
            self.assertEqual(client.get('/api/admin/zones/events').status_code,403)
        with TestClient(create_app(store),client=('127.0.0.1',50000),base_url='http://127.0.0.1') as client:
            url = '/api/admin/zones/'+self.a+'/adjust'
            payload = {'action':'remove','request_id':'request'}
            self.assertEqual(client.get('/admin').status_code,200)
            self.assertEqual(client.get('/').status_code,200)
            with client.websocket_connect('/ws/parking') as ws:
                ws.receive_json()
                self.assertEqual(client.post(url,json=payload).status_code,200)
                updated = ws.receive_json()
                self.assertEqual(next(z['vehicle_count'] for z in updated['zones'] if z['zone_id']==self.a),2)
            with client.websocket_connect('/ws/parking') as ws:
                self.assertEqual(next(z['vehicle_count'] for z in ws.receive_json()['zones'] if z['zone_id']==self.a),2)


class BlindSlotInferenceTests(unittest.TestCase):
    def test_two_cars_in_six_blind_slots_have_group_estimate(self):
        cells = [[[x,0],[x+1,0],[x+1,1],[x,1]] for x in range(6)]
        layout = {'slots': [dict(no=no, inner=cell, cell=cell)
                            for no, cell in zip(range(21,27), cells)]}
        zone = dict(zone_id='a', zone_name='A', polygon=[[0,0],[6,0],[6,3],[0,3]],
                    vehicle_count=2, count_status='CONFIRMED')
        snapshot = dict(engine_status='running', cameras=[dict(id='CAM-3',status='ok',valid=True)],
                        vehicles=[], slots=[dict(no=no,status='STATIC_BLIND') for no in range(21,27)])
        _, groups = infer_blind_occupancy(layout, snapshot, [zone], coverage={})
        self.assertEqual(groups, [dict(zone_id='a', zone_name='A', occupied=2, capacity=6,
                                       slots=list(range(21,27)))])
        zone['vehicle_count'] = 3
        snapshot['vehicles'] = [dict(id='aisle-1',status='TRACKED',kind='aisle',age_s=.1,
                                     observed_x_cm=3,observed_y_cm=2)]
        _, groups = infer_blind_occupancy(layout, snapshot, [zone], coverage={})
        self.assertEqual((groups[0]['occupied'],groups[0]['capacity']), (2,6))
        snapshot['vehicles'] = []
        zone['vehicle_count'] = 6
        estimates, groups = infer_blind_occupancy(layout, snapshot, [zone], coverage={})
        self.assertEqual(estimates, dict.fromkeys(range(21,27), 'INFERRED_OCCUPIED'))
        self.assertEqual(groups, [])

    def test_residual_count_and_uncertainty(self):
        slots = []
        for no in range(50,57):
            x = no - 50
            cell = [[x,0],[x+1,0],[x+1,1],[x,1]]
            slots.append(dict(no=no,inner=cell,cell=cell))
        zone = dict(polygon=[[0,0],[7,0],[7,3],[0,3]], vehicle_count=6, count_status='CONFIRMED')
        snapshot = dict(engine_status='running', cameras=[dict(id='CAM-3',status='ok',valid=True)], vehicles=[],
                        slots=[dict(no=no,status='OCCUPIED') for no in range(50,57)])
        def result():
            return infer_blind_slots(dict(slots=slots), snapshot, [zone])

        snapshot['slots'][5]['status'] = 'STATIC_BLIND'
        snapshot['slots'][6]['status'] = 'EMPTY'
        self.assertEqual(result(), {55:'INFERRED_OCCUPIED'})
        snapshot['cameras'].append(dict(id='CAM-1', status='offline', valid=False))
        coverage = {no: ['CAM-3'] for no in range(50,57) if no != 55}
        self.assertEqual(infer_blind_slots(dict(slots=slots), snapshot, [zone], coverage),
                         {55:'INFERRED_OCCUPIED'})
        coverage[50] = ['CAM-1']
        self.assertEqual(infer_blind_slots(dict(slots=slots), snapshot, [zone], coverage), {})
        snapshot['slots'][4]['status'] = 'STATIC_BLIND'
        zone['vehicle_count'] = 5
        self.assertEqual(result(), {54:'POSSIBLY_EMPTY',55:'POSSIBLY_EMPTY'})
        zone['vehicle_count'] = 6
        self.assertEqual(result(), {54:'INFERRED_OCCUPIED',55:'INFERRED_OCCUPIED'})
        snapshot['vehicles'] = [dict(status='TRACKED',observed_x_cm=4,observed_y_cm=2)]
        self.assertEqual(result(), {})
        snapshot['vehicles'] = []
        zone['count_status'] = 'REVIEW'
        self.assertEqual(result(), {})
        self.assertEqual(snapshot['slots'][5]['status'], 'STATIC_BLIND')

        zone.update(vehicle_count=0, count_status='CONFIRMED')
        for slot in snapshot['slots']:
            if slot['status'] != 'STATIC_BLIND':
                slot['status'] = 'EMPTY'
        self.assertEqual(result(), {54:'INFERRED_EMPTY',55:'INFERRED_EMPTY'})


if __name__ == '__main__':
    unittest.main()
