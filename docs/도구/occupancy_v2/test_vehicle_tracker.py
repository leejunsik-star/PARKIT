"""No cameras, models, network or slot-state changes needed."""
import itertools
import random
import unittest
from vehicle_tracker import Observation, TrackingConfig, VehicleTracker, assign


def feed(tr, t, points, camera='A', now=None):
    tr.update(camera, t, [Observation(camera, t, *p) for p in points], now)
    return list(tr.tracks.values())


class VehicleTests(unittest.TestCase):
    def tracker(self, **kw):
        return VehicleTracker(TrackingConfig(**kw), session_id='test')

    def test_single_move_confirm_and_heading(self):
        tr = self.tracker()
        for i in range(10):
            rows = feed(tr, i * .2, [(10 + i, 20)])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].id, 'test-0001')
        self.assertEqual(rows[0].status, 'TRACKED')
        self.assertIsNone(tr.snapshot(1.8)[0]['heading'])

    def test_parallel_and_detection_order(self):
        tr = self.tracker()
        for i in range(15):
            pts = [(i, 10), (i, 20), (i, 30)]
            if i % 2:
                pts.reverse()
            feed(tr, i * .2, pts)
        self.assertEqual(len(tr.tracks), 3)
        self.assertEqual([v.raw[1] for v in tr.tracks.values()], [10, 20, 30])

    def test_crossing_preserves_motion_identity(self):
        tr = self.tracker()
        for i in range(11):
            feed(tr, i * .2, [(10 + i, 20), (20 - i, 20)])
            if i == 5:
                self.assertTrue(all(v.status == 'UNCERTAIN' for v in tr.tracks.values()))
        self.assertEqual(len(tr.tracks), 2)
        self.assertGreater(tr.tracks['test-0001'].raw[0], tr.tracks['test-0002'].raw[0])

    def test_missing_lost_removed_and_no_reused_id(self):
        tr = self.tracker()
        for t in (0, .2, .4):
            feed(tr, t, [(5, 5)])
        self.assertEqual(tr.snapshot(2)[0]['status'], 'LOST')
        feed(tr, 2.1, [(5, 5)])
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(tr.snapshot(7)[0]['status'], 'REMOVED')
        feed(tr, 7.1, [(5, 5)])
        self.assertIn('test-0002', tr.tracks)

    def test_stop_restart(self):
        tr = self.tracker()
        for i, x in enumerate([0, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6]):
            feed(tr, i * .2, [(x, 10)])
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(next(iter(tr.tracks.values())).raw[0], 6)

    def test_jump_does_not_teleport_old_id(self):
        tr = self.tracker()
        for t in (0, .2, .4):
            feed(tr, t, [(5, 5)])
        feed(tr, .5, [(60, 50)])
        self.assertEqual(tr.tracks['test-0001'].raw, (5, 5))
        self.assertEqual(tr.tracks['test-0002'].status, 'CANDIDATE')

    def test_camera_handoff(self):
        tr = self.tracker()
        for i in range(6):
            feed(tr, i * .2, [(10 + i, 10)], 'A' if i < 3 else 'B')
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(set(tr.tracks['test-0001'].sources), {'A', 'B'})

    def test_simultaneous_same_car_and_distinct_hits(self):
        tr = self.tracker()
        feed(tr, 0, [(10, 10)], 'A')
        feed(tr, 0, [(10.3, 10)], 'B')
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(tr.tracks['test-0001'].hits, 1)
        for t in (.2, .4):
            feed(tr, t, [(10, 10)], 'A')
            feed(tr, t, [(10.3, 10)], 'B')
        self.assertEqual(tr.tracks['test-0001'].status, 'TRACKED')

    def test_distinct_cars_not_fused(self):
        tr = self.tracker()
        for t in (0, .2, .4):
            feed(tr, t, [(10, 10)], 'A')
            feed(tr, t, [(14, 10)], 'B')
        self.assertEqual(len(tr.tracks), 2)
        self.assertTrue(all(len(v.sources) == 1 for v in tr.tracks.values()))

    def test_ambiguous_cross_camera_not_forced(self):
        tr = self.tracker(position_noise_cm=2)
        feed(tr, 0, [(10, 10), (12, 10)])
        feed(tr, 0, [(11, 10)], 'B')
        self.assertEqual(len(tr.tracks), 2)
        self.assertTrue(all('B' not in v.sources and v.status == 'UNCERTAIN' for v in tr.tracks.values()))

    def test_timestamps_stale_repeat_and_late(self):
        tr = self.tracker()
        feed(tr, 1, [(10, 10)])
        feed(tr, 1, [(10, 10)])
        feed(tr, .9, [(10, 10)], 'B')
        feed(tr, 1.1, [(10, 10)], 'B', now=5)
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(tr.tracks['test-0001'].hits, 1)
        self.assertFalse(tr.snapshot(3)[0]['observations'][0]['valid'])

    def test_inference_latency_does_not_discard_new_frame(self):
        tr = self.tracker()
        feed(tr, 0, [(10, 10)], now=.3)
        tr.snapshot(.3)
        # Captured during inference, delivered after the previous publication.
        feed(tr, .2, [(11, 10)], now=.5)
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(tr.tracks['test-0001'].last_seen, .2)

    def test_async_motion_not_averaged_as_same_time(self):
        tr = self.tracker()
        for i in range(4):
            feed(tr, .2*i, [(10+i, 10)])
        feed(tr, .7, [(13.5, 10)], 'B')
        self.assertEqual(len(tr.tracks), 1)
        v = tr.tracks['test-0001']
        self.assertEqual(v.raw[0], 13.5)
        self.assertNotEqual(v.sources['A'].observed_at, v.sources['B'].observed_at)

    def test_invalid_unreliable_candidate_expiry(self):
        tr = self.tracker()
        tr.update('A', 0, [Observation('A', 0, float('nan'), 1), Observation('A', 0, 1, 1, reliable=False)])
        self.assertFalse(tr.tracks)
        feed(tr, .2, [(1, 1)])
        self.assertEqual(tr.snapshot(2)[0]['status'], 'REMOVED')
        self.assertEqual(tr.snapshot(4), [])

    def test_assignment_matches_bruteforce(self):
        rng = random.Random(42)
        for n in range(1, 5):
            for _ in range(10):
                a = [[rng.random()*10 for _ in range(n+1)] for _ in range(n)]
                actual = sum(a[i][j] for i, j in assign(a))
                expected = min(sum(a[i][j] for i, j in enumerate(js)) for js in itertools.permutations(range(n+1), n))
                self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
