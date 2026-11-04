"""Synthetic observations only: no camera, no model, no changes to calibration.
Run: python tracking_demo.py; build web/ first to serve the page on the same port.
"""
import argparse
import math
import time
from types import SimpleNamespace
from parkit_config import DEFAULT_RUN, load_run
from parkit_geom import SlotTable
from parking_api import ApiServer
from tracking_bridge import TrackingBridge


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--duration', type=float, default=0)
    args = ap.parse_args()
    table = SlotTable(load_run(args.run)['slot_table'])
    bridge = TrackingBridge(table, clock='wall', camera_ids=['CAM-1', 'CAM-2'], mode='synthetic')
    server = ApiServer(bridge.store, args.host, args.port)
    server.start()
    print('[합성 데모] 실제 탐지/점유 판정 결과가 아닙니다. Ctrl+C로 종료.')
    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            if args.duration and t >= args.duration:
                break
            x = 39 + 25 * math.sin(t*.14)
            # Two cameras overlap at x 35..44. Short periodic occlusion lets a
            # selected ID become LOST; a longer absence demonstrates REMOVED.
            hidden = 15 < t % 35 < 21
            for cam in ['CAM-1', 'CAM-2']:
                pts = []
                if not hidden and (cam == 'CAM-1' and x <= 44 or cam == 'CAM-2' and x >= 35):
                    pts.append((x, 16))
                if cam == 'CAM-1':
                    pts.append((19 + 9*math.sin(t*.1), 38))
                boxes = [SimpleNamespace(contact_mat=p, conf=.95, slot=None, kind='aisle', cut_by=None, edge=False) for p in pts]
                bridge.submit(cam, t, SimpleNamespace(boxes=boxes))
            slots = {s.no: SimpleNamespace(status='STATIC_BLIND' if s.no > 41 else 'DYNAMIC_BLIND' if s.no in (5,6) else 'OCCUPIED' if s.no in (2,4,9,30,35) else 'EMPTY', reason='SYNTHETIC_DEMO', last_known=None) for s in table}
            bridge.publish(t, slots, {'CAM-1':'ok', 'CAM-2':'ok'})
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == '__main__':
    main()
