"""기존 점유 엔진에 사각지대 집계를 추가하는 선택 실행 파일."""
import os
from pathlib import Path
import sys

import occupancy
from zone_counts import ZoneCounts, infer_blind_occupancy, slots_in_zone


class ZoneStore:
    def __init__(self, base, counts, coverage=None):
        self.base = base
        self.layout = base.layout
        self.counts = counts
        self.coverage = coverage

    def publish(self, snapshot):
        with self.counts.lock:
            self.counts.observe(snapshot)
            self.base.publish(snapshot)

    def read(self):
        with self.counts.lock:
            snapshot = self.base.read()
            zones = self.counts.snapshot()
            masked = {slot['no'] for zone in zones if zone.get('force_blind')
                      for slot in slots_in_zone(self.layout, zone['polygon'])}
            if masked:
                snapshot['slots'] = [
                    {**slot, 'status': 'STATIC_BLIND', 'reason': 'ADMIN_BLIND', 'last_known': None}
                    if slot['no'] in masked else slot for slot in snapshot['slots']
                ]
            estimates, groups = infer_blind_occupancy(self.layout, snapshot, zones, self.coverage)
            return {**snapshot, 'zones': zones, 'blind_slot_estimates': estimates,
                    'blind_group_estimates': groups}

    def install_routes(self, app, public):
        from fastapi import Depends, HTTPException, Request
        from fastapi.responses import FileResponse

        def admin(request: Request):
            local = {'127.0.0.1', '::1', 'localhost'}
            if not request.client or request.client.host not in local or request.url.hostname not in local:
                raise HTTPException(403, '관리자 기능은 서버 컴퓨터에서 사용하세요.')
            origin = request.headers.get('Origin')
            if origin and origin != f'{request.url.scheme}://{request.url.netloc}':
                raise HTTPException(403, '관리자 페이지에서 요청하세요.')

        @app.get('/admin', dependencies=[Depends(admin)], include_in_schema=False)
        def admin_page():
            index = public / 'index.html'
            if not index.is_file():
                raise HTTPException(503, '관리자 화면을 빌드하세요.')
            return FileResponse(index)

        @app.get('/api/admin/zones/events', dependencies=[Depends(admin)])
        def events():
            return self.counts.history()

        @app.post('/api/admin/zones', dependencies=[Depends(admin)])
        def create(payload: dict):
            try:
                return self.counts.create(payload.get('name'), payload.get('polygon', []),
                                          payload.get('gates', []), payload.get('count'))
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc

        @app.delete('/api/admin/zones', dependencies=[Depends(admin)])
        def delete_all():
            self.counts.delete()
            return {'deleted': True}

        @app.delete('/api/admin/zones/{zone_id}', dependencies=[Depends(admin)])
        def delete(zone_id: str):
            self.counts.delete(zone_id)
            return {'deleted': True}

        @app.put('/api/admin/zones/{zone_id}/blind-mode', dependencies=[Depends(admin)])
        def blind_mode(zone_id: str, payload: dict):
            try:
                return self.counts.set_blind_mode(zone_id, payload.get('enabled'))
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

        @app.post('/api/admin/zones/{zone_id}/adjust', dependencies=[Depends(admin)])
        def adjust(zone_id: str, payload: dict):
            try:
                return self.counts.adjust(zone_id, payload.get('action'), payload.get('request_id'),
                                          payload.get('value'), payload.get('target'))
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc


class Engine(occupancy.Engine):
    def __init__(self, args, headless=False):
        super().__init__(args, headless)
        self.zone_counts = None
        if args.web:
            path = Path(os.environ.get('PARKIT_ZONE_DB', Path(__file__).parent / 'logs/zone_counts.sqlite3'))
            path.parent.mkdir(parents=True, exist_ok=True)
            store = self.vehicle_tracking.store
            self.zone_counts = ZoneCounts(path, store.layout['mat'])
            coverage = {no: slot.cams for no, slot in self.tracker.slots.items()}
            self.vehicle_tracking.store = ZoneStore(store, self.zone_counts, coverage)
            print(f'[사각지대 관리자] http://127.0.0.1:{args.web_port}/admin')

    def close(self):
        try:
            super().close()
        finally:
            if self.zone_counts:
                with self.zone_counts.lock:
                    self.zone_counts.db.close()
                self.zone_counts = None


class OccApp(occupancy.OccApp):
    def open_web(self):
        super().open_web()
        if self.e.api_server is not None:
            import webbrowser
            webbrowser.open_new_tab(f"http://127.0.0.1:{self.e.args.web_port}/admin")


def main():
    original_engine, original_app = occupancy.Engine, occupancy.OccApp
    occupancy.Engine, occupancy.OccApp = Engine, OccApp
    try:
        return occupancy.main()
    finally:
        occupancy.Engine, occupancy.OccApp = original_engine, original_app


if __name__ == '__main__':
    sys.exit(main())
