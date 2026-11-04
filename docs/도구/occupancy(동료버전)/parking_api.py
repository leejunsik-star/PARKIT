"""Optional FastAPI server. It reads StateStore only; never calls Engine/tkinter."""
import asyncio
from pathlib import Path
import socket
import threading
import time
import anyio


def create_app(store, static_dir=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    app = FastAPI(title='PARKIT 2D tracking', docs_url='/api/docs')

    @app.get('/api/layout')
    def layout():
        return store.layout

    @app.get('/api/state')
    def state():
        return store.read()

    async def send_states(ws):
        while True:
            # Full snapshot immediately, then bounded-rate heartbeats. Slow
            # clients miss intermediate snapshots, never accumulate a backlog.
            await asyncio.wait_for(ws.send_json(store.read()), timeout=2.)
            await asyncio.sleep(.2)

    @app.websocket('/ws/parking')
    async def parking(ws: WebSocket):
        await ws.accept()
        sender = asyncio.create_task(send_states(ws))
        try:
            async def receive():
                while True:
                    await ws.receive_text()
            receiver = asyncio.create_task(receive())
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, TimeoutError, RuntimeError, OSError):
            pass
        finally:
            sender.cancel()
            receiver.cancel()
            # ASGI hosts may cancel the handler during disconnect. Finish child
            # task cleanup inside a shield so cancellation cannot leak tasks.
            with anyio.CancelScope(shield=True):
                await asyncio.gather(sender, receiver, return_exceptions=True)
                try:
                    with anyio.move_on_after(.5):
                        await ws.close()
                except (RuntimeError, OSError):
                    pass

    public = Path(static_dir) if static_dir else Path(__file__).parent / 'web' / 'dist'
    if public.is_dir() and (public / 'index.html').is_file():
        app.mount('/assets', StaticFiles(directory=public / 'assets'), name='assets')

        @app.get('/')
        def index():
            return FileResponse(public / 'index.html')
    else:
        @app.get('/')
        def instructions():
            return {"message": "Build web/ with npm ci && npm run build, then restart; or use Vite on port 5173."}
    return app


class ApiServer:
    def __init__(self, store, host='127.0.0.1', port=8765):
        import uvicorn
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind((host, port))
            self.socket.listen(128)
            self.server = uvicorn.Server(uvicorn.Config(create_app(store), host=host, port=port,
                                                        log_level='warning', ws='websockets',
                                                        timeout_graceful_shutdown=2))
        except BaseException:
            self.socket.close()
            raise
        self.thread = threading.Thread(target=self.server.run, kwargs={'sockets': [self.socket]}, daemon=True)

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError('PARKIT API server failed to start')
            time.sleep(.01)
        print(f"[2D] http://{self.server.config.host}:{self.socket.getsockname()[1]}")

    def close(self):
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=4)
        self.socket.close()
