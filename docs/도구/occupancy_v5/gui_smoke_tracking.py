"""Manual GUI smoke test with existing synthetic pipeline video, never a webcam.
Run after test_pipeline.py (or pass --run to another synthetic GT run config).
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from occupancy import Engine, OccApp
from parkit_tk import check_tk


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default=str(Path(__file__).parent/'logs/test_pipeline/run_test.json'))
    args=ap.parse_args()
    if not Path(args.run).is_file():
        raise SystemExit('Run test_pipeline.py first to create the synthetic GT fixture.')
    opts=SimpleNamespace(run=args.run,preset=None,source=None,log_frames=False,every=5,speed=1.,
                         model=None,conf=None,quiet=True,detector='gt',tracking=True,tracking_config=None,
                         web=False,duration=3.)
    check_tk()
    e=Engine(opts)
    app=OccApp(e)
    result={}
    def inspect():
        result.update(ready=app.ready,error=app.error,window=bool(app.root.winfo_exists()),
                      vehicles=len(e.vehicle_tracking.store.read()['vehicles']))
        app.quit()
    app.root.after(5000,inspect)
    app.root.mainloop()
    print('GUI_SMOKE',result)
    return 0 if result.get('ready') and result.get('window') and not result.get('error') and result.get('vehicles',0)>0 else 1


if __name__=='__main__':
    sys.exit(main())
