"""Replay identical synthetic videos with tracking off/on; preserve user logs."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from parkit_config import Params
from parkit_geom import SlotTable
import test_pipeline as fixture


class ReplayTests(unittest.TestCase):
    def test_original_and_tracking_replay_state_changes_identical(self):
        with tempfile.TemporaryDirectory(prefix='parkit-tracking-') as directory:
            previous = fixture.OUT
            fixture.OUT = directory
            try:
                table = SlotTable(fixture.TABLE)
                fixture.make_videos(table, Params({}, table))
            finally:
                fixture.OUT = previous
            base = Path(directory)
            cfg = json.loads((base/'run_test.json').read_text())
            records = []
            for label, flags in [('original', []), ('tracking', ['--tracking']), ('tracking_20x', ['--tracking'])]:
                cfg['log']['dir'] = str(base/label)
                path = base/'run_test.json'
                path.write_text(json.dumps(cfg))
                result = subprocess.run([sys.executable, str(Path(__file__).parent/'occupancy.py'),
                                         '--run', str(path), '--detector', 'gt', '--no-window', '--quiet',
                                         '--speed', '20' if label=='tracking_20x' else '0', *flags],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                log = next((base/label).glob('*/state_changes.csv'))
                with log.open(encoding='utf-8-sig') as f:
                    records.append([{k:v for k,v in row.items() if k!='wall'} for row in csv.DictReader(f)])
            self.assertEqual(records[0], records[1])
            self.assertEqual(records[0], records[2])
            self.assertGreater(len(records[0]), 56)
            print(f'Tracking off/on/20x replay: identical {len(records[0])} state-change rows (wall excluded).')


if __name__=='__main__':
    unittest.main(verbosity=2)
