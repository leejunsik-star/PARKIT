"""Larix 주소가 웹캠 번호로 잘못 변환되지 않는지 검증."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from camera_input import validate_source, LARIX_INPUTS
import calibrate


class CameraInputTests(unittest.TestCase):
    def test_webcam_and_phone_addresses(self):
        self.assertEqual(validate_source(' 2 '), 2)
        for url in LARIX_INPUTS:
            self.assertEqual(validate_source(url), url)
        self.assertEqual(validate_source('clip.mp4', allow_file=True), 'clip.mp4')
        for bad in ('', 'phone1', 'srt://localhost:1234', 'rtsp://', 'rtsp://localhost:bad/x'):
            with self.assertRaises(ValueError):
                validate_source(bad)

    def test_mapping_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(calibrate, 'WEBCAM_MAP', str(Path(folder) / 'map.json')):
                mapping = {'CAM-1': LARIX_INPUTS[0], 'CAM-2': LARIX_INPUTS[1], 'CAM-3': 0}
                calibrate.save_webcam_map(mapping)
                self.assertEqual(calibrate.load_webcam_map(), mapping)

    def test_reopen_keeps_url_and_restores_on_failure(self):
        src = calibrate.CalSource(0, None, (1280, 720))
        with patch.object(src, 'close'), patch.object(src, 'open', return_value=True):
            self.assertTrue(src.reopen((1280, 720), LARIX_INPUTS[0]))
            self.assertEqual(src.source, LARIX_INPUTS[0])
        with patch.object(src, 'close'), patch.object(src, 'open', side_effect=[False, False, True]):
            self.assertFalse(src.reopen((1280, 720), LARIX_INPUTS[1]))
            self.assertEqual(src.source, LARIX_INPUTS[0])

    def test_saved_source_preserves_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'cam.json')
            data = {'source': 0, 'contact_mode': 'center', 'H': [[1,0,0],[0,1,0],[0,0,1]]}
            calibrate.save_json(path, data)
            calibrate.set_saved_source(path, LARIX_INPUTS[0])
            data['source'] = LARIX_INPUTS[0]
            self.assertEqual(calibrate.load_json(path), data)


if __name__ == '__main__':
    unittest.main()
