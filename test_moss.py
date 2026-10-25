import json
import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image
import app


class MossContractTests(unittest.TestCase):
    def setUp(self):
        self.thread_patch = patch.object(app, 'threading', SimpleNamespace(Thread=Mock()))
        self.thread_patch.start()

    def tearDown(self):
        self.thread_patch.stop()
        if app.GEN_LOCK.locked():
            app.GEN_LOCK.release()
        app.JOBS.clear()

    def test_official_defaults_and_explicit_sampling(self):
        for backend in ('moss', 'moss_int8'):
            for supplied, expected in [({}, (50, 3)), ({'steps': 28, 'cfg': 4}, (28, 4))]:
                response = TestClient(app.app).post('/api/generate', json={
                    'prompt': 'a bird', 'backend': backend, **supplied})
                self.assertEqual(response.status_code, 200)
                request = app.threading.Thread.call_args.kwargs['args'][1]
                self.assertEqual((request.steps, request.cfg), expected)
                app.GEN_LOCK.release()

    def test_reference_editing_rejected_before_starting_worker(self):
        for backend in ('moss', 'moss_int8'):
            response = TestClient(app.app).post('/api/generate', json={
                'prompt': 'edit', 'backend': backend, 'images': ['reference']})
            self.assertEqual(response.status_code, 422)
        self.assertFalse(app.threading.Thread.called)
        self.assertFalse(app.GEN_LOCK.locked())

    def test_dimensions_require_16_multiple(self):
        response = TestClient(app.app).post('/api/generate', json={
            'prompt': 'a bird', 'backend': 'moss', 'width': 1000})
        self.assertEqual(response.status_code, 422)

    def test_moss_worker_path_progress_output_and_seed_zero(self):
        app.JOBS['test'] = queue.Queue()
        captured = {}

        def process(command, **kwargs):
            self.assertEqual(Path(command[2]).name, 'moss_worker.py')
            captured.update(json.loads(Path(command[-2]).read_text(encoding='utf-8')))
            Image.new('RGB', (1024, 768)).save(command[-1])
            worker = Mock()
            worker.stdout = ['STUDIO_EVENT {"type":"step","step":50,"total":50}\n']
            worker.wait.return_value = 0
            worker.__enter__ = Mock(return_value=worker)
            worker.__exit__ = Mock(return_value=False)
            return worker

        with patch.object(app, 'moss_config', return_value={'python': 'moss-python', 'model': 'local-model'}), \
                patch.object(app.subprocess, 'Popen', side_effect=process):
            app.run_moss_job('test', app.GenerateRequest(prompt='test', backend='moss', steps=50, cfg=3, seed=0))
        events = list(app.JOBS['test'].queue)
        self.assertEqual(captured['seed'], 0)
        self.assertIn({'type': 'step', 'step': 50, 'total': 50}, events)
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual(events[-1]['meta']['backend'], 'MOSS-Image-SFT')
        self.assertEqual(events[-1]['meta']['width'], 1024)
        self.assertEqual(app.STATE['status'], 'idle')

    def test_moss_releases_standard_model_first(self):
        app.GEN_LOCK.acquire()
        calls = []
        with patch.object(app, 'unload_diffusers', side_effect=lambda: calls.append('unload')), \
                patch.object(app, 'run_moss_job', side_effect=lambda *_: calls.append('moss')):
            app.run_generation('test', app.GenerateRequest(prompt='test', backend='moss'))
        self.assertEqual(calls, ['unload', 'moss'])
        self.assertFalse(app.GEN_LOCK.locked())

    def test_worker_rejects_images_without_importing_cuda(self):
        from moss_worker import run
        from moss_int8_worker import run as run_int8
        with self.assertRaisesRegex(ValueError, 'not reference-image editing'):
            run({'images': ['reference']}, 'unused.png')
        with self.assertRaisesRegex(ValueError, 'not support reference-image editing'):
            run_int8({'images': ['reference']}, 'unused.png')


if __name__ == '__main__':
    unittest.main()
