import unittest
import base64
import io
import json
import queue
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace

from fastapi.testclient import TestClient
import app


class TurboContractTests(unittest.TestCase):
    def setUp(self):
        self.thread_patch = patch.object(app, 'threading', SimpleNamespace(Thread=Mock()))
        self.thread_patch.start()

    def tearDown(self):
        self.thread_patch.stop()
        if app.GEN_LOCK.locked():
            app.GEN_LOCK.release()
        app.JOBS.clear()

    def test_unknown_backend_rejected(self):
        response = TestClient(app.app).post('/api/generate', json={'prompt': 'test', 'backend': 'typo'})
        self.assertEqual(response.status_code, 422)

    def test_turbo_uses_fixed_schedule(self):
        request = app.GenerateRequest(prompt='test', backend='turbo', steps=40, cfg=7)
        with patch.object(app.threading, 'Thread') as thread:
            response = TestClient(app.app).post('/api/generate', json=request.model_dump())
            self.assertEqual(response.status_code, 200)
            dispatched = thread.call_args.kwargs['args'][1]
            self.assertEqual((dispatched.steps, dispatched.cfg), (8, 1))
        app.GEN_LOCK.release()
        app.JOBS.pop(response.json()['job_id'])

    def test_turbo_dimensions_require_32_multiple(self):
        response = TestClient(app.app).post('/api/generate', json={'prompt': 'test', 'backend': 'turbo', 'width': 1000})
        self.assertEqual(response.status_code, 422)

    def test_existing_backends_remain_available(self):
        for backend in ['diffusers', 'gguf']:
            request = app.GenerateRequest(prompt='test', backend=backend, steps=20, cfg=2)
            self.assertEqual((request.steps, request.cfg), (20, 2))

    def test_official_sigmas(self):
        from turbo_worker import SIGMAS
        self.assertEqual(SIGMAS, [1, .978453, .95418, .926626, .89508, .845148, .704534, .414568, 0])

    def test_rgba_reference_keeps_alpha(self):
        from PIL import Image
        from turbo_worker import decode_reference
        data = io.BytesIO()
        Image.new('RGBA', (1001, 777), (20, 30, 40, 90)).save(data, format='PNG')
        encoded = 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()
        image = decode_reference(encoded)
        self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(image.size, (1001, 777))
        self.assertEqual(image.getpixel((0, 0)), (20, 30, 40, 90))

    def test_edit_ignores_unused_canvas_dimensions(self):
        response = TestClient(app.app).post('/api/generate', json={
            'prompt': 'test', 'backend': 'turbo', 'images': ['reference'], 'width': 1000})
        self.assertEqual(response.status_code, 200)

    def test_worker_dispatch_progress_result_and_seed_zero(self):
        from PIL import Image
        app.JOBS['test'] = queue.Queue()
        captured = {}

        def process(command, **kwargs):
            captured.update(json.loads(Path(command[-2]).read_text(encoding='utf-8')))
            Image.new('RGB', (1024, 768)).save(command[-1])
            worker = Mock()
            worker.stdout = ['progress: \u2588\n', 'STUDIO_EVENT {"type":"ready"}\n',
                             'STUDIO_EVENT {"type":"step","step":8,"total":8}\n']
            worker.wait.return_value = 0
            worker.__enter__ = Mock(return_value=worker)
            worker.__exit__ = Mock(return_value=False)
            return worker

        console = io.TextIOWrapper(io.BytesIO(), encoding='cp932')
        with patch.object(app, 'turbo_config', return_value={'python': 'python'}), \
                patch.object(app.subprocess, 'Popen', side_effect=process), \
                patch('sys.stdout', console):
            app.run_turbo_job('test', app.GenerateRequest(prompt='test', backend='turbo', seed=0))
        events = list(app.JOBS['test'].queue)
        self.assertEqual(captured['seed'], 0)
        self.assertIn({'type': 'step', 'step': 8, 'total': 8}, events)
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual((events[-1]['meta']['width'], events[-1]['meta']['height']), (1024, 768))
        self.assertEqual(app.STATE['status'], 'idle')

    def test_load_failure_reaches_events_and_status(self):
        app.JOBS['test'] = queue.Queue()
        with patch.object(app, 'load_model', side_effect=RuntimeError('load failed')):
            app.run_diffusers_job('test', app.GenerateRequest(prompt='test'))
        self.assertEqual(app.STATE['status'], 'error')
        self.assertIn('load failed', app.JOBS['test'].get()['message'])

    def test_dispatch_releases_own_model_before_turbo(self):
        app.GEN_LOCK.acquire()
        calls = []
        with patch.object(app, 'unload_diffusers', side_effect=lambda: calls.append('unload')), \
                patch.object(app, 'run_turbo_job', side_effect=lambda *_: calls.append('turbo')):
            app.run_generation('test', app.GenerateRequest(prompt='test', backend='turbo'))
        self.assertEqual(calls, ['unload', 'turbo'])
        self.assertFalse(app.GEN_LOCK.locked())

    def test_worker_failure_reaches_events_and_status(self):
        app.JOBS['test'] = queue.Queue()
        worker = Mock()
        worker.stdout = ['CUDA out of memory\n']
        worker.wait.return_value = 1
        worker.__enter__ = Mock(return_value=worker)
        worker.__exit__ = Mock(return_value=False)
        with patch.object(app, 'turbo_config', return_value={'python': 'python'}), \
                patch.object(app.subprocess, 'Popen', return_value=worker):
            app.run_turbo_job('test', app.GenerateRequest(prompt='test', backend='turbo'))
        events = list(app.JOBS['test'].queue)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('CUDA out of memory', events[-1]['message'])
        self.assertEqual(app.STATE['status'], 'error')


if __name__ == '__main__':
    unittest.main()
