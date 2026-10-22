"""CPU-only native node checks: .venv-turbo/Scripts/python check_native_turbo.py"""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parent / '.vendor' / 'ComfyUI'))
sys.argv = [sys.argv[0], '--cpu']
import torch
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from comfy_extras.nodes_custom_sampler import ManualSigmas
from turbo_worker import SIGMAS
sys.argv = [sys.argv[0]]


class NativeTurboTests(unittest.TestCase):
    def test_multiple_references_preserve_rgba_and_first_canvas_size(self):
        seen = []
        clip = SimpleNamespace(
            tokenize=lambda *args, **kwargs: None,
            encode_from_tokens_scheduled=lambda *args: [[torch.zeros(1), {}]])
        vae = SimpleNamespace(encode=lambda image: seen.append(tuple(image.shape)) or
                              torch.zeros(1, 64, image.shape[1] // 16, image.shape[2] // 16))
        _, _, latent = TextEncodeQwenImage21.execute(
            clip, 'test', '', vae=vae, resolution=0, images={
                'image_1': torch.zeros(1, 777, 1001, 4),
                'image_2': torch.zeros(1, 512, 512, 3)}).result
        self.assertEqual(seen, [(1, 768, 992, 4), (1, 512, 512, 3)])
        self.assertEqual(list(latent['samples'].shape), [1, 64, 48, 62])

    def test_manual_sigmas_keep_all_eight_steps_and_terminal_zero(self):
        sigmas = ManualSigmas.execute(','.join(map(str, SIGMAS)))[0]
        torch.testing.assert_close(sigmas, torch.tensor(SIGMAS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
