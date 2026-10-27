from pathlib import Path
import importlib
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
core = importlib.import_module(ROOT.name + '.train.train_cryohydro')


class ReleaseExampleTests(unittest.TestCase):
    def test_data_contract_and_spin_up(self):
        arrays = core.load_data(ROOT / 'data', core.CryoHydroConfig())
        self.assertEqual(len(arrays['dates']), 730)
        self.assertEqual(len(arrays['target_dates']), 365)
        self.assertTrue(np.isnan(arrays['y'][:365]).all())
        self.assertTrue(np.isfinite(arrays['y'][365:]).any())
        self.assertFalse((ROOT / 'data' / 'runoff_data.npz').exists())

    def test_saved_example_if_available(self):
        path = ROOT / 'data' / 'outputs' / 'CryoHydro_outputs.npz'
        if not path.exists():
            self.skipTest('Run train_cryohydro.py and run_cryohydro.py first')
        with np.load(path, allow_pickle=False) as z:
            self.assertEqual(z['outputs'].shape, (13, 826, 730))
            self.assertTrue(np.isfinite(z['outputs']).all())
            self.assertTrue(np.isnan(z['target'][:, :365]).all())
        checkpoint = torch.load(ROOT / 'data' / 'checkpoints' / 'CryoHydro.pt', map_location='cpu', weights_only=True)
        self.assertEqual(checkpoint['config']['warm_up'], 365)
        self.assertEqual(checkpoint['routing_method'], 'muskingum')


if __name__ == '__main__':
    unittest.main()
