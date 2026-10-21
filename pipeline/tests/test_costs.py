"""SELFTEST device-accounting arithmetic, not real experiment costs."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline.costs import cost_summary
class CostTests(unittest.TestCase):
    def test_partial_or_unsettled_cost_does_not_become_comparison(self):
        with tempfile.TemporaryDirectory(prefix='SELFTEST_') as d:
            root=Path(d);g=root/'pipeline_state/gpu';g.mkdir(parents=True)
            (g/'diagnostics_started.json').write_text(json.dumps({'event':'start','phase':'diagnostics'}))
            with patch('pipeline.costs.PARENT',root/'no_parent'):
                r=cost_summary(root);self.assertFalse(r['settled']);self.assertIsNone(r['remaining_gpu_seconds'])
                (g/'diagnostics_completion.json').write_text(json.dumps({'event':'end','phase':'diagnostics','gpu_device_seconds':80}))
                for arm,charge in [('paired',100),('paired_worst_witness',120)]:
                    (g/f'{arm}_started.json').write_text(json.dumps({'event':'start','phase':arm}))
                    (g/f'{arm}_completion.json').write_text(json.dumps({'event':'end','phase':arm,'gpu_device_seconds':charge}))
                    p=root/'pipeline_results'/arm;p.mkdir(parents=True);(p/'COMPLETE.json').write_text(json.dumps({'complete':True}))
                r=cost_summary(root);self.assertTrue(r['comparison_complete']);self.assertEqual(r['allocated_total_per_completed_arm'],{'paired':140.,'paired_worst_witness':160.})
