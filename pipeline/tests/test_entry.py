"""SELFTEST reporting incomplete artifacts never claims completed experiments."""
import json,tempfile,unittest
from pathlib import Path
from pipeline.entry import status

class EntryTests(unittest.TestCase):
    def test_underfit_record_is_not_a_completed_method_result(self):
        with tempfile.TemporaryDirectory(prefix='SELFTEST_') as d:
            root=Path(d);p=root/'pipeline_results/paired';p.mkdir(parents=True)
            (p/'COMPLETE.json').write_text(json.dumps({'complete':False,'reason':'UNDERFIT_INCONCLUSIVE'}))
            r=status(root)
            self.assertFalse(r['method_results_available']);self.assertFalse(r['gpu_started'])
            self.assertFalse(r['training_prerequisites_met'])
            g=root/'pipeline_state/gpu';g.mkdir(parents=True);(g/'paired_started.json').write_text('{}')
            (p/'COMPLETE.json').write_text(json.dumps({'complete':True,'reason':'WAITING_BLIND_REVIEW'}))
            r=status(root);self.assertTrue(r['method_results_available']);self.assertTrue(r['gpu_started'])
            self.assertIn('ONLY_GUARD',r['launch_permission'])
