"""CPU SELFTEST only; no native generator or efficacy artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from pipeline.model import OrdinaryUpdater,paired_objective,witness_masks
from pipeline.policy import phase_gate,remaining_budget,observed_pattern_errors

def pattern_manifest():
    rows=[]
    for scenario in ('cup','teapot','candy'):
        values=[('same','clean'),('same','clean'),('same','marked')] if scenario!='teapot' else [('clear','red'),('clear','blue'),('red','red')]
        for i,(a,b) in enumerate(values):rows.append({'id':scenario+str(i),'scenario':scenario,'split':'calibration','readouts':{'left':{'truth':a},'right':{'truth':b}}})
    return {'clips':rows,'units':[]}

class ModelPolicyTests(unittest.TestCase):
    def test_same_objective_graph_weight_difference_and_gradients(self):
        torch.manual_seed(10)
        p=torch.randn(3,2,3,4,6,requires_grad=True);t=torch.randn_like(p);m=torch.ones(3,3,4,6)
        a,sa=paired_objective(p,t,m,'paired');b,sb=paired_objective(p,t,m,'paired_worst_witness')
        self.assertEqual(sa,sb);self.assertAlmostEqual(float(b-a),.1*sa['worst_relation'],places=5)
        ga=torch.autograd.grad(a,p,retain_graph=True)[0];gb=torch.autograd.grad(b,p)[0]
        self.assertTrue(torch.isfinite(ga).all() and torch.isfinite(gb).all());self.assertGreater(float((ga-gb).abs().sum()),0)
    def test_recurrent_memory_shape_and_trainable_path(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        model=OrdinaryUpdater();x=torch.randn(1,4,64,1536);actions=torch.tensor([[0,1,2,6]])
        final,states=model(x,actions,True);self.assertEqual(tuple(final.shape),(1,64,1536));self.assertEqual(len(states),4)
        final.square().mean().backward();self.assertGreater(float(model.gate.weight.grad.abs().sum()),0)
    def test_tiny_readout_mask_not_silently_empty(self):
        clip={'rois':{'left':[.001,.001,.01,.01],'right':[.7,.7,.9,.9]},'window_fractions':{'left':[.001,.01],'right':[.8,.9]}}
        masks=witness_masks([clip]*3,(16,6,32,56));self.assertTrue(torch.all(masks.sum((1,2,3))>0))
    def test_budgets_do_not_reuse_prior_reservation_or_unsettled_stage(self):
        rows=[{'event':'start','phase':'diagnostics'},{'event':'end','phase':'diagnostics','gpu_device_seconds':100.}]
        self.assertAlmostEqual(remaining_budget(rows)['remaining_seconds'],5106.652899978682)
        with self.assertRaises(ValueError):remaining_budget(rows+[{'event':'start','phase':'paired'}])
    def test_missing_gates_and_ordinary_solves_stop(self):
        with tempfile.TemporaryDirectory(prefix='SELFTEST_') as d:
            r=Path(d);self.assertFalse(phase_gate(r,'paired')['allowed'])
            def put(path,obj):
                p=r/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
            put('pipeline_data/SOURCE_READY.json',{'calibration_ready':True,'formal_ready':True})
            put('pipeline_data/manifest.json',pattern_manifest())
            put('pipeline_state/MEASUREMENT_READY.json',{'ready':True})
            put('pipeline_state/reviews/diagnostics/score.json',{'complete':True,'judgeable':True,'all_correct':True})
            self.assertTrue(phase_gate(r,'cache')['allowed']);self.assertFalse(phase_gate(r,'paired')['allowed'])
            put('pipeline_cache/READY.json',{'ready':True})
            self.assertTrue(phase_gate(r,'paired')['allowed'])
            put('pipeline_results/paired/COMPLETE.json',{'training_adequate':True})
            put('pipeline_state/reviews/paired/score.json',{'complete':True,'judgeable':True,'all_correct':True})
            self.assertEqual(phase_gate(r,'paired_worst_witness')['reason'],'STOP_ORDINARY_SOLVES_PILOT')
            put('pipeline_state/reviews/paired/score.json',{'complete':True,'judgeable':False,'all_correct':False})
            self.assertFalse(phase_gate(r,'paired_worst_witness')['allowed'])
            put('pipeline_state/reviews/paired/score.json',{'complete':True,'judgeable':True,'all_correct':False})
            self.assertTrue(phase_gate(r,'paired_worst_witness')['allowed'])
    def test_actual_failed_reset_never_uses_planned_truth(self):
        good=pattern_manifest();self.assertEqual(observed_pattern_errors(good),[])
        good['clips'][1]['readouts']['left']['truth']='old_colour_still_present'
        self.assertTrue(observed_pattern_errors(good))

if __name__=='__main__':unittest.main()
