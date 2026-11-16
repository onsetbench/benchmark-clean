import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from onsetbench import submission as S
from onsetbench import cli, _frozen as F
import numpy as np

class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset.json'
        self.pred = self.root/'predictions.json'
        self.d = dict(schema='onsetbench.dataset/1', clock=S.CLOCK, inputs='standard_video',
                      records=[dict(recording='toy', starts_ms=[-200,0,0,800], span_ms=800, references={})])
        self.dataset.write_text(json.dumps(self.d))
    def tearDown(self): self.tmp.cleanup()
    def write(self,p): self.pred.write_text(json.dumps(p))
    def test_reference_free_template_and_validation(self):
        p=S.template(self.dataset,'example');self.write(p)
        with patch.object(cli, 'dataset', side_effect=AssertionError('No scorer reference reads')):
            self.assertEqual(S.validate(self.dataset,self.pred)['recordings'],1)
    def test_duplicate_events_negative_times_and_repeated_grid(self):
        p=S.template(self.dataset,'example')
        p['recordings']['toy']=dict(onsets_ms=[-100,0,0],midpoints_ms=[-100,0,0],contact=[1,0,-1,1])
        self.write(p);S.validate(self.dataset,self.pred)
        self.assertEqual(S.load(self.pred),p)
    def test_wrong_binding_refuses(self):
        p=S.template(self.dataset,'example');p['dataset_sha256']='0'*64;self.write(p)
        with self.assertRaisesRegex(ValueError,'checksum'): S.validate(self.dataset,self.pred)
    def test_bad_midpoint_count_refuses(self):
        p=S.template(self.dataset,'example');p['recordings']['toy'].update(onsets_ms=[0],midpoints_ms=[]);self.write(p)
        with self.assertRaisesRegex(ValueError,'midpoint'):S.validate(self.dataset,self.pred)
    def test_bad_contact_count_refuses(self):
        p=S.template(self.dataset,'example');p['recordings']['toy']['contact']=[1];self.write(p)
        with self.assertRaisesRegex(ValueError,'grid'):S.validate(self.dataset,self.pred)
    def test_epoch_clock_refuses(self):
        self.d['clock']='epoch_ms';self.dataset.write_text(json.dumps(self.d))
        with self.assertRaisesRegex(ValueError,'recording-relative'):S.template(self.dataset,'example')
    def test_frozen_scoring_translation_parity(self):
        truth=np.array([0.,1000.]);pred=np.array([-50.,1050.,1050.]);start=np.array([-200.,0.,0.,800.]);span=np.full(4,800.)
        offset=1700000000000.
        self.assertEqual(F.score_block(truth,pred,200),F.score_block(truth+offset,pred+offset,200))
        self.assertEqual(F.union_ms(start,span),F.union_ms(start+offset,span))
        ref={'spans':np.array([[0.,500.],[1000.,1500.]])}
        np.testing.assert_array_equal(F.window_labels(start,span,ref),F.window_labels(start+offset,span,{'spans':ref['spans']+offset}))

if __name__=='__main__':unittest.main()
