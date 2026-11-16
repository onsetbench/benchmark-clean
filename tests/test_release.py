import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from onsetbench import _frozen as F
from onsetbench import cli,job

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT/'examples',self.root/'examples')
        self.dataset = self.root/'examples/dataset.json'
        self.predictions = self.root/'examples/predictions.json'

    def tearDown(self):self.tmp.cleanup()

    def change_predictions(self,edit):
        d = cli.load(self.predictions)
        edit(d)
        self.predictions.write_text(json.dumps(d)+'\n')

    def test_hand_counted_end_to_end(self):
        r = cli.score(self.dataset,self.predictions)
        for tol,counts,f1 in [('200',(3,2,0),.75),('100',(2,3,1),.5)]:
            x = r['timing'][tol]['arms']['native']
            self.assertEqual((x['tp'],x['fp'],x['fn']),counts)
            self.assertEqual(x['f1'],f1)

    def test_missing_recording_refuses(self):
        self.change_predictions(lambda p:p['recordings'].pop('toy-b'))
        with self.assertRaisesRegex(ValueError,'every dataset recording'):cli.score(self.dataset,self.predictions)

    def test_unknown_recording_refuses(self):
        self.change_predictions(lambda p:p['recordings'].update(hidden={'onsets_ms':[]}))
        with self.assertRaisesRegex(ValueError,'every dataset recording'):cli.score(self.dataset,self.predictions)

    def test_wrong_manifest_binding_refuses(self):
        self.change_predictions(lambda p:p.update(dataset_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'checksum differs'):cli.score(self.dataset,self.predictions)

    def test_reference_tamper_refuses(self):
        p = self.root/'examples/refs/toy-a.json'
        p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Reference checksum'):cli.score(self.dataset,self.predictions)

    def test_boolean_time_refuses(self):
        self.change_predictions(lambda p:p['recordings']['toy-a'].update(onsets_ms=[True]))
        with self.assertRaisesRegex(ValueError,'numbers'):cli.score(self.dataset,self.predictions)

    def test_nonfinite_json_refuses(self):
        self.predictions.write_text('{"x": NaN}')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):cli.load(self.predictions)

    def test_repeated_json_fields_refuse(self):
        self.predictions.write_text('{"x": 1, "x": 2}')
        with self.assertRaisesRegex(ValueError,'Repeated'):cli.load(self.predictions)

    def test_silent_policy_keeps_truth_and_exposure(self):
        self.change_predictions(lambda p:[e.update(onsets_ms=[]) for e in p['recordings'].values()])
        r = cli.score(self.dataset,self.predictions)
        x = r['timing']['200']['arms']['native']
        self.assertEqual(x['fn'],3)
        self.assertIsNone(x['precision'])
        self.assertIsNone(x['lag_median_ms'])
        self.assertGreater(r['support_hours'],0)

    def test_midpoint_identity_is_required(self):
        self.change_predictions(lambda p:p['recordings']['toy-a'].update(midpoints_ms=[]))
        with self.assertRaisesRegex(ValueError,'every recording'):cli.score(self.dataset,self.predictions)

    def test_outside_events_are_counted_not_clipped(self):
        self.change_predictions(lambda p:p['recordings']['toy-b']['onsets_ms'].append(1700000010000))
        r = cli.score(self.dataset,self.predictions)
        x = next(x for x in r['raw'] if x['recording']=='toy-b' and x['arm']=='native' and x['tolerance_ms']==200)
        self.assertEqual(x['n_emitted'],4)
        self.assertEqual(x['predictions_outside_block'],1)

    def test_all_unknown_contact_reduces_coverage(self):
        self.change_predictions(lambda p:[e.update(contact=[0]*10) for e in p['recordings'].values()])
        r = cli.score(self.dataset,self.predictions)['contact_state']
        self.assertEqual(r['decision_coverage'],0)
        self.assertEqual(r['balanced_correct_class_coverage'],0)
        self.assertIsNone(r['precision_contact'])

    def test_paths_cannot_escape(self):
        with self.assertRaisesRegex(ValueError,'leaves'):cli.contained(self.root,'../elsewhere.json')

    def test_functions_match_saved_source_segments(self):
        provenance = cli.load(ROOT/'SCORER_PROVENANCE.json')
        text = (ROOT/'src/onsetbench/_frozen.py').read_text()
        nodes = {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        for p in provenance['functions']:
            part = ast.get_source_segment(text,nodes[p['function']])
            self.assertEqual(hashlib.sha256(part.encode()).hexdigest(),p['function_sha256'])

    def test_runner_seals_before_reference_access(self):
        inputs = self.root/'inputs'
        inputs.mkdir()
        rows = []
        for rid in ('toy-a','toy-b'):
            video = inputs/(rid+'.mp4');video.write_bytes(b'toy video')
            clock = inputs/(rid+'.json')
            clock.write_text(json.dumps(dict(recording=rid,clock='epoch_ms',video_sha256=cli.sha(video),
                                             frame_timestamps_ms=[1700000000000]))+'\n')
            rows.append(dict(recording=rid,video=video.name,timestamps=clock.name,frames=1,
                             video_sha256=cli.sha(video),timestamps_sha256=cli.sha(clock)))
        manifest = inputs/'inputs.json'
        manifest.write_text(json.dumps(dict(schema='onsetbench.inputs/1',dataset_id='synthetic-example',
             complete=True,inputs='standard_video',recordings=rows))+'\n')
        class Model:
            @staticmethod
            def predict(**kwargs):
                self.assertEqual(set(kwargs),{'video','frame_timestamps_ms','window_starts_ms','window_span_ms'})
                return {'onsets_ms':[]}
        out = self.root/'job'
        original_load = cli.load
        def checked_load(path):
            if Path(path).parent.name=='refs':
                self.assertTrue((out/'prediction_seal.json').is_file())
            return original_load(path)
        with patch.object(job.importlib,'import_module',return_value=Model),patch.object(cli,'load',side_effect=checked_load):
            job.run(self.dataset,manifest,'local:predict','toy','SENSOR',out)
        self.assertEqual(cli.load(out/'score.json')['timing']['200']['arms']['native']['fn'],3)

    def test_invalid_target_refuses(self):
        self.change_predictions(lambda p:p.update(target="INVALID"))
        with self.assertRaisesRegex(ValueError,"Target must"):
            cli.score(self.dataset,self.predictions)

    def test_unknown_reference_target_refuses(self):
        d=cli.load(self.dataset)
        r=d['records'][0]
        r['references']['OTHER']=r['references'].pop('SENSOR')
        self.dataset.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'Unknown reference target'):
            cli.dataset(self.dataset)

    def test_runner_unknown_target_refuses_before_io(self):
        with self.assertRaisesRegex(ValueError,'Target must be SENSOR'):
            job.run('missing','missing','missing','example','OTHER','missing')

    def test_cli_unknown_target_refuses(self):
        import contextlib,io
        for command in ('template','run'):
            args=[command,'--dataset','missing','--out','missing','--target','OTHER']
            if command=='run':args+=['--inputs','missing','--predictor','missing','--name','example']
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as cm:
                cli.main(args)
            self.assertEqual(cm.exception.code,2)


if __name__=='__main__':unittest.main()
