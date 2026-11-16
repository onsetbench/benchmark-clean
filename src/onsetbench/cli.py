"""Validate public files and score delivered predictions with frozen functions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from . import _frozen as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError('Repeated JSON field: '+key)
        result[key] = value
    return result


def load(path):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object, parse_constant=nonfinite)


def save(path, value):
    with Path(path).open('xb') as f:
        f.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def numbers(value, label):
    if not isinstance(value,list) or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in value):
        raise ValueError(label+' must be a list of numbers')
    a = np.asarray(value,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():
        raise ValueError(label+' must contain finite numbers')
    return a


def contained(root, relative):
    p = (root/relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('File path leaves the package')
    return p


def dataset(path):
    path = Path(path).resolve()
    d = load(path)
    if d.get('schema')!='onsetbench.dataset/1' or d.get('clock')!='epoch_ms' or d.get('inputs')!='standard_video':
        raise ValueError('Unsupported dataset schema, clock or inputs')
    records = d.get('records')
    if not isinstance(records,list) or not records:
        raise ValueError('Dataset has no recordings')
    ids = [r['recording'] for r in records]
    if any(not isinstance(r,str) or not r for r in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Invalid or repeated recording id')
    root = path.parent.parent
    for r in records:
        starts = numbers(r['starts_ms'],'starts_ms')
        span = r['span_ms']
        if not len(starts) or np.any(np.diff(starts)<0) or isinstance(span,bool) or not isinstance(span,(int,float)) or not np.isfinite(span) or span<=0:
            raise ValueError('Invalid retained scoring grid')
        for target,spec in r['references'].items():
            if target != 'SENSOR':
                raise ValueError('Unknown reference target')
            p = contained(root,spec['path'])
            if sha(p)!=spec['sha256']:
                raise ValueError('Reference checksum mismatch: '+r['recording'])
            ref = load(p)
            if ref.get('recording')!=r['recording']:
                raise ValueError('Reference recording differs')
            on = numbers(ref['events_ms'],'reference onsets')
            if np.any(np.diff(on)<0):
                raise ValueError('Reference onsets are unordered')
            if target=='SENSOR':
                off = numbers(ref['releases_ms'],'reference releases')
                if off.shape!=on.shape or np.any(off<on):
                    raise ValueError('Reference releases differ or precede onset')
    return d,root


def score(dataset_path,predictions_path):
    d,root = dataset(dataset_path)
    p = load(predictions_path)
    if p.get('schema')!='onsetbench.predictions/1' or p.get('dataset_sha256')!=sha(dataset_path):
        raise ValueError('Prediction schema or dataset checksum differs')
    if p.get('inputs')!='standard_video' or not isinstance(p.get('name'),str) or not p['name'].strip():
        raise ValueError('Declare an entry name and standard_video inputs')
    target = p.get('target')
    if target != 'SENSOR':
        raise ValueError('Target must be SENSOR')
    predictions = p.get('recordings')
    ids = {r['recording'] for r in d['records']}
    if not isinstance(predictions,dict) or set(predictions)!=ids:
        raise ValueError('Submission must include every dataset recording exactly once')
    fields = {'onsets_ms','midpoints_ms','contact'}
    for rid,e in predictions.items():
        if not isinstance(e,dict) or set(e)-fields:
            raise ValueError('Unknown prediction field in '+rid)
    has_contact = [e.get('contact') is not None for e in predictions.values()]
    has_midpoint = [e.get('midpoints_ms') is not None for e in predictions.values()]
    for values,label in [(has_contact,'Contact decisions'),(has_midpoint,'Source midpoints')]:
        if any(values) and not all(values):
            raise ValueError(label+' must cover every recording or be omitted')
    raw,hours,states = [],0.0,[]
    for r in sorted(d['records'],key=lambda r:r['recording']):
        rid = r['recording']
        ref = load(contained(root,r['references'][target]['path']))
        truth = numbers(ref['events_ms'],'reference')
        e = predictions[rid]
        native = numbers(e['onsets_ms'],'onsets_ms')
        mid = None if e.get('midpoints_ms') is None else numbers(e['midpoints_ms'],'midpoints_ms')
        if len(truth)*len(native)>2_000_000:
            raise ValueError('Event stream exceeds scorer memory capacity; no score was produced')
        starts = np.asarray(r['starts_ms'])
        lo,hi = float(starts.min()),float(starts.max()+r['span_ms'])
        hours += F.union_ms(starts,np.full(len(starts),r['span_ms']))/3_600_000
        arms = F.controlled_times(native,mid,lo,hi)
        for tol in (200,100):
            for arm,times in arms.items():
                raw.append(dict(F.score_block(truth,times,tol),recording=rid,arm=arm,tolerance_ms=tol,
                                reference_outside_block=int(((truth<lo)|(truth>=hi)).sum()),
                                predictions_outside_block=int(((np.asarray(times)<lo)|(np.asarray(times)>=hi)).sum())))
        if all(has_contact):
            decisions = numbers(e['contact'],'contact')
            if len(decisions)!=len(starts) or not np.isin(decisions,[-1,0,1]).all():
                raise ValueError('Contact must contain one -1, 0 or 1 decision per retained window')
            on,off = np.asarray(ref['events_ms']),np.asarray(ref['releases_ms'])
            labels = F.window_labels(starts,np.full(len(starts),r['span_ms']),{'spans':np.stack([on,off],axis=1)})
            states.append(F.state_counts(decisions,labels))
    timing = {}
    for tol in (200,100):
        rows = [r for r in raw if r['tolerance_ms']==tol]
        arms = sorted({r['arm'] for r in rows})
        counts = {a:{r['recording']:r for r in rows if r['arm']==a} for a in arms}
        contrast = {'mean_shift':F.paired_contrast(counts['native'],[counts['shift_'+str(x)] for x in (25,50,75)])}
        if 'midpoint' in counts:
            contrast['midpoint'] = F.paired_contrast(counts['native'],[counts['midpoint']])
        timing[str(tol)] = {'arms':{a:F.summarize(list(counts[a].values()),hours) for a in arms},'paired_contrasts':contrast}
    state = F.state_component({k:sum(r[k] for r in states) for k in F.STATE_KEYS}) if states else None
    return dict(schema='onsetbench.score/1',evaluator_version=__version__,name=p['name'],target=target,
                dataset_id=d['dataset_id'],split=d['split'],clock=d['clock'],recordings=len(ids),
                dataset_sha256=sha(dataset_path),predictions_sha256=sha(predictions_path),
                support_hours=hours,timing=timing,contact_state=state,raw=raw,
                scope='Public-split evaluation.')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Evaluate physical events found using standard video.')
    ap.add_argument('--version',action='version',version=__version__)
    sub = ap.add_subparsers(dest='command',required=True)
    for command in ('verify','template','score','prepare-video','run'):
        c = sub.add_parser(command)
        c.add_argument('--dataset',required=True,type=Path)
        if command in ('template','score','prepare-video','run'):c.add_argument('--out',required=True,type=Path)
        if command=='score':c.add_argument('--predictions',required=True,type=Path)
        if command=='template':c.add_argument('--target',choices=['SENSOR'],default='SENSOR')
        if command=='prepare-video':
            c.add_argument('--source',required=True,type=Path)
            c.add_argument('--recording',help='Prepare one recording for a smoke check')
        if command=='run':
            c.add_argument('--inputs',required=True,type=Path)
            c.add_argument('--predictor',required=True,help='Local model adapter as module:function')
            c.add_argument('--name',required=True)
            c.add_argument('--target',choices=['SENSOR'],default='SENSOR')
    a = ap.parse_args(argv)
    try:
        if a.command=='run':
            from .job import run
            run(a.dataset,a.inputs,a.predictor,a.name,a.target,a.out)
        elif a.command=='score':
            result = score(a.dataset,a.predictions)
            save(a.out,result)
            print(json.dumps({'status':'SCORED','recordings':result['recordings'],'output':str(a.out)}))
        else:
            d,_ = dataset(a.dataset)
            if a.command=='verify':print(json.dumps({'status':'VERIFIED','recordings':len(d['records']),'sha256':sha(a.dataset)}))
            elif a.command=='template':
                save(a.out,dict(schema='onsetbench.predictions/1',dataset_sha256=sha(a.dataset),name='my-model',inputs='standard_video',
                               target=a.target,recordings={r['recording']:{'onsets_ms':[]} for r in d['records']}))
            else:
                from .video import prepare
                prepare(d,a.source,a.out,a.recording)
    except (ValueError,KeyError,TypeError,OSError) as e:
        print('ERROR: '+str(e),file=sys.stderr)
        raise SystemExit(2)
