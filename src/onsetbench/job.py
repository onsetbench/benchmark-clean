"""Local entrant job: video -> predictions -> sealed file -> scoring."""
import importlib
from pathlib import Path
import time

from .cli import contained,load,numbers,save,score,sha


def run(dataset_path,inputs_path,predictor,name,target,out):
    if target != 'SENSOR':
        raise ValueError('Target must be SENSOR')
    # Read input geometry, not reference files, before predictions are sealed.
    d = load(dataset_path)
    if d.get('schema')!='onsetbench.dataset/1':raise ValueError('Unsupported dataset')
    inp = load(inputs_path)
    if inp.get('schema')!='onsetbench.inputs/1' or inp.get('dataset_id')!=d['dataset_id'] or inp.get('inputs')!='standard_video' or not inp.get('complete'):
        raise ValueError('A complete matching standard-video input package is required')
    rows = inp['recordings']
    ids = [r['recording'] for r in rows]
    geometry = {r['recording']:r for r in d['records']}
    if len(ids)!=len(set(ids)) or set(ids)!=set(geometry):raise ValueError('Input recording roster differs')
    checked = {}
    base = Path(inputs_path).resolve().parent
    for r in rows:
        video = contained(base,r['video'])
        clock = contained(base,r['timestamps'])
        if sha(video)!=r['video_sha256'] or sha(clock)!=r['timestamps_sha256']:
            raise ValueError('Video input checksum differs')
        stamps = load(clock)
        if stamps.get('clock')!='epoch_ms' or stamps.get('recording')!=r['recording'] or stamps.get('video_sha256')!=r['video_sha256']:
            raise ValueError('Frame clock identity differs')
        times = numbers(stamps['frame_timestamps_ms'],'frame timestamps')
        if len(times)!=r['frames']:raise ValueError('Frame timestamp count differs')
        checked[r['recording']] = (video,times.tolist())
    module,sep,function = predictor.partition(':')
    if not sep:raise ValueError('Predictor must be module:function')
    predict = getattr(importlib.import_module(module),function)
    if not callable(predict):raise ValueError('Predictor is not callable')
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    predictions = {}
    for rid in sorted(geometry):
        video,times = checked[rid]
        g = geometry[rid]
        predictions[rid] = predict(video=video,frame_timestamps_ms=times,
                                   window_starts_ms=g['starts_ms'],window_span_ms=g['span_ms'])
        print('Predicted '+rid,flush=True)
    pred_path = out/'predictions.json'
    save(pred_path,dict(schema='onsetbench.predictions/1',dataset_sha256=sha(dataset_path),name=name,
                        inputs='standard_video',target=target,recordings=predictions))
    # This is written before score() opens any reference file.
    save(out/'prediction_seal.json',dict(predictions_sha256=sha(pred_path),dataset_sha256=sha(dataset_path),
         inputs_sha256=sha(inputs_path),predictor=predictor,inference_wall_seconds=time.monotonic()-started))
    result = score(dataset_path,pred_path)
    save(out/'score.json',result)
    print('Scored '+str(len(predictions))+' recordings',flush=True)
