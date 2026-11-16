"""Extract the public wrist-camera video and its native clock from REASSEMBLE."""
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import zlib

import numpy as np

from .cli import save,sha


def identity(path):
    digest = hashlib.sha256()
    crc = 0
    with Path(path).open('rb') as f:
        while block := f.read(8*1024*1024):
            digest.update(block)
            crc = zlib.crc32(block,crc)
    return digest.hexdigest(),format(crc,'08x')


def extract(record,source,out):
    try:
        import h5py
        import cv2
    except ImportError as e:
        raise ValueError('Install the video extra: pip install ".[video]"') from e
    source_sha,crc = identity(source)
    expected = record['source']
    if Path(source).stat().st_size!=expected['bytes'] or crc!=expected['crc32']:
        raise ValueError('Source recording differs from the upstream archive: '+record['recording'])
    rid = record['recording']
    out.mkdir(parents=True,exist_ok=True)
    video,clock = out/(rid+'.mp4'),out/(rid+'.timestamps.json')
    if video.exists() or clock.exists():
        raise ValueError('Output exists: '+rid)
    with h5py.File(source,'r') as f:
        if 'hand' not in f or 'timestamps/hand' not in f:
            raise ValueError('Wrist-camera video or timestamps missing')
        times = np.asarray(f['timestamps/hand'],dtype=float)*1000
        if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<0) or times.min()<1e12:
            raise ValueError('Invalid native camera clock')
        payload = bytes(f['hand'][()])
    with video.open('xb') as f:
        f.write(payload)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Video cannot be decoded')
        count = 0
        while True:
            ok,_ = cap.read()
            if not ok:break
            count += 1
    finally:
        cap.release()
    if count!=len(times):
        raise ValueError(f'Frame clock mismatch for {rid}: {count} decoded frames / {len(times)} timestamps')
    save(clock,dict(recording=rid,clock='epoch_ms',frame_timestamps_ms=times.tolist(),video_sha256=sha(video)))
    return dict(recording=rid,video=video.name,timestamps=clock.name,frames=count,
                video_sha256=sha(video),timestamps_sha256=sha(clock),source_sha256=source_sha,
                source_crc32=crc,camera='hand',inputs='standard_video')


def prepare(d,source,out,recording=None):
    records = [r for r in d['records'] if recording is None or r['recording']==recording]
    if not records:raise ValueError('Recording is outside the public dataset')
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    manifest = out/'inputs.json'
    if manifest.exists():raise ValueError('Input manifest already exists')
    prepared = []
    if source.is_file() and source.suffix=='.zip':
        digest = hashlib.md5()
        with source.open('rb') as f:
            while b := f.read(8*1024*1024):digest.update(b)
        if source.stat().st_size!=d['source_archive']['archive_bytes'] or digest.hexdigest()!=d['source_archive']['archive_md5']:
            raise ValueError('Upstream archive checksum differs')
        with zipfile.ZipFile(source) as z:
            for r in records:
                spec = r['source']
                entry = z.getinfo(spec['path'])
                if entry.file_size!=spec['bytes'] or format(entry.CRC,'08x')!=spec['crc32']:
                    raise ValueError('Upstream archive entry differs')
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td)/(r['recording']+'.h5')
                    with z.open(entry) as inp,p.open('xb') as dest:
                        while b := inp.read(8*1024*1024):dest.write(b)
                    prepared.append(extract(r,p,out))
                print('Prepared '+r['recording'],flush=True)
    else:
        paths = {r['recording']:source/(r['recording']+'.h5') for r in records}
        missing = [r for r,p in paths.items() if not p.is_file()]
        if missing:raise ValueError(f'Missing {len(missing)} source recordings; first: {missing[0]}')
        for r in records:
            prepared.append(extract(r,paths[r['recording']],out))
            print('Prepared '+r['recording'],flush=True)
    save(manifest,dict(schema='onsetbench.inputs/1',dataset_id=d['dataset_id'],camera=d['camera'],clock='epoch_ms',
                       inputs='standard_video',complete=len(records)==len(d['records']),recordings=prepared))
