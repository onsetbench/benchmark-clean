"""Create and validate reference-free recording-relative submissions."""
import argparse
from pathlib import Path
import numpy as np
from .cli import load, numbers, save, sha

CLOCK = 'recording_relative_ms'

def geometry(path):
    d = load(path)
    if d.get('schema') != 'onsetbench.dataset/1' or d.get('clock') != CLOCK or d.get('inputs') != 'standard_video':
        raise ValueError('Expected a recording-relative standard-video manifest')
    records = d.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('No recording grid')
    ids = set()
    for r in records:
        rid = r.get('recording')
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError('Invalid or duplicate recording')
        ids.add(rid)
        starts = numbers(r.get('starts_ms'), 'starts_ms')
        span = r.get('span_ms')
        if not len(starts) or np.any(np.diff(starts) < 0) or isinstance(span, bool) or not isinstance(span, (int, float)) or not np.isfinite(span) or span <= 0:
            raise ValueError('Invalid retained scoring grid')
    return d

def template(dataset_path, name):
    d = geometry(dataset_path)
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Supply an entry name and revision')
    return dict(schema='onsetbench.predictions/1', dataset_sha256=sha(dataset_path),
                name=name, inputs='standard_video', target='SENSOR',
                recordings={r['recording']: {'onsets_ms': []} for r in d['records']})

def validate(dataset_path, predictions_path):
    d = geometry(dataset_path)
    p = load(predictions_path)
    if p.get('schema') != 'onsetbench.predictions/1' or p.get('dataset_sha256') != sha(dataset_path):
        raise ValueError('Prediction schema or dataset checksum differs')
    if p.get('inputs') != 'standard_video' or p.get('target') != 'SENSOR' or not isinstance(p.get('name'), str) or not p['name'].strip():
        raise ValueError('Expected named standard-video SENSOR submission')
    rows = p.get('recordings')
    grid = {r['recording']: r for r in d['records']}
    if not isinstance(rows, dict) or set(rows) != set(grid):
        raise ValueError('Include every recording exactly once')
    for rid, row in rows.items():
        if not isinstance(row, dict) or set(row) - {'onsets_ms', 'midpoints_ms', 'contact'}:
            raise ValueError('Unknown prediction field')
        events = numbers(row.get('onsets_ms'), 'onsets_ms')
        if 'midpoints_ms' in row:
            mid = numbers(row['midpoints_ms'], 'midpoints_ms')
            if len(mid) != len(events):
                raise ValueError('One midpoint required per emitted event')
        if 'contact' in row:
            dec = numbers(row['contact'], 'contact')
            if len(dec) != len(grid[rid]['starts_ms']) or not np.isin(dec, [-1, 0, 1]).all():
                raise ValueError('One -1, 0 or 1 decision required per grid entry')
    for field in ('midpoints_ms', 'contact'):
        flags = [field in row for row in rows.values()]
        if any(flags) and not all(flags):
            raise ValueError(field + ' must cover all recordings or none')
    return dict(status='VALIDATED', clock=CLOCK, recordings=len(rows),
                predictions_sha256=sha(predictions_path), dataset_sha256=sha(dataset_path))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    t = sub.add_parser('template')
    t.add_argument('--dataset', required=True, type=Path)
    t.add_argument('--name', required=True)
    t.add_argument('--out', required=True, type=Path)
    v = sub.add_parser('validate')
    v.add_argument('--dataset', required=True, type=Path)
    v.add_argument('--predictions', required=True, type=Path)
    a = ap.parse_args()
    if a.command == 'template':
        save(a.out, template(a.dataset, a.name))
    else:
        import json
        print(json.dumps(validate(a.dataset, a.predictions)))

if __name__ == '__main__':
    main()
