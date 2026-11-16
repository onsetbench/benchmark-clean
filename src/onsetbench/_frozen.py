"""Unchanged public scoring functions. See SCORER_PROVENANCE.json."""

import sys
import numpy as np
S = P = sys.modules[__name__]

STATE_KEYS = ('tp_c','n_c','tp_n','n_n','n_dec','n_win','ref_c','ref_n')

def match_events(truth, pred, tol):
    """One-to-one, in block, max cardinality then min total |lag|.

    Both lists sorted. DP over (i, j) with skips; returns the matched pairs as
    (truth index, pred index, lag = pred - truth)."""
    n, m = len(truth), len(pred)
    if n == 0 or m == 0:
        return []
    dp_m = [[0] * (m + 1) for _ in range(n + 1)]
    dp_l = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row_m, row_l = dp_m[i], dp_l[i]
        nxt_m, nxt_l = dp_m[i + 1], dp_l[i + 1]
        ti = truth[i]
        for j in range(m - 1, -1, -1):
            bm, bl = nxt_m[j], nxt_l[j]                      # skip truth i
            cm, cl = row_m[j + 1], row_l[j + 1]              # skip pred j
            if cm > bm or (cm == bm and cl < bl):
                bm, bl = cm, cl
            d = abs(ti - pred[j])
            if d <= tol:
                cm, cl = nxt_m[j + 1] + 1, nxt_l[j + 1] + d
                if cm > bm or (cm == bm and cl < bl):
                    bm, bl = cm, cl
            row_m[j], row_l[j] = bm, bl
    pairs = []
    i = j = 0
    while i < n and j < m:
        bm, bl = dp_m[i][j], dp_l[i][j]
        d = abs(truth[i] - pred[j])
        if d <= tol and dp_m[i + 1][j + 1] + 1 == bm and abs(dp_l[i + 1][j + 1] + d - bl) < 1e-9:
            pairs.append((i, j, float(pred[j] - truth[i])))
            i += 1
            j += 1
        elif dp_m[i + 1][j] == bm and abs(dp_l[i + 1][j] - bl) < 1e-9:
            i += 1
        else:
            j += 1
    return pairs

def window_labels(starts, spans, ref):
    """The centre rule. Returns float y in {0, 1}, or None for a spanless target."""
    if ref["spans"] is None:
        return None
    centres = starts + spans / 2.0
    y = np.zeros(centres.size, dtype="float64")
    for a, b in ref["spans"]:
        y[(centres >= a) & (centres <= b)] = 1.0
    return y

def union_ms(starts, spans):
    """Union of [start, start + span) in ms."""
    if starts.size == 0:
        return 0.0
    order = np.argsort(starts, kind="stable")
    a, b = starts[order], starts[order] + spans[order]
    total = 0.0
    cur_a, cur_b = a[0], b[0]
    for i in range(1, a.size):
        if a[i] > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a[i], b[i]
        else:
            cur_b = max(cur_b, b[i])
    return float(total + (cur_b - cur_a))

def controlled_times(native, midpoint, lo, hi):
    x = np.asarray(native, dtype=float)
    if hi <= lo or not np.isfinite([lo, hi]).all() or not np.isfinite(x).all():
        raise ValueError('invalid scoring block or predictions')
    out = {'native': sorted(x.tolist())}
    if midpoint is not None:
        if len(midpoint) != len(x) or not np.isfinite(midpoint).all():
            raise ValueError('missing or invalid source-window identity')
        out['midpoint'] = sorted(map(float, midpoint))
    for pct in (25, 50, 75):
        out[f'shift_{pct}'] = sorted((lo + (x-lo+(hi-lo)*pct/100) % (hi-lo)).tolist())
    return out

def score_block(truth, pred, tol):
    truth, pred = sorted(map(float, truth)), sorted(map(float, pred))
    if not np.isfinite(truth + pred).all():
        raise ValueError('nonfinite event time')
    pairs = S.match_events(truth, pred, tol)
    return {'n_truth': len(truth), 'n_emitted': len(pred), 'tp': len(pairs),
            'fp': len(pred)-len(pairs), 'fn': len(truth)-len(pairs),
            'lags_ms': [p[2] for p in pairs]}

def summarize(rows, hours=None):
    out = {k: sum(r[k] for r in rows) for k in ('n_truth', 'n_emitted', 'tp', 'fp', 'fn')}
    t, p, m = out['n_truth'], out['n_emitted'], out['tp']
    out.update(recall=m/t if t else None, precision=m/p if p else None,
               f1=2*m/(t+p) if t+p else None,
               false_per_hour=out['fp']/hours if hours else None)
    lag = [abs(d) for r in rows for d in r['lags_ms']]
    out.update(lag_median_ms=float(np.median(lag)) if lag else None,
               lag_p90_ms=float(np.percentile(lag, 90)) if lag else None)
    return out

def contrast(groups):
    a = np.asarray([[groups[k][v] for v in ('truth', 'native', 'control')]
                    for k in sorted(groups)], dtype=float)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(a), (10000, len(a)))
    counts = a[draws].sum(axis=1)
    valid = counts[:, 0] > 0
    diffs = (counts[valid, 1]-counts[valid, 2])/counts[valid, 0]
    return {'difference': float((a[:, 1].sum()-a[:, 2].sum())/a[:, 0].sum()),
            'ci95': np.percentile(diffs, [2.5, 97.5]).tolist() if len(diffs) else None,
            'undefined_draws': int((~valid).sum()), 'n_clusters': len(a)}

def require(ok,message):
    if not ok:raise ValueError(message)

def integers(row,keys):
    vals=[row[k] for k in keys]
    require(all(isinstance(x,(int,float,np.integer,np.floating)) and np.isfinite(x) and x>=0 and x==int(x) for x in vals),'invalid integer counts')
    return {k:int(v) for k,v in zip(keys,vals)}

def state_counts(dec,labels):
    d=np.asarray(dec);y=np.asarray(labels)
    require(d.ndim==y.ndim==1 and d.shape==y.shape and np.isin(d,[-1,0,1]).all() and np.isin(y,[0,1]).all(),'invalid state arrays')
    return dict(tp_c=int(((d==1)&(y==1)).sum()),n_c=int((d==1).sum()),tp_n=int(((d==-1)&(y==0)).sum()),n_n=int((d==-1).sum()),
                n_dec=int((d!=0).sum()),n_win=len(d),ref_c=int((y==1).sum()),ref_n=int((y==0).sum()))

def state_component(row):
    c=integers(row,STATE_KEYS)
    require(c['n_c']+c['n_n']==c['n_dec']<=c['n_win'] and c['ref_c']+c['ref_n']==c['n_win'] and
            c['tp_c']<=min(c['n_c'],c['ref_c']) and c['tp_n']<=min(c['n_n'],c['ref_n']) and
            c['n_c']-c['tp_c']<=c['ref_n']-c['tp_n'] and c['n_n']-c['tp_n']<=c['ref_c']-c['tp_c'],'inconsistent state counts')
    ratio=lambda a,b:a/b if b else None
    cc=ratio(c['tp_c'],c['ref_c']);nc=ratio(c['tp_n'],c['ref_n'])
    return dict(counts=c,balanced_correct_class_coverage=None if cc is None or nc is None else .5*(cc+nc),
                contact_class_coverage=cc,nocontact_class_coverage=nc,precision_contact=ratio(c['tp_c'],c['n_c']),
                precision_nocontact=ratio(c['tp_n'],c['n_n']),decision_coverage=ratio(c['n_dec'],c['n_win']),constant_class_baseline=.5)

def paired_contrast(native,controls):
    ids=sorted(native)
    a=np.array([[native[r]['n_truth'],native[r]['n_emitted'],native[r]['tp']] for r in ids],dtype=float)
    cs=[np.array([[c[r]['n_truth'],c[r]['n_emitted'],c[r]['tp']] for r in ids],dtype=float) for c in controls]
    if any(not np.array_equal(a[:,0],c[:,0]) for c in cs):
        raise ValueError('paired truth mismatch')
    draws=np.random.default_rng(0).integers(0,len(ids),(10000,len(ids)))
    def values(x):
        sums=x[draws].sum(axis=1)
        total=x.sum(axis=0)
        with np.errstate(divide='ignore',invalid='ignore'):
            return (total[2]/total[0],2*total[2]/(total[0]+total[1])),(sums[:,2]/sums[:,0],2*sums[:,2]/(sums[:,0]+sums[:,1]))
    n,nb=values(a)
    cv=[values(c) for c in cs]
    out={}
    for k,name in enumerate(('recall','f1')):
        diff=nb[k]-np.mean([v[1][k] for v in cv],axis=0)
        valid=np.isfinite(diff)
        point=n[k]-np.mean([v[0][k] for v in cv])
        out[name]=dict(difference=float(point) if np.isfinite(point) else None,
            ci95=np.percentile(diff[valid],[2.5,97.5]).tolist() if valid.any() else None,
            undefined_draws=int((~valid).sum()),n_clusters=len(ids))
    # Verify the established recording-bootstrap recall implementation too.
    if a[:,0].sum():
        legacy=P.contrast({r:dict(truth=a[i,0],native=a[i,2],control=np.mean([c[i,2] for c in cs])) for i,r in enumerate(ids)})
        if not np.allclose(out['recall']['ci95'],legacy['ci95'],atol=1e-14):
            raise ValueError('recording contrast parity mismatch')
    return out
