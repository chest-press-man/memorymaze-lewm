"""Revalidate downloaded or transferred NPZs using their recorded membership."""
from collections import Counter
from pathlib import Path

import numpy as np

from .data import load_episode
from .io import read_json,sha256,write_json


def validate_data(data_root,reference_path=None):
    root=Path(data_root);records=read_json(root/'acquisition.json')['episodes']
    if not records:raise ValueError('Empty acquisition manifest')
    if reference_path is not None:
        reference={r['episode_id']:r['sha256'] for r in read_json(reference_path)['episodes']}
        current={r['episode_id']:r['sha256'] for r in records}
        if any(current.get(k)!=v for k,v in reference.items()):raise ValueError('Shared files differ from the baseline reference')
    seen=set();total=0
    for record in records:
        if record['source'] not in ['train','eval']:raise ValueError('Unknown data source')
        path=root/record['source']/Path(record['path']).name
        if record['episode_id'] in seen:raise ValueError('Duplicate episode ID')
        seen.add(record['episode_id'])
        if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise ValueError(f'size/hash mismatch for {path.name}')
        d=load_episode(path);n=len(d['image'])
        if n!=1001 or record['frames']!=n:raise ValueError('Unexpected episode length')
        if d['reset'].shape!=(n,) or d['terminal'].shape!=(n,):raise ValueError('Invalid boundary flag shape')
        if np.any(d['reset'][1:]) or np.any(d['terminal'][:-1]):raise ValueError('Unexpected internal episode boundary')
        total+=path.stat().st_size
    result={'episodes':len(records),'sources':dict(Counter(r['source'] for r in records)),
            'bytes':total,'sha256_mismatches':0,'schema_errors':0,
            'frames_per_episode':1001,'transitions_per_episode':1000,
            'path_policy':'relative to current data root; source manifest paths may belong to the acquisition host'}
    write_json(root/'validation_report.json',result)
    return result
