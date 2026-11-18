from pathlib import Path
import requests
from .io import sha256,write_json

REVISION='b0747c5002e86d2ce8f3cd8178004b97524c587d'
WEIGHTS_SHA256='2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89'


def download_cube(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sources=[]
    for name in ['config.json','weights.pt']:
        target=out/name;url=f'https://huggingface.co/quentinll/lewm-cube/resolve/{REVISION}/{name}'
        if not target.exists():
            with requests.get(url,stream=True,timeout=(15,120)) as response:
                response.raise_for_status()
                with target.with_suffix('.part').open('wb') as f:
                    for chunk in response.iter_content(1024*1024):f.write(chunk)
            target.with_suffix('.part').replace(target)
        digest=sha256(target)
        if name=='weights.pt' and digest!=WEIGHTS_SHA256:raise ValueError('Official weight hash mismatch')
        sources.append({'file':name,'url':url,'sha256':digest})
    write_json(out/'source.json',sources)
    return sources
