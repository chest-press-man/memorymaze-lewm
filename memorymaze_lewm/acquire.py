"""Bounded, resumable extraction from the official public Drive ZIP archives."""
import concurrent.futures
import io
import struct
import time
import zipfile
import zlib
from pathlib import Path

import requests

from .io import sha256, write_json

SOURCES={
    'train':('1EdWafDjZG3VUo7CXTERx7fTx4IqWWNvC',10083118932),
    'eval':('18YPeLnu_TkVx7r4mZjeQ9gSMPEI-jtiP',3483085619),
}


class RemoteZip(io.RawIOBase):
    def __init__(self, source):
        ident,self.size=SOURCES[source]
        self.url=f'https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t'
        self.pos=0
        self.session=requests.Session()

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos

    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        if n>12_000_000:raise ValueError('Refusing unbounded ZIP request')
        start=self.pos;end=start+n-1
        for retry in range(4):
            try:
                r=self.session.get(self.url+f'&rangecheck={start}-{end}',headers={'Range':f'bytes={start}-{end}'},timeout=(15,90))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}' or len(r.content)!=n:
                    raise ValueError('Source changed or server did not honor byte range')
                self.pos+=n
                return r.content
            except (requests.RequestException,ValueError):
                if retry==3:raise
                time.sleep(2**retry)


def download_one(source,info,root):
    target=Path(root)/source/Path(info.filename).name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==info.file_size and zlib.crc32(target.read_bytes())==info.CRC:
        payload=None
    else:
        remote=RemoteZip(source);remote.seek(info.header_offset)
        header=remote.read(30)
        if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
        name_len,extra_len=struct.unpack_from('<HH',header,26)
        remote.seek(info.header_offset+30+name_len+extra_len)
        compressed=remote.read(info.compress_size)
        if info.compress_type==zipfile.ZIP_STORED:payload=compressed
        elif info.compress_type==zipfile.ZIP_DEFLATED:payload=zlib.decompress(compressed,-15)
        else:raise ValueError('Unsupported compression')
        if len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC:raise ValueError('ZIP member CRC mismatch')
        tmp=target.with_suffix('.part');tmp.write_bytes(payload);tmp.replace(target)
    from .data import load_episode
    episode=load_episode(target)
    return {'path':str(target.resolve()),'episode_id':target.stem,'source':source,
            'archive':SOURCES[source][0],'archive_name':info.filename,
            'bytes':target.stat().st_size,'sha256':sha256(target),'frames':len(episode['image'])}


def acquire(root,train_count=160,eval_count=30,eval_skip=10,workers=4):
    root=Path(root);records=[]
    for source,count,skip in [('train',train_count,0),('eval',eval_count,eval_skip)]:
        if count==0:continue
        remote=RemoteZip(source)
        with zipfile.ZipFile(remote) as archive:
            infos=sorted([i for i in archive.infolist() if i.filename.endswith('.npz')],key=lambda i:i.filename)[skip:skip+count]
        if len(infos)!=count:raise ValueError('Insufficient source episodes')
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i,record in enumerate(pool.map(lambda info:download_one(source,info,root),infos),1):
                records.append(record)
                write_json(root/'acquisition.json',{'source_folder':'https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB','episodes':records})
                print(f'{source} {i}/{count}: {record["episode_id"]}',flush=True)
    return records
