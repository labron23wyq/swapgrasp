"""Seekable read-only HF range reader for selecting a diagnostic image subset.

ZIP CRC checks individual extracted files; this does NOT claim full LFS SHA
verification. Full archives are downloaded and verified independently.
"""
from collections import OrderedDict
import io
import hashlib
from pathlib import Path
import tempfile
import time
import requests


class RemoteConcat(io.RawIOBase):
    def __init__(self, segments, endpoint='https://hf-mirror.com',cache_dir=None,local_only=False):
        super().__init__(); self.segments=segments;self.endpoint=endpoint
        self.total=sum(s['size'] for s in segments);self.position=0;self.cache=OrderedDict()
        self.downloaded_bytes=0
        self.local_only=local_only
        self.session=requests.Session()
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.cache_dir:self.cache_dir.mkdir(parents=True,exist_ok=True)

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        if whence==0:self.position=offset
        elif whence==1:self.position+=offset
        elif whence==2:self.position=self.total+offset
        else:raise ValueError('Invalid whence')
        if self.position<0:raise ValueError('Negative seek')
        return self.position

    def fetch(self,segment,start,end):
        key=(segment['repo'],segment['path'],start,end)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        local=segment.get('local_path')
        if local and Path(local).is_file() and Path(local).stat().st_size>end:
            with open(local,'rb') as stream:stream.seek(start);return stream.read(end-start+1)
        if self.local_only:raise OSError(f'Incomplete local segment: {segment["path"]}')
        disk=None
        if self.cache_dir:
            disk=self.cache_dir/hashlib.sha256(repr((segment['revision'],key)).encode()).hexdigest()
            if disk.is_file() and disk.stat().st_size==end-start+1:return disk.read_bytes()
        url=f'{self.endpoint}/datasets/{segment["repo"]}/resolve/{segment["revision"]}/{segment["path"]}?download=true&offset={start}&range_end={end}'
        for attempt in range(12):
            try:
                with self.session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90)) as r:
                    r.raise_for_status()
                    expected=f'bytes {start}-{end}/{segment["size"]}'
                    if r.status_code!=206 or r.headers.get('Content-Range')!=expected:
                        raise IOError('Invalid server range response')
                    data=r.content
                    if len(data)!=end-start+1:raise IOError('Truncated range')
                    self.downloaded_bytes+=len(data)
                    if disk:
                        # Concurrent preparation jobs may fetch the same range.
                        # Each writer owns its temporary file; replacement is atomic.
                        with tempfile.NamedTemporaryFile(dir=disk.parent,delete=False) as stream:
                            temporary=Path(stream.name);stream.write(data)
                        try:temporary.replace(disk)
                        finally:temporary.unlink(missing_ok=True)
                    if len(data)<=1024*1024:
                        self.cache[key]=data
                        while len(self.cache)>8:self.cache.popitem(last=False)
                    return data
            except (requests.RequestException,IOError) as error:
                if attempt==11:raise
                delay=min(120,5*2**min(attempt,5))
                if isinstance(error,requests.HTTPError) and error.response is not None:
                    retry=error.response.headers.get('Retry-After','')
                    if retry.isdigit():delay=max(delay,min(60,int(retry)))
                    if error.response.status_code==429:
                        delay=min(300,max(delay,90*(attempt+1)))
                print('RANGE RETRY',segment['path'],start,end,attempt+1,type(error).__name__,'wait',delay,flush=True)
                time.sleep(delay)

    def read(self,n=-1):
        n=min(self.total-self.position,n) if n>=0 else self.total-self.position
        if n<=0:return b''
        remaining=n; chunks=[];base=0
        for segment in self.segments:
            if self.position>=base+segment['size']:
                base+=segment['size'];continue
            start=self.position-base;length=min(remaining,segment['size']-start)
            # Fetch exact requested ranges: ZIP central directory can be large.
            end=start+length-1
            if length<1024*1024 and not self.local_only:
                block_start=(start//(1024*1024))*(1024*1024)
                block_end=min(segment['size']-1,max(end,block_start+1024*1024-1))
                block=self.fetch(segment,block_start,block_end)
                chunks.append(block[start-block_start:start-block_start+length])
            else:chunks.append(self.fetch(segment,start,end))
            self.position+=length;remaining-=length
            if not remaining:break
            base+=segment['size']
        return b''.join(chunks)


def segments_from_catalog(catalog,repo,paths):
    cat=catalog[repo]
    return [{'repo':repo,'revision':cat['revision'],'path':p,'size':cat['files'][p]['size']} for p in paths]

