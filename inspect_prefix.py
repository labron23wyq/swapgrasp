"""Inspect complete leading ZIP members while an archive is downloading.

Only use on archives from authorized official repositories. Per-member CRC is
checked, but this is explicitly preliminary until whole-archive SHA passes.
"""
import io
import json
from pathlib import Path
import pickle
import struct
import zlib
import numpy as np
import torch


def prefix_entries(path,limit=5):
    with open(path,'rb') as f:
        count=0
        while count<limit:
            header=f.read(30)
            if len(header)<30 or header[:4]!=b'PK\x03\x04':return
            _,version,flags,method,_,_,crc,compressed,uncompressed,nlen,xlen=struct.unpack('<IHHHHHIIIHH',header)
            name=f.read(nlen).decode('utf-8');extra=f.read(xlen)
            if flags & 8: raise ValueError('Data-descriptor ZIP requires central directory')
            if compressed==0xffffffff or uncompressed==0xffffffff:
                pos=0
                while pos+4<=len(extra):
                    kind,length=struct.unpack_from('<HH',extra,pos);pos+=4
                    if kind==1:
                        vals=struct.unpack('<'+'Q'*(length//8),extra[pos:pos+length]);j=0
                        if uncompressed==0xffffffff:uncompressed=vals[j];j+=1
                        if compressed==0xffffffff:compressed=vals[j]
                        break
                    pos+=length
            if compressed>100_000_000:raise ValueError('Unexpected giant member')
            raw=f.read(compressed)
            if len(raw)<compressed:return
            if method==8:data=zlib.decompress(raw,-15)
            elif method==0:data=raw
            else:raise ValueError('Unsupported ZIP compression')
            if len(data)!=uncompressed or zlib.crc32(data)&0xffffffff!=crc:raise ValueError('ZIP member CRC failure')
            if not name.endswith('/'):
                yield name,data
                count+=1


def describe(value):
    if isinstance(value,(np.ndarray,torch.Tensor)):
        if isinstance(value,np.ndarray) and value.dtype==object:
            return {'type':'object_array','shape':list(value.shape),'value':repr(value.tolist())[:5000]}
        arr=np.asarray(value)
        return {'type':type(value).__name__,'shape':list(arr.shape),'dtype':str(arr.dtype),
                'min':float(arr.min()) if arr.size else None,'max':float(arr.max()) if arr.size else None,
                'first':arr.reshape(-1)[:18].tolist()}
    if isinstance(value,dict):return {str(k):describe(v) for k,v in list(value.items())[:6]}
    return {'type':type(value).__name__,'value':repr(value)[:5000]}


if __name__=='__main__':
    output=[]
    for glob in ['Grasp-Anything-pp/grasp_instructions.zip*','Grasp-Anything-pp/grasp_label_positive.zip*',
                 'Grasp-Anything-pp/part_mask.zip*','Grasp-Anything/mask.zip*']:
        for path in Path('data/downloads').glob(glob):
            if path.suffix=='.json':continue
            try:
                for name,data in prefix_entries(path,3):
                    if name.endswith('.pkl'):value=pickle.loads(data)
                    elif name.endswith('.npy'):value=np.load(io.BytesIO(data),allow_pickle=True)
                    elif name.endswith('.pt'):value=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
                    else:continue
                    output.append({'archive':str(path),'member':name,'preliminary_crc_only':True,'value':describe(value)})
            except Exception as e:output.append({'archive':str(path),'error':repr(e)})
    Path('artifacts/real_prefix_inspection.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)
