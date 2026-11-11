"""Read complete local-header ZIP members from an immutable downloaded prefix.

This proves member CRC integrity only, not whole-archive LFS SHA integrity.
Unsupported formats fail explicitly. Reads never cross the indexed snapshot.
"""
import os
from pathlib import Path
import struct
import zipfile
import zlib


class PrefixZip:
    integrity='member_crc_only_full_archive_sha_pending'

    def __init__(self,path):
        self.path=Path(path);self.entries={}
        path=self.path
        if not path.exists() and path.suffix=='.part':path=path.with_suffix('')
        with path.open('rb') as stream:
            self.snapshot_size=os.fstat(stream.fileno()).st_size
            while stream.tell()+30<=self.snapshot_size:
                offset=stream.tell();header=stream.read(30)
                if header[:4] in [b'PK\x01\x02',b'PK\x05\x06',b'PK\x06\x06']:
                    break
                if header[:4]!=b'PK\x03\x04':
                    raise ValueError(f'Invalid ZIP local header at {offset}')
                _,_,flags,method,_,_,crc,compressed,size,nlen,xlen=struct.unpack('<IHHHHHIIIHH',header)
                if flags&9:
                    raise ValueError('Encrypted/data-descriptor ZIP requires complete archive')
                if stream.tell()+nlen+xlen>self.snapshot_size:break
                name=stream.read(nlen).decode('utf-8' if flags&0x800 else 'cp437')
                extra=stream.read(xlen)
                if compressed==0xffffffff or size==0xffffffff:
                    pos=0;found=False
                    while pos+4<=len(extra):
                        kind,length=struct.unpack_from('<HH',extra,pos);pos+=4
                        if kind==1:
                            values=struct.unpack('<'+'Q'*(length//8),extra[pos:pos+length]);i=0
                            if size==0xffffffff:size=values[i];i+=1
                            if compressed==0xffffffff:compressed=values[i]
                            found=True;break
                        pos+=length
                    if not found:raise ValueError('Missing ZIP64 sizes')
                data_offset=stream.tell()
                if data_offset+compressed>self.snapshot_size:break
                if not name.endswith('/'):
                    if name in self.entries:raise ValueError('Duplicate ZIP member')
                    info=zipfile.ZipInfo(name);info.CRC=crc;info.compress_type=method
                    info.compress_size=compressed;info.file_size=size;info.header_offset=data_offset
                    self.entries[name]=info
                stream.seek(compressed,1)

    def infolist(self):return self.entries.values()

    def read(self,member):
        info=self.entries[member if isinstance(member,str) else member.filename]
        if info.file_size>128*1024*1024:raise ValueError('Unexpected giant annotation member')
        path=self.path
        if not path.exists() and path.suffix=='.part':path=path.with_suffix('')
        with path.open('rb') as stream:
            stream.seek(info.header_offset);raw=stream.read(info.compress_size)
        if len(raw)!=info.compress_size:raise IOError('Indexed ZIP prefix shrank')
        if info.compress_type==0:data=raw
        elif info.compress_type==8:data=zlib.decompress(raw,-15)
        else:raise ValueError('Unsupported ZIP compression')
        if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:
            raise ValueError('ZIP member CRC failure')
        return data
