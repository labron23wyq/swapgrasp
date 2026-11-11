import io
import math
import zipfile
import numpy as np
import pytest
import torch
from prepare_real import canonical_grasps,target_key
from swapgrasp.remote_zip import RemoteConcat


def test_real_label_contract_and_clockwise_canonical_angle():
    data=torch.tensor([[.5,100.,120.,40.,20.,150.]])
    g=canonical_grasps(data)[0]
    assert np.allclose(g[:4],[100,120,40,20])
    assert abs(g[4]-math.radians(-30))<1e-6
    with pytest.raises(ValueError):canonical_grasps(torch.zeros(2,5))
    with pytest.raises(ValueError):canonical_grasps(torch.tensor([[.5,1,2,-1,4,0]]))
    image='a'*64
    assert target_key(f'positive_grasp/{image}_12_34.pt')==(image,12,34)


def test_split_zip_range_reader_handles_cross_file_members():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('image/a.jpg',b'example image'*100)
        z.writestr('image/b.jpg',b'other image'*300)
    archive=buf.getvalue();cut=len(archive)//2
    class FakeRemote(RemoteConcat):
        def fetch(self,segment,start,end):
            return segment['bytes'][start:end+1]
    reader=FakeRemote([{'size':cut,'bytes':archive[:cut]},{'size':len(archive)-cut,'bytes':archive[cut:]}])
    with zipfile.ZipFile(reader) as z:
        assert z.read('image/a.jpg')==b'example image'*100
        assert z.read('image/b.jpg')==b'other image'*300


def test_partial_annotation_zip_reads_only_complete_crc_valid_members(tmp_path):
    from swapgrasp.prefix_zip import PrefixZip
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('labels/a.pt',b'first annotation')
        z.writestr('labels/b.pt',b'second annotation')
    data=archive.getvalue();cut=data.index(b'PK\x03\x04',4)+35
    path=tmp_path/'labels.zip.part';path.write_bytes(data[:cut])
    z=PrefixZip(path)
    assert [i.filename for i in z.infolist()]==['labels/a.pt']
    assert z.read('labels/a.pt')==b'first annotation'
    with pytest.raises(KeyError):z.read('labels/b.pt')
    # Keep the parser contract intact but alter the expected CRC.
    z.entries['labels/a.pt'].CRC^=1
    with pytest.raises(ValueError,match='CRC'):z.read('labels/a.pt')

