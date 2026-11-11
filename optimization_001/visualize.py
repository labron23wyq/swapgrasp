import argparse
from swapgrasp.data import ManifestDataset,read_jsonl
from swapgrasp.visualization import gallery,curves

p=argparse.ArgumentParser(); p.add_argument('--manifest'); p.add_argument('--split',default='test')
p.add_argument('--root'); p.add_argument('--predictions'); p.add_argument('--log'); p.add_argument('--output',required=True)
a=p.parse_args()
if a.log:
    curves(a.log,a.output)
else:
    dataset=ManifestDataset(a.manifest,a.split,root=a.root)
    gallery(dataset,read_jsonl(a.predictions) if a.predictions else [],a.output)
