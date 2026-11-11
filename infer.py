"""Predict grasps from one image and instruction, without annotations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--text',required=True,help='English grasp instruction')
    parser.add_argument('--output',required=True,help='Directory for JSON, overlays, heatmap and HTML')
    parser.add_argument('--device',help='cuda, cuda:0 or cpu; defaults to automatic selection')
    args=parser.parse_args()
    image=Path(args.image).resolve();checkpoint=Path(args.checkpoint).resolve()
    if not image.is_file():parser.error(f'Image not found: {image}')
    if not checkpoint.is_file():parser.error(f'Checkpoint not found: {checkpoint}')
    if not args.text.strip():parser.error('Instruction must not be empty')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    row={'sample_id':'user_query','image_id':image.stem,'image':str(image),
         'text':args.text,'split':'inference'}
    with TemporaryDirectory(prefix='swapgrasp_infer_') as temporary:
        manifest=Path(temporary)/'request.jsonl'
        manifest.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
        command=[sys.executable,'evaluate.py','predict','--checkpoint',str(checkpoint),
                 '--manifest',str(manifest),'--split','inference','--batch-size','1',
                 '--output',str(output/'predictions.jsonl'),'--visualize-dir',str(output)]
        if args.device:command.extend(['--device',args.device])
        subprocess.run(command,cwd=Path(__file__).resolve().parent,check=True)
    prediction=json.loads((output/'predictions.jsonl').read_text(encoding='utf-8'))
    (output/'request.json').write_text(json.dumps({**row,'checkpoint':str(checkpoint)},
                                                ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(prediction,indent=2))
    print('Original-image pixels: x, y, width, height; clockwise radians modulo pi.')
    print('Red: top-1; amber: remaining candidates. Scores are ranking responses.')
    print('Visualization:',output/'index.html')


if __name__=='__main__':main()
