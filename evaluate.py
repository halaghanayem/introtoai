"""Evaluate independent, single-face probe images. See README before using."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from recognition import FaceEngine
from storage import Store, now

ROOT = Path(__file__).resolve().parent


def metrics(rows):
    classes = sorted({r['true_id'] for r in rows})
    per_class = {}
    for label in classes:
        tp=sum(r['true_id']==label and r['predicted_id']==label for r in rows)
        fp=sum(r['true_id']!=label and r['predicted_id']==label for r in rows)
        fn=sum(r['true_id']==label and r['predicted_id']!=label for r in rows)
        p=tp/(tp+fp) if tp+fp else 0
        r=tp/(tp+fn) if tp+fn else 0
        per_class[label]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,
                          'support':sum(x['true_id']==label for x in rows)}
    count=len(rows)
    known=[r for r in rows if r['true_id']!='unknown']
    unknown=[r for r in rows if r['true_id']=='unknown']
    return {'images':count,'accuracy':sum(r['true_id']==r['predicted_id'] for r in rows)/count if count else None,
            'macro_precision':sum(x['precision'] for x in per_class.values())/len(classes) if classes else None,
            'macro_recall':sum(x['recall'] for x in per_class.values())/len(classes) if classes else None,
            'macro_f1':sum(x['f1'] for x in per_class.values())/len(classes) if classes else None,
            'single_face_detection_rate':sum(r['face_count']==1 for r in rows)/count if count else None,
            'unknown_false_accept_rate':sum(r['predicted_id'] not in ('unknown','no_single_face') for r in unknown)/len(unknown) if unknown else None,
            'known_identification_rate':sum(r['true_id']==r['predicted_id'] for r in known)/len(known) if known else None,
            'per_class':per_class}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='CSV: path,true_id,condition; paths relative to the CSV')
    parser.add_argument('--split', choices=['validation','test'], required=True)
    args=parser.parse_args()
    manifest=Path(args.manifest).resolve()
    config=json.loads((ROOT/'config.json').read_text())
    engine=FaceEngine(config,ROOT)
    engine.load()
    store=Store(ROOT/'data/attendance.sqlite3')
    gallery=store.gallery(engine.fingerprint)
    enrolled={sid for sid,_ in gallery}
    if not enrolled:
        parser.error('Enroll the evaluation gallery first.')
    output=[]
    with manifest.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'path','true_id','condition'} <= set(reader.fieldnames or []):
            parser.error('CSV columns must include path,true_id,condition.')
        for row in reader:
            truth=row['true_id'].strip()
            if truth!='unknown' and truth not in enrolled:
                parser.error(f'{truth} is not enrolled. Label unenrolled probe identities as unknown.')
            image=ImageOps.exif_transpose(Image.open(manifest.parent/row['path'])).convert('RGB')
            image.thumbnail((1280,1280))
            faces=[{'embedding':engine.embed_crop(image)}] # Notebook probes are already cropped faces.
            predicted='no_single_face'
            distance=None
            if len(faces)==1:
                result=engine.match(faces[0]['embedding'],gallery)
                predicted=result['student_id'] or 'unknown'
                distance=result['distance']
            output.append(dict(path=row['path'],true_id=truth,condition=row['condition'],
                               predicted_id=predicted,face_count=len(faces),distance=distance))
    if not output:
        parser.error('Manifest is empty.')
    report={'created_at':now(),'split':args.split,'model_fingerprint':engine.fingerprint,
            'config':config,'overall':metrics(output),
            'by_condition':{c:metrics([r for r in output if r['condition']==c]) for c in sorted({r['condition'] for r in output})},
            'confusion_counts':[{ 'true_id':a,'predicted_id':b,'count':n} for (a,b),n in Counter((r['true_id'],r['predicted_id']) for r in output).items()],
            'limitations':'Already-cropped probe evaluation following notebook preprocessing; detector and live attendance need separate validation.'}
    folder=ROOT/'data'/'evaluation'
    folder.mkdir(exist_ok=True)
    (folder/'latest.json').write_text(json.dumps(report,indent=2))
    (folder/f'{args.split}.json').write_text(json.dumps(report,indent=2))
    with (folder/f'{args.split}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    print(json.dumps(report['overall'],indent=2))
    print('Results saved to data/evaluation. Refresh System & setup in the website.')


if __name__=='__main__':
    main()
