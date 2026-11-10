"""Load team files, import training-only gallery, check Colab inference parity."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from recognition import FaceEngine
from storage import Store

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check-model','import-dataset','rebuild-gallery','check-parity'])
    parser.add_argument('--folder',default=None)
    args=parser.parse_args()
    config=json.loads((ROOT/'config.json').read_text())
    engine=FaceEngine(config,ROOT)
    engine.load()
    if args.command=='check-model':
        print('Team MobileFaceNet loaded:',engine.fingerprint)
        return
    if args.command=='check-parity':
        reference=json.loads((ROOT/'data/reference_probe.json').read_text())
        actual=engine.embed_crop(Image.open(ROOT/reference['image']).convert('RGB'))
        expected=np.array(reference['embedding'],dtype=np.float32)
        if not np.allclose(actual,expected,rtol=1e-4,atol=1e-5):
            raise ValueError('Colab embedding differs. Check checkpoint, source, transforms and dependency versions.')
        templates=list(zip(reference['gallery_labels'],reference['gallery_vectors']))
        prediction=engine.match(actual,templates)
        expected_id=reference['expected_id'] if reference['expected_id'] in reference['gallery_labels'] else None
        if prediction['student_id']!=expected_id or not np.isclose(prediction['similarity'],reference['expected_similarity'],atol=1e-5):
            raise ValueError('Gallery search differs from Colab reference.')
        print('PASS: embedding and gallery search match the exported Colab reference.')
        return
    folder=Path(args.folder or config['dataset_folder'])
    if not folder.is_absolute():folder=ROOT/folder
    if not folder.is_dir():parser.error('Training split missing. Extract the Colab integration bundle first.')
    store=Store(ROOT/'data/attendance.sqlite3')
    if args.command=='rebuild-gallery':
        if any(s['ended'] is None for s in store.sessions()):
            parser.error('End the active session before rebuilding the gallery.')
        with store.connect() as db:
            db.execute('DELETE FROM templates')
        for student in store.students():
            sid=student['id']
            dataset_dir=ROOT/'data'/'face_splits'/'train'/sid
            enrolled_dir=ROOT/'data'/'enrolled'/sid
            vectors=[]
            if dataset_dir.is_dir():
                for path in sorted(dataset_dir.glob('*.jpg')):
                    with Image.open(path) as image:vectors.append(engine.embed_crop(image.convert('RGB')))
            elif enrolled_dir.is_dir():
                for path in sorted(enrolled_dir.glob('*.jpg')):
                    with Image.open(path) as image:
                        faces=engine.faces(image.convert('RGB'))
                    if len(faces)!=1:
                        raise ValueError(f'{path}: expected exactly one face; detected {len(faces)}')
                    vectors.append(faces[0]['embedding'])
            if vectors:
                store.enroll(sid,student['name'],vectors,engine.fingerprint)
                print('REBUILT',sid,':',len(vectors),'L2-normalized templates')
            else:
                print('SKIP',sid,': no source images found')
        print('Done. Gallery rebuilt with L2-normalized embeddings.')
        return
    for directory in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.')):
        sid=directory.name # Preserve notebook class labels; use friendly display names separately.
        name=sid.replace('Tr-Subject-','student')
        if any(s['id']==sid for s in store.students()):
            print('SKIP existing ID:',sid)
            continue
        vectors=[]
        for path in sorted(directory.glob('*.jpg')):
            # These are already-cropped faces; do not run a second detector.
            with Image.open(path) as image:vectors.append(engine.embed_crop(image.convert('RGB')))
        if vectors:
            store.enroll(sid,name,vectors,engine.fingerprint)
            print('IMPORTED',sid,'as',name,':',len(vectors),'training-only templates')
    print('Done. Do not add validation/test images to this gallery.')


if __name__=='__main__':main()
