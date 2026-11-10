"""Adapter to the team's FaceRecognition.ipynb. No substitute backbone."""
import hashlib
import sys
import threading
from pathlib import Path
import numpy as np


def unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    if v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-10:
        raise ValueError('Invalid face embedding')
    return v / np.linalg.norm(v)



class FaceEngine:
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.lock=threading.RLock()
        self.model=None
        self.error=None
        self._fingerprint=None

    def required_paths(self):
        source=self.root/self.config['team_source']
        checkpoint=self.root/self.config['checkpoint']
        needed=[source/'data_loader.py',source/'generate_embeddings.py',source/'networks/models_facenet.py',checkpoint]
        return source,checkpoint,[str(p.relative_to(self.root)) for p in needed if not p.is_file()]

    @property
    def fingerprint(self):
        if self._fingerprint is None:
            source,checkpoint,missing=self.required_paths()
            if missing:raise ValueError('Team files missing: '+', '.join(missing)+'. Run EXPORT_FROM_COLAB.py in your notebook and extract its ZIP here.')
            digest=hashlib.sha256()
            for path in sorted(source.rglob('*.py')):
                digest.update(str(path.relative_to(source)).encode());digest.update(path.read_bytes())
            digest.update(checkpoint.read_bytes())
            digest.update(b'MobileFaceNet512:build_transforms(false,112x112):rgb:l2:v3')
            self._fingerprint='team-mobilefacenet:'+digest.hexdigest()
        return self._fingerprint

    def load(self):
        with self.lock:
            if self.model is not None:return
            try:
                source,checkpoint,missing=self.required_paths()
                if missing:raise FileNotFoundError('Missing team files: '+', '.join(missing))
                import torch
                import cv2
                sys.path.insert(0,str(source.resolve()))
                from networks.models_facenet import MobileFaceNet
                from data_loader import build_transforms
                from generate_embeddings import search_gallery
                self.torch=torch
                torch.set_num_threads(self.config.get('cpu_threads',2))
                self.device=torch.device(self.config.get('device','cpu'))
                model=MobileFaceNet(embedding_size=512)
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if isinstance(state,dict) and 'model_state_dict' in state:state=state['model_state_dict']
                model.load_state_dict(state,strict=True)
                self.transform=build_transforms(train=False,crop_size=(112,112))
                self.search_gallery=search_gallery
                # Additional full-frame detector: notebook inputs are already cropped faces.
                self.cv2=cv2
                self.detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
                if self.detector.empty():raise RuntimeError('OpenCV face detector could not load.')
                self.model=model.to(self.device).eval()
                self._fingerprint=None
                self.error=None
            except Exception as exc:
                self.error=f'{type(exc).__name__}: {exc}'
                raise RuntimeError('Team MobileFaceNet is not ready. '+self.error) from exc

    def embed_crop(self,image):
        """Exact notebook inference for an already cropped RGB face."""
        self.load()
        with self.lock,self.torch.no_grad():
            tensor=self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
            vector=self.model(tensor).detach().cpu().numpy()[0]
            if vector.shape!=(512,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-10:
                raise ValueError('Expected a finite, nonzero 512-D MobileFaceNet embedding.')
            return vector

    def faces(self,image):
        self.load()
        with self.lock:
            gray=self.cv2.cvtColor(np.asarray(image),self.cv2.COLOR_RGB2GRAY)
            # Normalize contrast and use a slightly more tolerant cascade so
            # normal webcam lighting and small head movement do not make the
            # face disappear between recognition requests.
            gray=self.cv2.equalizeHist(gray)
            raw_boxes=self.detector.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=4,minSize=(32,32))
            height,width=gray.shape
            candidates=[]
            for x,y,w,h in raw_boxes:
                # Clothing graphics and furniture often trigger Haar at the
                # bottom of a portrait frame; classroom faces remain in the
                # upper portion of the camera view.
                if int(y) > int(height*0.72):
                    continue
                candidates.append((int(x),int(y),int(w),int(h)))

            # Haar can return a large head box and a smaller face box for the
            # same person. Keep the tighter box and retain separate faces.
            def overlap(a,b):
                ax,ay,aw,ah=a; bx,by,bw,bh=b
                ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by+bh)-max(ay,by))
                inter=ix*iy
                return inter/min(aw*ah,bw*bh) if inter else 0
            boxes=[]
            for candidate in sorted(candidates,key=lambda box:box[2]*box[3]):
                if any(overlap(candidate,kept)>0.45 for kept in boxes):
                    continue
                boxes.append(candidate)
            return [{'bbox':[int(x),int(y),int(x+w),int(y+h)],
                     'embedding':self.embed_crop(image.crop((int(x),int(y),int(x+w),int(y+h))))}
                    for x,y,w,h in boxes]

    def match(self,vector,templates):
        self.load()
        if not templates:
            return {'student_id':None,'distance':None,'similarity':None,'reason':'No enrolled students'}
        labels=np.array([sid for sid,_ in templates])
        gallery=np.stack([v for _,v in templates]).astype(np.float32)
        query=np.asarray(vector,dtype=np.float32).reshape(1,-1)
        threshold=float(self.config.get('cosine_similarity_threshold',0.5))
        match_label,confidence=self.search_gallery(query,gallery,labels,topk=1,threshold=threshold)[0]
        confidence=float(confidence)
        if not np.isfinite(confidence):raise ValueError('Cosine similarity returned an invalid value.')
        sid=None if match_label == 'Unknown' else str(match_label)
        return {'student_id':sid,'similarity':confidence,'distance':1-confidence,
                'reason':'Recognized by cosine similarity' if sid else 'Below cosine similarity threshold'}
