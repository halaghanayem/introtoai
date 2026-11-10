"""Local classroom attendance website. Run: python app.py"""
import argparse
import base64
import csv
import io
import json
import os
import re
import secrets
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps, UnidentifiedImageError
from recognition import FaceEngine
from storage import Store

ROOT = Path(__file__).resolve().parent
MAX_BODY = 24 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def decode_image(value):
    if not isinstance(value, str):
        raise ValueError('An image is required.')
    try:
        raw = base64.b64decode(value.split(',', 1)[-1], validate=True)
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > 20_000_000:
            raise ValueError('Image is too large.')
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((1280, 1280))
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Upload a valid JPEG or PNG image.') from exc


class Application:
    def __init__(self, root=ROOT, engine=None):
        self.root = Path(root)
        self.config = json.loads((self.root / 'config.json').read_text())
        self.validate_config()
        self.store = Store(self.root / 'data' / 'attendance.sqlite3')
        self.engine = engine or FaceEngine(self.config, self.root)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.confirmations = {}

    def validate_config(self):
        for key, low, high in [('cosine_similarity_threshold', -1, 1)]:
            value = self.config[key]
            if not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f'Invalid config: {key}')
        if not isinstance(self.config['confirmations'], int) or self.config['confirmations'] < 1:
            raise ValueError('confirmations must be a positive integer.')
        if self.config['confirmation_gap_seconds'] <= 0:
            raise ValueError('confirmation_gap_seconds must be positive.')

    def state(self):
        return {'students': self.store.students(), 'sessions': self.store.sessions(),
                'model_ready': self.engine.model is not None, 'model_error': self.engine.error,
                'config': self.config, 'token': self.token,
                'missing_team_files': self.engine.required_paths()[2] if hasattr(self.engine, 'required_paths') else []}

    def enroll(self, body):
        sid, name = str(body.get('student_id', '')).strip(), str(body.get('name', '')).strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', sid):
            raise ValueError('Student ID: use 1–50 letters, numbers, underscores or hyphens.')
        if sid.lower() == 'unknown' or not name or len(name) > 100:
            raise ValueError('Provide a name of 1–100 characters and a nonreserved ID.')
        images = body.get('images', [])
        if not isinstance(images, list) or not 1 <= len(images) <= 20:
            raise ValueError('Provide between 1 and 20 enrollment images.')
        vectors = []
        enrollment_images = []
        for index, raw in enumerate(images, 1):
            image = decode_image(raw)
            faces = self.engine.faces(image)
            if len(faces) != 1:
                raise ValueError(f'Photo {index}: expected exactly one face; detected {len(faces)}. No changes saved.')
            vectors.append(faces[0]['embedding'])
            enrollment_images.append(image)
        self.store.enroll(sid, name, vectors, self.engine.fingerprint)
        # Keep an auditable copy of accepted source images as part of the local
        # enrollment dataset. Recognition still uses the stored embeddings, so
        # adding a student never requires retraining the MobileFaceNet model.
        dataset_dir = self.root / 'data' / 'enrolled' / sid
        dataset_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        for index, image in enumerate(enrollment_images, 1):
            image.save(dataset_dir / f'{stamp}_{index:02d}.jpg', format='JPEG', quality=92)
        return {'message': f'Saved {name}: {len(vectors)} face template(s).'}

    def recognize(self, body):
        source = body.get('source', 'live')
        if source not in ('live', 'image', 'video'):
            raise ValueError('Invalid recognition source.')
        faces = self.engine.faces(decode_image(body.get('image')))
        gallery = self.store.gallery(self.engine.fingerprint)
        names = {r['id']: r['name'] for r in self.store.students()}
        active = next((s for s in self.store.sessions() if s['ended'] is None), None)
        results, hits = [], {}
        for face in faces:
            result = self.engine.match(face['embedding'], gallery)
            sid = result['student_id']
            result.update(bbox=face['bbox'], name=names.get(sid, 'Unknown'), confirmed=False)
            if sid:
                hits[sid] = min(hits.get(sid, 2), result['distance'])
            results.append(result)
        # Only live frames may mark attendance. Still-image/video analysis is a preview.
        # One vote per student per request, regardless of duplicate detections.
        confirmed = set()
        if active and source == 'live':
            timestamp = time.monotonic()
            session_id = active['id']
            current_keys = {(session_id, sid) for sid in hits}
            for key in list(self.confirmations):
                if key not in current_keys:
                    del self.confirmations[key]
            for sid, distance in hits.items():
                key = (session_id, sid)
                count, previous = self.confirmations.get(key, (0, 0))
                count = count + 1 if timestamp - previous <= self.config['confirmation_gap_seconds'] else 1
                self.confirmations[key] = (count, timestamp)
                if count >= self.config['confirmations']:
                    self.store.mark(session_id, sid, distance)
                    confirmed.add(sid)
        for result in results:
            result['confirmed'] = result['student_id'] in confirmed
        return {'faces': results, 'session_id': active['id'] if active else None,
                'attendance_enabled': source == 'live' and active is not None}

    def replace_student_photos(self, body):
        sid = str(body.get('student_id', '')).strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', sid):
            raise ValueError('Invalid student ID.')
        images = body.get('images', [])
        if not isinstance(images, list) or not 1 <= len(images) <= 20:
            raise ValueError('Provide between 1 and 20 replacement photos.')
        vectors, enrollment_images = [], []
        for index, raw in enumerate(images, 1):
            image = decode_image(raw)
            faces = self.engine.faces(image)
            if len(faces) != 1:
                raise ValueError(f'Photo {index}: expected exactly one face; detected {len(faces)}. No changes saved.')
            vectors.append(faces[0]['embedding'])
            enrollment_images.append(image)
        self.store.replace_templates(sid, vectors, self.engine.fingerprint)
        dataset_dir = self.root / 'data' / 'enrolled' / sid
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for path in dataset_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                path.unlink()
        stamp = int(time.time() * 1000)
        for index, image in enumerate(enrollment_images, 1):
            image.save(dataset_dir / f'{stamp}_{index:02d}.jpg', format='JPEG', quality=92)
        return {'message': f'Updated {sid}: {len(vectors)} face template(s).'}

    def post(self, route, body):
        with self.lock:
            if route == '/api/model/load':
                self.engine.load()
                return {'message': 'Team MobileFaceNet loaded.'}
            if route == '/api/students':
                return self.enroll(body)
            if route == '/api/students/photos':
                return self.replace_student_photos(body)
            if route == '/api/students/remove':
                self.store.remove(str(body.get('student_id', '')))
                return {'message': 'Student removed. Historical session records retained.'}
            if route == '/api/recognize':
                return self.recognize(body)
            if route == '/api/sessions/start':
                name = str(body.get('name', '')).strip()
                if not name or len(name) > 100:
                    raise ValueError('Enter a session name of 1–100 characters.')
                sid = str(uuid.uuid4())
                self.store.start(sid, name, dict(self.config, model_fingerprint=self.engine.fingerprint))
                self.confirmations.clear()
                return {'message': 'Session started.', 'id': sid}
            if route == '/api/sessions/end':
                self.store.end(str(body.get('session_id', '')))
                self.confirmations.clear()
                return {'message': 'Session ended.'}
        raise ValueError('Unknown operation.')


def csv_text(rows):
    out = io.StringIO(newline='')
    fields = ['student_id', 'name', 'status', 'first_seen', 'distance']
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    # Prevent spreadsheet formula interpretation of user-entered names.
    for row in rows:
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v)
                         for k, v in row.items()})
    return out.getvalue()


def handler_for(application):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if '/api/recognize' not in str(args[0]):
                super().log_message(fmt, *args)

        def reply(self, status, payload, content_type='application/json; charset=utf-8', attachment=False):
            raw = json.dumps(payload).encode() if content_type.startswith('application/json') else payload
            if isinstance(raw, str):
                raw = raw.encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'same-origin')
            self.send_header('Permissions-Policy', 'camera=(self), microphone=()')
            if attachment:
                self.send_header('Content-Disposition', 'attachment; filename="attendance.csv"')
            self.end_headers()
            self.wfile.write(raw)

        def valid_host(self):
            hostname = self.headers.get('Host', '').split(':')[0]
            allowed = {'localhost', '127.0.0.1'}
            render_host = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
            if render_host:
                allowed.add(render_host.split(':')[0])
            return hostname in allowed

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {'error': 'Use http://localhost:8000 on this computer.'})
                return
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/api/state':
                    self.reply(200, application.state())
                elif parsed.path == '/api/evaluation':
                    path = application.root / 'data/evaluation/latest.json'
                    self.reply(200, json.loads(path.read_text()) if path.exists() else {'message': 'No evaluation run yet. See README.'})
                elif parsed.path == '/api/report':
                    sid = parse_qs(parsed.query).get('session_id', [''])[0]
                    rows = application.store.report(sid)
                    if 'csv' in parse_qs(parsed.query):
                        self.reply(200, csv_text(rows), 'text/csv; charset=utf-8', True)
                    else:
                        self.reply(200, {'rows': rows})
                else:
                    paths = {'/': ('templates/index.html', 'text/html; charset=utf-8'),
                             '/static/app.js': ('static/app.js', 'text/javascript; charset=utf-8'),
                             '/static/style.css': ('static/style.css', 'text/css; charset=utf-8')}
                    if parsed.path not in paths:
                        self.reply(404, {'error': 'Not found.'})
                        return
                    path, mime = paths[parsed.path]
                    self.reply(200, (application.root / path).read_bytes(), mime)
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})
            except Exception as exc:
                self.reply(500, {'error': str(exc)})

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-App-Token') != application.token:
                self.reply(403, {'error': 'Refresh this page before continuing.'})
                return
            origin = self.headers.get('Origin')
            forwarded_proto = self.headers.get('X-Forwarded-Proto', 'http').split(',')[0].strip()
            expected_origin = forwarded_proto + '://' + self.headers.get('Host', '')
            if origin and origin != expected_origin:
                self.reply(403, {'error': 'Cross-origin requests are not allowed.'})
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= MAX_BODY:
                    self.reply(413, {'error': 'Request exceeds 24 MB or has no content.'})
                    return
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object.')
                self.reply(200, application.post(urlparse(self.path).path, body))
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception as exc:
                self.reply(503, {'error': str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    application = Application()
    port = args.port or int(os.environ.get('PORT', application.config['port']))
    host = os.environ.get('HOST', '0.0.0.0')
    server = ThreadingHTTPServer((host, port), handler_for(application))
    print(f'Listening on {host}:{port} — press Ctrl+C to stop.', flush=True)
    print('Camera images are processed locally. Team MobileFaceNet loads when requested.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
