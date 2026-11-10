import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS students(id TEXT PRIMARY KEY, name TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS templates(id INTEGER PRIMARY KEY, student_id TEXT REFERENCES students(id) ON DELETE CASCADE, vector TEXT NOT NULL, fingerprint TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, name TEXT NOT NULL, started TEXT NOT NULL, ended TEXT, config TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_open_session ON sessions((1)) WHERE ended IS NULL;
            CREATE TABLE IF NOT EXISTS roster(session_id TEXT REFERENCES sessions(id), student_id TEXT, name TEXT, PRIMARY KEY(session_id,student_id));
            CREATE TABLE IF NOT EXISTS attendance(session_id TEXT, student_id TEXT, first_seen TEXT, distance REAL, PRIMARY KEY(session_id,student_id));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def students(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT s.*,COUNT(t.id) AS photos FROM students s LEFT JOIN templates t ON t.student_id=s.id GROUP BY s.id ORDER BY s.id')]

    def enroll(self, sid, name, vectors, fingerprint):
        with self.connect() as db:
            if db.execute('SELECT 1 FROM sessions WHERE ended IS NULL').fetchone():
                raise ValueError('End the active session before changing enrollment.')
            db.execute('INSERT INTO students VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name', (sid, name, now()))
            db.executemany('INSERT INTO templates(student_id,vector,fingerprint) VALUES(?,?,?)',
                           [(sid, json.dumps(v.tolist()), fingerprint) for v in vectors])

    def replace_templates(self, sid, vectors, fingerprint):
        with self.connect() as db:
            if db.execute('SELECT 1 FROM sessions WHERE ended IS NULL').fetchone():
                raise ValueError('End the active session before changing enrollment.')
            if not db.execute('SELECT 1 FROM students WHERE id=?', (sid,)).fetchone():
                raise ValueError('Student not found.')
            db.execute('DELETE FROM templates WHERE student_id=?', (sid,))
            db.executemany('INSERT INTO templates(student_id,vector,fingerprint) VALUES(?,?,?)',
                           [(sid, json.dumps(v.tolist()), fingerprint) for v in vectors])

    def remove(self, sid):
        with self.connect() as db:
            if db.execute('SELECT 1 FROM sessions WHERE ended IS NULL').fetchone():
                raise ValueError('End the active session before changing enrollment.')
            db.execute('DELETE FROM students WHERE id=?', (sid,))

    def gallery(self, fingerprint):
        with self.connect() as db:
            rows = db.execute('SELECT student_id,vector,fingerprint FROM templates').fetchall()
        if any(r['fingerprint'] != fingerprint for r in rows):
            raise ValueError('Model changed. Remove/re-enroll students to rebuild all templates before recognition.')
        return [(r['student_id'], json.loads(r['vector'])) for r in rows]

    def sessions(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM sessions ORDER BY started DESC')]

    def start(self, sid, name, config):
        with self.connect() as db:
            if db.execute('SELECT 1 FROM sessions WHERE ended IS NULL').fetchone():
                raise ValueError('A session is already active.')
            if not db.execute('SELECT 1 FROM students').fetchone():
                raise ValueError('Enroll at least one student first.')
            db.execute('INSERT INTO sessions VALUES(?,?,?,NULL,?)', (sid,name,now(),json.dumps(config)))
            db.execute('INSERT INTO roster SELECT ?,id,name FROM students', (sid,))

    def end(self, sid):
        with self.connect() as db:
            db.execute('UPDATE sessions SET ended=? WHERE id=? AND ended IS NULL', (now(),sid))

    def mark(self, session_id, student_id, distance):
        with self.connect() as db:
            active = db.execute('SELECT 1 FROM sessions WHERE id=? AND ended IS NULL', (session_id,)).fetchone()
            member = db.execute('SELECT 1 FROM roster WHERE session_id=? AND student_id=?', (session_id,student_id)).fetchone()
            if active and member:
                db.execute('INSERT OR IGNORE INTO attendance VALUES(?,?,?,?)', (session_id, student_id, now(), distance))

    def report(self, sid):
        with self.connect() as db:
            session = db.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
            if not session:
                raise ValueError('Session not found.')
            rows = db.execute('''SELECT r.student_id,r.name,a.first_seen,a.distance FROM roster r LEFT JOIN attendance a
                                 ON r.session_id=a.session_id AND r.student_id=a.student_id WHERE r.session_id=? ORDER BY r.student_id''', (sid,)).fetchall()
        return [dict(r, status='present' if r['first_seen'] else ('not observed' if session['ended'] else 'pending')) for r in rows]
