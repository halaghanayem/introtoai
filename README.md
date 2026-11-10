# Presence

A local classroom attendance website using the project's MobileFaceNet embedding pipeline and cosine-similarity recognition.

## Current behavior

- Neutral, polished dashboard with live camera recognition.
- Cosine-similarity threshold set to **0.50** in `config.json`.
- Students can be enrolled with new face photos and existing student photos can be replaced.
- Attendance sessions, student rosters, history, and CSV export are stored locally in SQLite.
- Recognition feed shows names and status without exposing raw similarity values.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://localhost:8000>.

On Windows, create the environment with `py -3.11 -m venv .venv`, activate it with `.venv\\Scripts\\activate`, then run `python app.py`.

## Privacy and data

Local face photos, embeddings, and attendance records are intentionally ignored by Git. Start the app and enroll students from the **Students** view to create a fresh local gallery.

The included `vendor/mobile_weights/model_mobilefacenet.pth` checkpoint is the original MobileFaceNet model used by the application. The website does not use an SVM classifier; identity decisions use the original cosine-similarity gallery search.

## Main files

- `app.py` — web server and API routes
- `recognition.py` — model loading, face detection, embeddings, and gallery search
- `storage.py` — SQLite persistence
- `templates/index.html` — page structure
- `static/app.js` — camera, enrollment, and API interactions
- `static/style.css` — neutral visual theme
- `config.json` — recognition threshold and model settings

