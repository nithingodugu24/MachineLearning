# Movie Recommender App

A simple Flask-based web app that serves movie recommendations using precomputed models.

## Structure
- `app.py`: Flask app entry point.
- `models/`: Model artifacts (`.pkl`) organized by version.
- `templates/`, `static/`: Frontend assets.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install flask numpy pandas scikit-learn
   ```

## Run
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Notes
- Ensure model files exist under `models/` as referenced by `app.py`.
- For production, consider `gunicorn`/`waitress` and environment variables for configuration.
