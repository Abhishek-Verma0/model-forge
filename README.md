# Model Forge

## Setup

Clone, then set up backend and frontend separately.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Runs at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at http://localhost:3000

The frontend defaults to `http://localhost:8000` for the API. To point elsewhere, create `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the backend first — CORS is set to allow `http://localhost:3000`.
