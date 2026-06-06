# 🎱 KINO Data Lab

A full-stack data analysis platform for **OPAP KINO** (Greek lottery), providing real-time statistical analysis, number relation tracking, heat mapping, and trend visualization across historical draw frames.

---

## 📸 Overview

KINO Data Lab gives players and analysts a deep look into KINO draw history through three core analytical views:

- **Single Number Relations** — detailed co-occurrence analysis for a specific number
- **General Relations** — hot, cold, and middle anchor numbers for any given frame window
- **Trend Frames** — heat map visualization of all 80 numbers across a rolling 20-game base frame

---

## 🧱 Tech Stack

### Backend
- **Python / Django** (Django REST Framework)
- ASGI server (`asgi.py`, `wsgi.py`)
- Django app: `kino` with modules for `management`, `migrations`, `services`, `api`
- Models, serializers, views, and URL routing (`models.py`, `serializers.py`, `views.py`, `urls.py`, `admin.py`, `apps.py`)

### Frontend
- **React + TypeScript** (Vite)
- Pages: `SingleNumberRelationsPage.tsx`, `GeneralRelationsPage.tsx`, `TrendFramesPage.tsx`
- API layer: `kino.ts` (typed API client)
- Utilities: `frameSelection.ts`, `relationSelection.ts`, `window_relations.py`
- Styling: `App.css`
- Dark navy blue UI theme with amber/orange/red color coding

---

## 📁 Project Structure

```
kino-data-lab/
├── backend/
│   ├── kino/                        # Main Django app
│   │   ├── api/                     # REST API endpoints
│   │   ├── management/              # Django management commands
│   │   ├── migrations/              # DB migrations
│   │   ├── services/                # Business logic / analysis engine
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── frontend/
    ├── public/
    ├── src/
    │   ├── api/
    │   │   └── kino.ts              # API client
    │   ├── assets/
    │   ├── components/
    │   ├── pages/
    │   │   ├── GeneralRelationsPage.tsx
    │   │   ├── SingleNumberRelationsPage.tsx
    │   │   └── TrendFramesPage.tsx
    │   ├── styles/
    │   ├── utils/
    │   │   ├── frameSelection.ts
    │   │   └── relationSelection.ts
    │   ├── App.css
    │   ├── App.tsx
    │   ├── index.html
    │   ├── main.tsx
    │   └── kino.ts
    ├── eslint.config.js
    ├── package.json
    └── package-lock.json
```

---

## 🔍 Features

### 1. Single Number Relations (`/single-number-relations`)

Analyzes the relationship of a **selected number** (e.g. number 35) with all other numbers across a configurable window of draws.

**Controls:**
- Window ID selection
- Frame selector
- Mode: `20 games / step 10` (configurable sliding window)

**Stats displayed:**
- **Selected number** (e.g. 35)
- **Anchor appearances** — total appearances, split (e.g. `2 | 1`)
- **Top connected numbers** — sorted by connection count, each showing:
  - Number of shared appearances (connections)
  - Split ratio (how many times they appeared together in each half of the window)
  - Trend delta badge (e.g. `-2`, `0`)

### 2. General Relations (`/general-relations`)

Auto-selects **hot, cold, and middle** anchor numbers for the current frame.

**Controls:**
- Window ID & Frame selector
- Mode: `20 games / step 10`

**Anchor classifications:**
- 🔴 **Hot** — high-frequency numbers in the current window (e.g. 77, 16, 23, 32, 41)
- 🔵 **Cold** — low-frequency numbers (e.g. 74, 6, 11)
- 🟡 **Middle** — moderate frequency (implied by the heat scale)

Each anchor shows its **Heat value** (e.g. Heat 9, Heat 8) indicating relative draw frequency.

### 3. Trend Frames (`/trends`)

A **20-game base frame** heat map visualizing all 80 KINO numbers (1–80) in an 8×10 grid.

**Features:**
- Displays draw range (e.g. Draws `1303738 → 1303757`)
- Each number cell shows:
  - The **number** (large, centered)
  - A **frequency count** below it
- Color-coded by heat scale:
  - 🔴 **Red** — hottest (most frequent), value `11+`
  - 🟠 **Orange/Amber** — warm
  - 🟡 **Yellow** — moderate
  - ⚪ **Light/White** — cool
  - 🔵 **Blue** — coldest (least frequent), value `0–1`
- **Heat scale legend** at the top (0 → 11+, left = lower heat, right = hotter)

---

## ⚙️ Configuration

### Backend (Django)

Configure your database and settings in `backend/settings.py`.

```python
# Example: SQLite for development
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### Frontend (Vite + React)

The frontend communicates with the Django backend via REST API calls defined in `src/api/kino.ts`. The base URL can be configured via environment variables.

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## 🔌 API Reference

### General Relations

```
GET /api/general-relations/?window_id=<id>&games=20&step=10
```

Returns anchor numbers (hot/cold/middle) with heat values for the given window.

**Response shape:**
```json
{
  "anchors": [
    { "number": 77, "type": "hot", "heat": 9 },
    { "number": 74, "type": "cold", "heat": 1 }
  ]
}
```

### Single Number Relations

```
GET /api/single-number-relations/?window_id=<id>&frame=<frame>&number=<n>
```

Returns co-occurrence statistics for a specific number relative to all others.

---

## 📊 Analysis Logic

### Window & Frame System

- A **window** is a group of consecutive KINO draws
- A **frame** is a subset within that window (e.g. frame 13 of a 20-game window with step 10)
- The sliding window (`step 10`) moves forward by 10 draws per frame, allowing trend tracking over time

### Heat Scoring

Numbers are scored based on how frequently they appear in the selected window. The heat scale (0–11+) reflects draw frequency relative to the statistical mean for that window size.

### Relation / Connection Scoring

Two numbers are "connected" if they appear together in the same draw. The **split** value shows how that co-occurrence is distributed across the two halves of the window, indicating whether the relationship is recent or historical.

---

## 🗃️ Database

The project uses Django ORM with SQLite (default) or any Django-supported DB. Draw data is stored and queried via the `kino` app models.

The `db.sqlite3` file at the backend root stores historical KINO draw data. You can seed it using Django management commands in `kino/management/`.

---

## 🧪 Development Notes

- The frontend uses **TypeScript** with strict typing for all API responses
- Frame and relation selection logic is abstracted into utility modules (`frameSelection.ts`, `relationSelection.ts`)
- The `GeneralRelationsPage` fetches data with `fetchGeneralRelations(selection.windowId, 20, 20)` and sets the first anchor as the default selected anchor
- Error and loading states are handled per-page with `setLoading(false)` in `finally` blocks

---

## 📦 Deployment

The project is hosted on GitHub at:

```
https://github.com/Panos-04/kino-data-lab
```

To deploy:
1. Push backend to a Python-compatible host (e.g. Railway, Render, or VPS)
2. Build frontend: `npm run build` → serve `dist/` with Nginx or Vercel
3. Set `VITE_API_BASE_URL` to your production API URL

---

## 📝 License

MIT — feel free to use, modify, and distribute.

---

> Built with ❤️ for KINO analysis enthusiasts. Not affiliated with OPAP S.A.