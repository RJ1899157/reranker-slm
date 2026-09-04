# ── Base Image ──────────────────────────────────────────────
FROM python:3.10-slim

# ── System Dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ───────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ───────────────────────────────────
COPY . .

# ── Default Execution ───────────────────────────────────────
CMD ["python", "data/download.py"]
