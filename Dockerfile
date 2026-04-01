# ── Dockerfile – TNS Mini-Projet ESP/UCAD ──────────────────────────────────
# Image de base légère Python 3.11 (slim)
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="DIC2 Génie Informatique – ESP/UCAD"
LABEL description="Traitement Numérique du Signal – App Flask"

# ── Dépendances système (ffmpeg pour pydub, libsndfile pour librosa) ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Répertoire de travail ────────────────────────────────────────────────────
WORKDIR /app

# ── Installation des dépendances Python ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code source ─────────────────────────────────────────────────────
COPY . .

# ── Création des répertoires de données ─────────────────────────────────────
RUN mkdir -p database segments uploads

# ── Exposition du port ───────────────────────────────────────────────────────
EXPOSE 5000

# ── Variables d'environnement de production ──────────────────────────────────
ENV FLASK_ENV=production
ENV SECRET_KEY=change-me-in-production

# ── Lancement via Gunicorn (serveur WSGI production) ─────────────────────────
# 4 workers, timeout 120s pour les longues opérations FFT
CMD ["gunicorn", \
     "--workers", "4", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "run:app"]
