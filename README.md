

🏗️ Architecture

```
tns_app/
├── run.py                        # Point d'entrée Flask
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── __init__.py               # Factory d'application
│   ├── audio_processing.py       # Numérisation, validation, segmentation
│   ├── fft_processing.py         # FFT, masque rectangulaire, graphiques
│   ├── blueprints/
│   │   ├── numerisation.py       # Partie 1 – routes
│   │   ├── filtrage.py           # Partie 2 – routes
│   │   └── api.py                # API utilitaire
│   ├── templates/
│   │   ├── base.html             # Layout commun
│   │   ├── numerisation.html     # Interface Partie 1
│   │   └── filtrage.html         # Interface Partie 2
│   └── static/                   # CSS/JS additionnels
├── database/                     # Base audio organisée
├── segments/                     # Segments extraits
└── uploads/                      # Fichiers uploadés
```

---

## 🚀 Lancement local

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate         # Linux/Mac
venv\Scripts\activate            # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
python run.py
```

Accès : **http://localhost:5000**

---

## 🐳 Déploiement Docker

```bash
# Build et lancement
docker-compose up --build -d

# Vérifier l'état
docker-compose ps
curl http://localhost:5000/api/health

# Logs en direct
docker-compose logs -f

# Arrêt
docker-compose down
```

---

## 📡 Déploiement sur serveur (Ubuntu/Debian)

```bash
# 1. Cloner le projet
git clone <repo-url> tns_app && cd tns_app

# 2. Installer Docker
curl -fsSL https://get.docker.com | sh

# 3. Lancer
docker-compose up -d

# 4. (Optionnel) Reverse proxy Nginx sur port 80
# Voir section Nginx ci-dessous
```

### Configuration Nginx (optionnel)
```nginx
server {
    listen 80;
    server_name votre-domaine.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 100M;
    }
}
```

---

## 🧪 Tests rapides

```bash
# Santé de l'API
curl http://localhost:5000/api/health

# Liste des enregistrements
curl http://localhost:5000/api/recordings
```

---

## 📐 Contraintes respectées

| Contrainte | Status |
|-----------|--------|
| Python 3.x uniquement | ✅ |
| Flask uniquement | ✅ |
| Format audio WAV (conversion auto) | ✅ |
| Filtre rectangulaire uniquement | ✅ |
| Deux interfaces distinctes | ✅ |
| Docstrings sur toutes les fonctions | ✅ |
| Base de données organisée locuteur/session | ✅ |
| Fréquences autorisées : 16/22.05/44.1 kHz | ✅ |
| Codage : 16 ou 32 bits | ✅ |

---

## 📚 Références

- Flask : https://flask.palletsprojects.com
- SciPy FFT : https://docs.scipy.org/doc/scipy/reference/fft.html
- LibROSA : https://librosa.org
- Oppenheim & Schafer, *Discrete-Time Signal Processing*, Pearson 2010
