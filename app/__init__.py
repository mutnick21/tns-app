"""
Usine d'application Flask – TNS Mini-Projet
Auteur : DIC2 Génie Informatique – ESP/UCAD
"""

import os
from flask import Flask


def create_app():
    """
    Crée et configure l'instance Flask.

    Returns:
        Flask: L'application configurée avec tous les blueprints enregistrés.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Configuration générale ──────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "tns-esp-ucad-2025")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

    # Répertoires de travail
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config["DATABASE_DIR"] = os.path.join(BASE_DIR, "database")
    app.config["SEGMENTS_DIR"] = os.path.join(BASE_DIR, "segments")
    app.config["UPLOADS_DIR"] = os.path.join(BASE_DIR, "uploads")

    # Création des répertoires s'ils n'existent pas
    for d in [app.config["DATABASE_DIR"], app.config["SEGMENTS_DIR"], app.config["UPLOADS_DIR"]]:
        os.makedirs(d, exist_ok=True)

    # ── Enregistrement des Blueprints ───────────────────────────────────────
    from app.blueprints.numerisation import bp as num_bp
    from app.blueprints.filtrage import bp as flt_bp
    from app.blueprints.api import bp as api_bp

    app.register_blueprint(num_bp, url_prefix="/numerisation")
    app.register_blueprint(flt_bp, url_prefix="/filtrage")
    app.register_blueprint(api_bp, url_prefix="/api")

    # Route racine → Partie 1
    @app.route("/")
    def index():
        from flask import redirect, url_for
        return redirect(url_for("numerisation.index"))

    return app
