"""
blueprints/api.py – Endpoints API utilitaires
ESP/UCAD – TNS Mini-Projet 2025-2026
"""

from flask import Blueprint, jsonify, current_app
from app.audio_processing import list_database

bp = Blueprint("api", __name__)


@bp.route("/recordings")
def get_recordings():
    """
    Retourne la liste complète des enregistrements de la base de données.

    Returns:
        JSON: {recordings: [...]}
    """
    recordings = list_database(current_app.config["DATABASE_DIR"])
    return jsonify({"recordings": recordings})


@bp.route("/health")
def health():
    """
    Endpoint de santé pour les sondes Docker / load balancer.

    Returns:
        JSON: {status: "ok"}
    """
    return jsonify({"status": "ok", "app": "TNS Mini-Projet ESP/UCAD"})
