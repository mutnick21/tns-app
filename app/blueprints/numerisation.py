"""
blueprints/numerisation.py – Partie 1 : Numérisation et segmentation
ESP/UCAD – TNS Mini-Projet 2025-2026
"""

import os
from flask import (Blueprint, render_template, request, jsonify,
                   current_app, send_from_directory)
from app.audio_processing import (
    validate_params, save_wav_from_float32, build_db_path,
    generate_wav_filename, list_database, segment_audio,
    ALLOWED_SAMPLE_RATES, ALLOWED_BIT_DEPTHS,
)
import numpy as np

bp = Blueprint("numerisation", __name__)


@bp.route("/")
def index():
    """
    Affiche l'interface Partie 1 – Numérisation / Segmentation.

    Returns:
        str: Page HTML rendue.
    """
    recordings = list_database(current_app.config["DATABASE_DIR"])
    return render_template("numerisation.html",
                           recordings=recordings,
                           allowed_rates=ALLOWED_SAMPLE_RATES,
                           allowed_bits=ALLOWED_BIT_DEPTHS)


@bp.route("/save", methods=["POST"])
def save_recording():
    """
    Reçoit un enregistrement audio (blob WebM/WAV) depuis le navigateur,
    valide les paramètres, et sauvegarde le fichier WAV dans la base.

    Form data attendu :
        audio      : fichier audio (blob)
        sampleRate : fréquence d'échantillonnage (int)
        bitDepth   : profondeur de codage (int)
        locuteur   : identifiant locuteur (str)
        session    : identifiant session (str)

    Returns:
        JSON: {success, message, filepath}
    """
    try:
        # Récupération des paramètres
        sample_rate = int(request.form.get("sampleRate", 16000))
        bit_depth   = int(request.form.get("bitDepth", 16))
        locuteur    = request.form.get("locuteur", "01").strip()
        session     = request.form.get("session", "01").strip()

        validate_params(sample_rate, bit_depth)

        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"success": False,
                            "message": "Aucun fichier audio reçu."}), 400

        # Détermination de l'index (prochain numéro disponible)
        db_dir  = current_app.config["DATABASE_DIR"]
        session_dir = os.path.join(
            db_dir,
            f"locuteur_{locuteur.zfill(2)}",
            f"session_{session.zfill(2)}"
        )
        os.makedirs(session_dir, exist_ok=True)
        existing = [f for f in os.listdir(session_dir) if f.endswith(".wav")]
        index = len(existing) + 1

        filename = generate_wav_filename(sample_rate, bit_depth, index)
        filepath = os.path.join(session_dir, filename)

        # Lecture des données brutes depuis le blob
        audio_bytes = audio_file.read()

        # Le navigateur envoie maintenant un WAV PCM 16 bits valide (RIFF)
        # encodé directement côté client via AudioContext + encodeur maison.
        # On vérifie le magic bytes, puis on rééchantillonne si besoin.
        import io as _io
        from scipy.io import wavfile as _wf
        import numpy as _np

        if audio_bytes[:4] != b'RIFF':
            return jsonify({
                "success": False,
                "message": (
                    "Format inattendu reçu (non-WAV). "
                    "Rechargez la page et réessayez."
                )
            }), 400

        sr_orig, data = _wf.read(_io.BytesIO(audio_bytes))

        # Mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Normalisation float32
        if _np.issubdtype(data.dtype, _np.integer):
            data = data.astype(_np.float32) / _np.iinfo(data.dtype).max
        else:
            data = data.astype(_np.float32)

        # Rééchantillonnage si la fréquence source diffère de la cible
        if sr_orig != sample_rate:
            from scipy.signal import resample as _res
            n = int(len(data) * sample_rate / sr_orig)
            data = _res(data, n).astype(_np.float32)

        # Conversion vers la profondeur cible
        if bit_depth == 16:
            pcm = (_np.clip(data, -1, 1) * 32767).astype(_np.int16)
        else:
            pcm = (_np.clip(data, -1, 1) * 2147483647).astype(_np.int32)

        _wf.write(filepath, sample_rate, pcm)

        size_kb = round(os.path.getsize(filepath) / 1024, 1)

        return jsonify({
            "success":  True,
            "message":  f"Enregistrement sauvegardé : {filename}",
            "filename": filename,
            "filepath": filepath,
            "size_kb":  size_kb,
            "locuteur": f"locuteur_{locuteur.zfill(2)}",
            "session":  f"session_{session.zfill(2)}",
        })

    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Erreur serveur : {str(e)}"}), 500


@bp.route("/segment", methods=["POST"])
def segment():
    """
    Lance la segmentation automatique d'un fichier WAV existant.

    JSON body attendu :
        filepath     (str):   Chemin relatif ou absolu du fichier WAV.
        threshold    (float): Seuil d'amplitude (0-1).
        min_silence  (int):   Durée minimale de silence en ms.

    Returns:
        JSON: {success, segments: [{filename, duration_s, url}]}
    """
    try:
        data         = request.get_json(force=True)
        filepath     = data.get("filepath", "")
        threshold    = float(data.get("threshold", 0.02))
        min_silence  = int(data.get("min_silence", 300))

        if not os.path.isfile(filepath):
            return jsonify({"success": False,
                            "message": f"Fichier introuvable : {filepath}"}), 404

        segments_dir = current_app.config["SEGMENTS_DIR"]
        results = segment_audio(filepath, segments_dir,
                                amplitude_threshold=threshold,
                                min_silence_ms=min_silence)

        # Ajout de l'URL de téléchargement
        for seg in results:
            seg["url"] = f"/numerisation/segments/{seg['filename']}"

        return jsonify({
            "success":  True,
            "count":    len(results),
            "segments": results,
        })

    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Erreur segmentation : {str(e)}"}), 500


@bp.route("/segments/<filename>")
def serve_segment(filename: str):
    """
    Sert un fichier de segment audio depuis le dossier segments/.

    Args:
        filename (str): Nom du fichier WAV segment.

    Returns:
        Response: Fichier audio.
    """
    return send_from_directory(current_app.config["SEGMENTS_DIR"], filename)


@bp.route("/database/<path:subpath>")
def serve_database(subpath: str):
    """
    Sert un fichier WAV depuis la base de données.

    Args:
        subpath (str): Chemin relatif dans database/.

    Returns:
        Response: Fichier audio.
    """
    return send_from_directory(current_app.config["DATABASE_DIR"], subpath)
