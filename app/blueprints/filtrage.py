"""
blueprints/filtrage.py – Partie 2 : Analyse FFT et filtrage rectangulaire
ESP/UCAD – TNS Mini-Projet 2025-2026
"""

import os
import uuid
from flask import (Blueprint, render_template, request, jsonify,
                   current_app, send_from_directory)
from app.fft_processing import (
    load_audio, compute_fft, apply_rectangular_filter,
    plot_time_signal, plot_spectrum, plot_comparison, export_filtered_wav,
)

bp = Blueprint("filtrage", __name__)

# Formats audio acceptés à l'upload
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac"}


def _allowed(filename: str) -> bool:
    """Vérifie si l'extension du fichier est autorisée."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _convert_to_wav(src_path: str, dst_path: str) -> str:
    """
    Convertit un fichier audio quelconque en WAV mono via pydub.

    Args:
        src_path (str): Chemin du fichier source.
        dst_path (str): Chemin WAV de destination.

    Returns:
        str: Chemin du fichier WAV créé.
    """
    from pydub import AudioSegment
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1)       # Mono
    audio.export(dst_path, format="wav")
    return dst_path


@bp.route("/")
def index():
    """
    Affiche l'interface Partie 2 – FFT / Filtrage.

    Returns:
        str: Page HTML rendue.
    """
    return render_template("filtrage.html")


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Reçoit un fichier audio, le convertit en WAV si nécessaire,
    calcule FFT et retourne les graphiques en Base64.

    Form data :
        file : fichier audio (WAV, MP3, OGG…)

    Returns:
        JSON: {success, filename, wav_path, plot_time, plot_fft,
               duration_s, sample_rate}
    """
    try:
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify({"success": False,
                            "message": "Aucun fichier reçu."}), 400

        if not _allowed(f.filename):
            return jsonify({"success": False,
                            "message": "Format non supporté. "
                                       f"Acceptés : {ALLOWED_EXTENSIONS}"}), 400

        uploads_dir = current_app.config["UPLOADS_DIR"]
        uid         = uuid.uuid4().hex[:8]
        orig_ext    = f.filename.rsplit(".", 1)[1].lower()
        orig_path   = os.path.join(uploads_dir, f"{uid}_orig.{orig_ext}")
        wav_path    = os.path.join(uploads_dir, f"{uid}.wav")

        f.save(orig_path)

        # Conversion → WAV si nécessaire
        if orig_ext != "wav":
            _convert_to_wav(orig_path, wav_path)
        else:
            wav_path = orig_path

        signal, sr = load_audio(wav_path)
        freqs, amps = compute_fft(signal, sr)

        plot_t  = plot_time_signal(signal, sr)
        plot_f  = plot_spectrum(freqs, amps)

        return jsonify({
            "success":     True,
            "uid":         uid,
            "filename":    f.filename,
            "wav_file":    os.path.basename(wav_path),
            "duration_s":  round(len(signal) / sr, 2),
            "sample_rate": sr,
            "plot_time":   plot_t,
            "plot_fft":    plot_f,
            "max_freq":    int(sr / 2),  # Fréquence de Nyquist
        })

    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Erreur upload : {str(e)}"}), 500


@bp.route("/filter", methods=["POST"])
def filter_audio():
    """
    Applique le filtre rectangulaire sur le fichier WAV chargé.

    JSON body :
        uid         (str):   Identifiant unique du fichier chargé.
        wav_file    (str):   Nom du fichier WAV.
        f_min       (float): Fréquence minimale en Hz.
        f_max       (float): Fréquence maximale en Hz.
        filter_type (str):   "passband" ou "stopband".

    Returns:
        JSON: {success, plot_time_orig, plot_time_filt,
               plot_fft_orig, plot_fft_filt,
               download_url, filtered_filename}
    """
    try:
        data        = request.get_json(force=True)
        uid         = data.get("uid", "")
        wav_file    = data.get("wav_file", "")
        f_min       = float(data.get("f_min", 300))
        f_max       = float(data.get("f_max", 3400))
        filter_type = data.get("filter_type", "passband")

        uploads_dir = current_app.config["UPLOADS_DIR"]
        wav_path    = os.path.join(uploads_dir, wav_file)

        if not os.path.isfile(wav_path):
            return jsonify({"success": False,
                            "message": "Fichier WAV introuvable."}), 404

        signal, sr    = load_audio(wav_path)
        freqs, amps   = compute_fft(signal, sr)

        # Application du masque rectangulaire
        signal_filt   = apply_rectangular_filter(signal, sr, f_min, f_max, filter_type)

        # Export WAV filtré
        out_filename  = f"{uid}_filtered.wav"
        out_path      = os.path.join(uploads_dir, out_filename)
        export_filtered_wav(signal_filt, sr, out_path)

        # Graphiques comparatifs
        t_orig, t_filt, f_orig, f_filt = plot_comparison(signal, signal_filt, sr)

        # Re-plot spectre original avec bande mise en évidence
        f_orig_band = plot_spectrum(freqs, amps,
                                    title="Spectre FFT – Avant filtrage",
                                    highlight_band=(f_min, f_max),
                                    filter_type=filter_type)

        freqs_f, amps_f = compute_fft(signal_filt, sr)
        f_filt_band = plot_spectrum(freqs_f, amps_f,
                                    title="Spectre FFT – Après filtrage",
                                    highlight_band=(f_min, f_max),
                                    filter_type=filter_type)

        return jsonify({
            "success":        True,
            "plot_time_orig": t_orig,
            "plot_time_filt": t_filt,
            "plot_fft_orig":  f_orig_band,
            "plot_fft_filt":  f_filt_band,
            "download_url":   f"/filtrage/download/{out_filename}",
            "stream_url":     f"/filtrage/stream/{out_filename}",
            "filtered_file":  out_filename,
        })

    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False,
                        "message": f"Erreur filtrage : {str(e)}"}), 500


@bp.route("/download/<filename>")
def download(filename: str):
    """
    Permet le téléchargement du fichier WAV filtré.

    Args:
        filename (str): Nom du fichier WAV filtré.

    Returns:
        Response: Fichier WAV en téléchargement.
    """
    return send_from_directory(
        current_app.config["UPLOADS_DIR"],
        filename,
        as_attachment=True,
        download_name=f"signal_filtre_{filename}",
    )


@bp.route("/stream/<filename>")
def stream(filename: str):
    """
    Sert le fichier WAV filtré en streaming pour le lecteur HTML5.

    Args:
        filename (str): Nom du fichier WAV filtré.

    Returns:
        Response: Fichier audio pour lecture directe.
    """
    return send_from_directory(
        current_app.config["UPLOADS_DIR"],
        filename,
        mimetype="audio/wav",
    )
