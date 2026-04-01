"""
audio_processing.py – Numérisation, segmentation et utilitaires audio
ESP/UCAD – TNS Mini-Projet 2025-2026
"""

import os
import io
import wave
import struct
import numpy as np
from scipy.io import wavfile


# ── Constantes autorisées ───────────────────────────────────────────────────
ALLOWED_SAMPLE_RATES = [16000, 22050, 44100]
ALLOWED_BIT_DEPTHS   = [16, 32]


def validate_params(sample_rate: int, bit_depth: int) -> None:
    """
    Valide les paramètres de numérisation.

    Args:
        sample_rate (int): Fréquence d'échantillonnage en Hz.
        bit_depth   (int): Profondeur de codage en bits.

    Raises:
        ValueError: Si l'un des paramètres est invalide.
    """
    if sample_rate not in ALLOWED_SAMPLE_RATES:
        raise ValueError(
            f"Fréquence invalide : {sample_rate} Hz. "
            f"Valeurs autorisées : {ALLOWED_SAMPLE_RATES}"
        )
    if bit_depth not in ALLOWED_BIT_DEPTHS:
        raise ValueError(
            f"Codage invalide : {bit_depth} bits. "
            f"Valeurs autorisées : {ALLOWED_BIT_DEPTHS}"
        )


def save_wav(audio_data: bytes, sample_rate: int, bit_depth: int, filepath: str) -> str:
    """
    Sauvegarde des données PCM brutes dans un fichier WAV.

    Args:
        audio_data  (bytes): Données audio PCM brutes.
        sample_rate (int):   Fréquence d'échantillonnage en Hz.
        bit_depth   (int):   Profondeur de codage (16 ou 32 bits).
        filepath    (str):   Chemin complet du fichier de sortie.

    Returns:
        str: Chemin absolu du fichier WAV créé.

    Raises:
        ValueError: Si les paramètres sont invalides.
    """
    validate_params(sample_rate, bit_depth)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Conversion bytes → tableau numpy selon la profondeur
    dtype = np.int16 if bit_depth == 16 else np.int32
    samples = np.frombuffer(audio_data, dtype=np.int8)

    # Rééchantillonnage du tableau en entiers selon la profondeur cible
    if bit_depth == 16:
        pcm = np.frombuffer(audio_data, dtype=np.int16)
    else:
        # WebAudio renvoie du Float32 ; on le convertit en Int32
        float_samples = np.frombuffer(audio_data, dtype=np.float32)
        pcm = (float_samples * 2**31).astype(np.int32)

    wavfile.write(filepath, sample_rate, pcm)
    return filepath


def save_wav_from_float32(float_data: np.ndarray, sample_rate: int,
                           bit_depth: int, filepath: str) -> str:
    """
    Sauvegarde un signal Float32 (normalisé -1..1) en WAV 16 ou 32 bits.

    Args:
        float_data  (np.ndarray): Signal normalisé entre -1.0 et 1.0.
        sample_rate (int):        Fréquence d'échantillonnage en Hz.
        bit_depth   (int):        Profondeur de codage (16 ou 32).
        filepath    (str):        Chemin de destination.

    Returns:
        str: Chemin du fichier créé.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if bit_depth == 16:
        pcm = (float_data * 32767).astype(np.int16)
    else:
        pcm = (float_data * 2147483647).astype(np.int32)
    wavfile.write(filepath, sample_rate, pcm)
    return filepath


def build_db_path(base_dir: str, locuteur: str,
                  session: str, filename: str) -> str:
    """
    Construit le chemin hiérarchique de la base de données audio.

    Structure : base_dir/locuteur_XX/session_YY/filename.wav

    Args:
        base_dir  (str): Répertoire racine de la base.
        locuteur  (str): Identifiant du locuteur (ex. "01").
        session   (str): Identifiant de session (ex. "01").
        filename  (str): Nom du fichier WAV.

    Returns:
        str: Chemin complet du fichier.
    """
    path = os.path.join(
        base_dir,
        f"locuteur_{locuteur.zfill(2)}",
        f"session_{session.zfill(2)}",
        filename,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def generate_wav_filename(sample_rate: int, bit_depth: int,
                           index: int = 1) -> str:
    """
    Génère un nom de fichier normalisé selon la convention du projet.

    Args:
        sample_rate (int): Fréquence d'échantillonnage en Hz.
        bit_depth   (int): Profondeur de codage en bits.
        index       (int): Numéro d'enregistrement.

    Returns:
        str: Nom de fichier, ex. "enreg_001_16kHz_16b.wav".
    """
    rate_label = f"{sample_rate // 1000}kHz" if sample_rate in [16000, 32000, 44100] else "22kHz"
    if sample_rate == 22050:
        rate_label = "22kHz"
    elif sample_rate == 44100:
        rate_label = "44kHz"
    elif sample_rate == 16000:
        rate_label = "16kHz"
    return f"enreg_{index:03d}_{rate_label}_{bit_depth}b.wav"


def list_database(base_dir: str) -> list:
    """
    Parcourt récursivement la base de données et liste les fichiers WAV.

    Args:
        base_dir (str): Répertoire racine de la base.

    Returns:
        list[dict]: Liste de dicts {locuteur, session, filename, path, size}.
    """
    entries = []
    if not os.path.isdir(base_dir):
        return entries

    for locuteur in sorted(os.listdir(base_dir)):
        loc_path = os.path.join(base_dir, locuteur)
        if not os.path.isdir(loc_path):
            continue
        for session in sorted(os.listdir(loc_path)):
            ses_path = os.path.join(loc_path, session)
            if not os.path.isdir(ses_path):
                continue
            for fname in sorted(os.listdir(ses_path)):
                if fname.lower().endswith(".wav"):
                    fpath = os.path.join(ses_path, fname)
                    entries.append({
                        "locuteur": locuteur,
                        "session":  session,
                        "filename": fname,
                        "path":     fpath,
                        "size_kb":  round(os.path.getsize(fpath) / 1024, 1),
                    })
    return entries


# ── Segmentation ────────────────────────────────────────────────────────────

def segment_audio(filepath: str, segments_dir: str,
                  amplitude_threshold: float = 0.02,
                  min_silence_ms: int = 300) -> list:
    """
    Segmente un fichier WAV en découpant sur les silences.

    Algorithme :
        1. Lecture et normalisation du signal.
        2. Calcul de l'énergie sur des fenêtres glissantes.
        3. Détection des zones de silence (énergie < seuil).
        4. Extraction et sauvegarde des segments vocaux.

    Args:
        filepath            (str):   Chemin du fichier WAV source.
        segments_dir        (str):   Répertoire de sortie des segments.
        amplitude_threshold (float): Seuil d'amplitude (0-1) en dessous
                                     duquel on considère un silence.
        min_silence_ms      (int):   Durée minimale d'un silence en ms.

    Returns:
        list[dict]: Chaque dict contient {filename, duration_s, path, url}.
    """
    os.makedirs(segments_dir, exist_ok=True)

    # Lecture du fichier WAV
    sample_rate, data = wavfile.read(filepath)

    # Mono : si stéréo, on moyenne les canaux
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Normalisation en float32 entre -1 et 1
    max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
    signal = data.astype(np.float32) / max_val

    # Taille de fenêtre = durée minimale de silence en échantillons
    window_size = int(sample_rate * min_silence_ms / 1000)
    window_size = max(window_size, 1)

    # Calcul de l'amplitude RMS par fenêtre
    n_windows = len(signal) // window_size
    is_voiced = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        chunk = signal[i * window_size: (i + 1) * window_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        is_voiced[i] = rms > amplitude_threshold

    # Regroupement des fenêtres voisines voisées en segments
    segments_windows = []
    in_segment = False
    start = 0

    for i, voiced in enumerate(is_voiced):
        if voiced and not in_segment:
            start = i
            in_segment = True
        elif not voiced and in_segment:
            segments_windows.append((start, i))
            in_segment = False
    if in_segment:
        segments_windows.append((start, n_windows))

    # Sauvegarde de chaque segment
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    results = []

    for idx, (w_start, w_end) in enumerate(segments_windows):
        s_start = w_start * window_size
        s_end   = min(w_end * window_size, len(signal))
        segment = signal[s_start:s_end]

        if len(segment) < sample_rate * 0.1:   # Ignore segments < 100 ms
            continue

        seg_filename = f"{base_name}_seg{idx + 1:03d}.wav"
        seg_path     = os.path.join(segments_dir, seg_filename)
        save_wav_from_float32(segment, sample_rate, 16, seg_path)

        duration = len(segment) / sample_rate
        results.append({
            "filename":   seg_filename,
            "duration_s": round(duration, 2),
            "path":       seg_path,
        })

    return results
