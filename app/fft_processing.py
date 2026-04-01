"""
fft_processing.py – Analyse fréquentielle et filtrage par masque rectangulaire
ESP/UCAD – TNS Mini-Projet 2025-2026
"""

import os
import io
import base64
import numpy as np
from scipy import fft as scipy_fft
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")          # Backend non-interactif (pas de GUI)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Palette visuelle du projet ───────────────────────────────────────────────
PALETTE = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "accent":   "#00e5ff",
    "accent2":  "#ff6b6b",
    "text":     "#e6edf3",
    "grid":     "#21262d",
}


def _fig_style():
    """Applique le style sombre cohérent à la figure Matplotlib courante."""
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "text.color":        PALETTE["text"],
        "font.family":       "monospace",
    })


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Charge un fichier WAV et retourne le signal normalisé.

    Args:
        filepath (str): Chemin vers le fichier WAV.

    Returns:
        tuple: (signal float32 normalisé [-1,1], fréquence d'échantillonnage)
    """
    sample_rate, data = wavfile.read(filepath)

    # Conversion stéréo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Normalisation
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
    else:
        max_val = 1.0

    signal = data.astype(np.float32) / max_val
    return signal, sample_rate


def compute_fft(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la Transformée de Fourier Rapide (FFT) du signal.

    Args:
        signal      (np.ndarray): Signal temporel normalisé.
        sample_rate (int):        Fréquence d'échantillonnage en Hz.

    Returns:
        tuple: (freqs array Hz, amplitudes array) — côté positif uniquement.
    """
    N     = len(signal)
    X     = scipy_fft.fft(signal)
    freqs = scipy_fft.fftfreq(N, d=1.0 / sample_rate)

    # On ne conserve que les fréquences positives
    pos_mask   = freqs >= 0
    freqs_pos  = freqs[pos_mask]
    amp_pos    = np.abs(X[pos_mask]) * 2 / N   # Normalisation amplitude

    return freqs_pos, amp_pos


def apply_rectangular_filter(signal: np.ndarray, sample_rate: int,
                               f_min: float, f_max: float,
                               filter_type: str = "passband") -> np.ndarray:
    """
    Applique un masque fréquentiel rectangulaire puis reconstruit par IFFT.

    Masque passe-bande  : H(f) = 1  si f_min ≤ |f| ≤ f_max, 0 sinon
    Masque coupe-bande  : H̄(f) = 1 − H(f)

    Args:
        signal      (np.ndarray): Signal temporel source (float32 normalisé).
        sample_rate (int):        Fréquence d'échantillonnage en Hz.
        f_min       (float):      Borne inférieure du masque en Hz.
        f_max       (float):      Borne supérieure du masque en Hz.
        filter_type (str):        "passband" ou "stopband".

    Returns:
        np.ndarray: Signal filtré, normalisé entre -1 et 1.

    Raises:
        ValueError: Si filter_type est inconnu ou si f_min >= f_max.
    """
    if filter_type not in ("passband", "stopband"):
        raise ValueError(f"Type de filtre inconnu : '{filter_type}'. "
                         "Utilisez 'passband' ou 'stopband'.")
    if f_min >= f_max:
        raise ValueError(f"f_min ({f_min} Hz) doit être < f_max ({f_max} Hz).")

    N     = len(signal)
    X     = scipy_fft.fft(signal)
    freqs = scipy_fft.fftfreq(N, d=1.0 / sample_rate)

    # Construction du masque H(f) rectangulaire (bilatéral)
    abs_f = np.abs(freqs)
    H = np.where((abs_f >= f_min) & (abs_f <= f_max), 1.0, 0.0)

    # Inversion pour filtre coupe-bande
    if filter_type == "stopband":
        H = 1.0 - H

    # Application du masque et reconstruction par IFFT
    X_filtered      = X * H
    signal_filtered = np.real(scipy_fft.ifft(X_filtered)).astype(np.float32)

    # Renormalisation pour éviter l'écrêtage
    peak = np.max(np.abs(signal_filtered))
    if peak > 0:
        signal_filtered /= peak

    return signal_filtered


# ── Génération des graphiques ────────────────────────────────────────────────

def _plot_to_base64(fig) -> str:
    """
    Convertit une figure Matplotlib en chaîne Base64 (PNG).

    Args:
        fig: Figure Matplotlib.

    Returns:
        str: Image encodée en Base64.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def plot_time_signal(signal: np.ndarray, sample_rate: int,
                     title: str = "Signal temporel") -> str:
    """
    Génère le graphique amplitude = f(temps).

    Args:
        signal      (np.ndarray): Signal normalisé.
        sample_rate (int):        Fréquence d'échantillonnage.
        title       (str):        Titre du graphique.

    Returns:
        str: Image PNG encodée en Base64.
    """
    _fig_style()
    t   = np.linspace(0, len(signal) / sample_rate, len(signal))
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(t, signal, color=PALETTE["accent"], linewidth=0.6, alpha=0.85)
    ax.set_title(title, color=PALETTE["accent"], fontsize=11, pad=8)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _plot_to_base64(fig)


def plot_spectrum(freqs: np.ndarray, amplitudes: np.ndarray,
                  title: str = "Spectre FFT",
                  highlight_band: tuple | None = None,
                  filter_type: str = "passband") -> str:
    """
    Génère le graphique du spectre d'amplitude |X(f)|.

    Args:
        freqs         (np.ndarray): Tableau des fréquences positives en Hz.
        amplitudes    (np.ndarray): Amplitudes normalisées.
        title         (str):        Titre du graphique.
        highlight_band (tuple):     (f_min, f_max) à mettre en évidence.
        filter_type   (str):        "passband" ou "stopband" (couleur de zone).

    Returns:
        str: Image PNG encodée en Base64.
    """
    _fig_style()
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(freqs, amplitudes, color=PALETTE["accent"], linewidth=0.7, alpha=0.9)

    if highlight_band:
        fmin, fmax = highlight_band
        color = PALETTE["accent"] if filter_type == "passband" else PALETTE["accent2"]
        ax.axvspan(fmin, fmax, alpha=0.18, color=color,
                   label=f"Bande {filter_type} [{fmin}–{fmax} Hz]")
        ax.axvline(fmin, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axvline(fmax, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.legend(fontsize=8, loc="upper right",
                  facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"])

    ax.set_title(title, color=PALETTE["accent"], fontsize=11, pad=8)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("|X(f)|")
    ax.set_xlim(0, freqs[-1])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _plot_to_base64(fig)


def plot_comparison(signal_orig: np.ndarray, signal_filt: np.ndarray,
                    sample_rate: int) -> tuple[str, str, str, str]:
    """
    Génère les quatre graphiques de comparaison avant/après filtrage.

    Args:
        signal_orig (np.ndarray): Signal original.
        signal_filt (np.ndarray): Signal filtré.
        sample_rate (int):        Fréquence d'échantillonnage.

    Returns:
        tuple: (b64_time_orig, b64_time_filt, b64_fft_orig, b64_fft_filt)
    """
    b64_time_orig = plot_time_signal(signal_orig, sample_rate, "Signal original – Temporel")
    b64_time_filt = plot_time_signal(signal_filt, sample_rate, "Signal filtré – Temporel")

    freqs_o, amp_o = compute_fft(signal_orig, sample_rate)
    freqs_f, amp_f = compute_fft(signal_filt, sample_rate)

    b64_fft_orig = plot_spectrum(freqs_o, amp_o, "Spectre FFT – Avant filtrage")
    b64_fft_filt = plot_spectrum(freqs_f, amp_f, "Spectre FFT – Après filtrage")

    return b64_time_orig, b64_time_filt, b64_fft_orig, b64_fft_filt


def export_filtered_wav(signal_filtered: np.ndarray, sample_rate: int,
                         output_path: str) -> str:
    """
    Exporte le signal filtré dans un fichier WAV 16 bits.

    Args:
        signal_filtered (np.ndarray): Signal filtré normalisé.
        sample_rate     (int):        Fréquence d'échantillonnage.
        output_path     (str):        Chemin du fichier de sortie.

    Returns:
        str: Chemin absolu du fichier exporté.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pcm = (signal_filtered * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, pcm)
    return output_path
