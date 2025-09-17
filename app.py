"""
Streamlit-app: Grafologische analyse van handschrift uit een geüploade afbeelding

Wat het doet
- Neemt een afbeelding (foto/scan) met handschrift aan
- Voert eenvoudige computer vision-metingen uit (heuristisch):
  * lettergrootte (schatting via connected components)
  * schrijfdruk (proxy via gemiddelde streek-donkerte / inktdekking)
  * hellingshoek (slant) van letters (proxy via gradient-orientaties)
  * woord- en regelspatiëring (via projectieprofielen en tussenruimtes)
  * paginamarges links/rechts/boven/onder
  * optionele handtekening-indicatie (grote, vloeiende component onderaan)
- Zet metingen om naar traditionele grafologische interpretaties (rule-based)
- Schrijft een samenhangend, licht mystiek profiel
- Sluit af met 2–3 reflectieve vragen

Benodigdheden
pip install streamlit opencv-python pillow numpy scikit-image

Starten
streamlit run grafologie_app.py
"""

from __future__ import annotations
import io
import math
import textwrap
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize
import cv2
import streamlit as st

# -------------------------
# Hulpfuncties beeldbewerking
# -------------------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def preprocess(img_pil: Image.Image, resize_max: int = 1600) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gray, bin_inv, vis) as uint8 arrays.
    - gray: grayscale
    - bin_inv: binary (ink=1, background=0)
    - vis: contrast-stretched preview
    """
    img_pil = ImageOps.exif_transpose(img_pil)
    # Downscale for speed, preserve aspect
    w, h = img_pil.size
    scale = 1.0
    if max(w, h) > resize_max:
        scale = resize_max / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    cv = pil_to_cv(img_pil)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # Contrast normalize
    gray_eq = cv2.equalizeHist(gray)
    # Adaptive threshold (robust for lighting)
    bin_ = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 12)
    bin_inv = (255 - bin_) // 255  # 1=ink, 0=bg
    vis = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    return gray_eq, bin_inv.astype(np.uint8), vis


@dataclass
class Measures:
    letter_height_px: float
    ink_coverage: float
    avg_slant_deg: float
    word_gap_px: float
    line_gap_px: float
    margins_pct: Tuple[float, float, float, float]  # left, right, top, bottom (0-100)
    signature_likelihood: float  # 0..1


# -------------------------
# Metingen (heuristieken)
# -------------------------

def estimate_letter_height(bin_inv: np.ndarray) -> float:
    # Connected components
    labels, stats = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)[1:3]
    # stats: [label, x, y, w, h, area] but OpenCV returns separate arrays; we use stats
    # Filter noise and huge components
    hs = []
    for s in stats:
        x, y, w, h, area = s
        if 5 <= h <= 200 and 5 <= w <= 200 and 20 <= area <= 2000:
            hs.append(h)
    if not hs:
        return float('nan')
    return float(np.median(hs))


def estimate_ink_coverage(bin_inv: np.ndarray) -> float:
    return float(bin_inv.mean())  # fraction of ink pixels


def estimate_slant(gray: np.ndarray, bin_inv: np.ndarray) -> float:
    # Use gradients only where ink is present
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)  # radians
    mask = (mag > np.percentile(mag, 75)) & (bin_inv.astype(bool))
    if mask.sum() < 100:
        return float('nan')
    # Convert to letter slant: vertical down is pi/2; we want italic angle relative to vertical.
    angles = ang[mask]
    # Map to [-pi/2, pi/2]
    angles = ((angles + np.pi/2 + np.pi) % np.pi) - np.pi/2
    # Convert to degrees; positive = right slant
    deg = np.degrees(np.median(angles))
    # Clamp to reasonable range
    return float(np.clip(deg, -45, 45))


def horizontal_projection(bin_inv: np.ndarray) -> np.ndarray:
    return bin_inv.sum(axis=1)


def vertical_projection(bin_inv: np.ndarray) -> np.ndarray:
    return bin_inv.sum(axis=0)


def estimate_line_spacing(bin_inv: np.ndarray) -> float:
    proj = horizontal_projection(bin_inv)
    thresh = np.percentile(proj, 60)
    lines = (proj > thresh).astype(np.uint8)
    # Find gaps (runs of zeros between lines)
    gaps = []
    run = 0
    for v in lines:
        if v == 0:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    if not gaps:
        return float('nan')
    return float(np.median(gaps))


def estimate_word_spacing(bin_inv: np.ndarray) -> float:
    # Morph close to merge letters into words, then measure gaps along text rows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    words = cv2.morphologyEx(bin_inv * 255, cv2.MORPH_CLOSE, kernel)
    proj = horizontal_projection((words // 255).astype(np.uint8))
    rows = np.where(proj > np.percentile(proj, 60))[0]
    if len(rows) < 3:
        return float('nan')
    # Sample a few rows evenly
    sample_rows = rows[::max(1, len(rows)//10)]
    gaps_all = []
    for r in sample_rows:
        row = words[r, :]
        # find gaps between word blobs
        is_ink = row > 0
        gap = 0
        for px in is_ink:
            if not px:
                gap += 1
            elif gap:
                gaps_all.append(gap)
                gap = 0
    gaps_all = [g for g in gaps_all if 3 <= g <= 200]
    if not gaps_all:
        return float('nan')
    return float(np.median(gaps_all))


def estimate_margins(bin_inv: np.ndarray) -> Tuple[float, float, float, float]:
    h, w = bin_inv.shape
    cols = vertical_projection(bin_inv)
    rows = horizontal_projection(bin_inv)
    # left margin: first column index with ink after a run of zeros
    try:
        left = int(np.argmax(cols > 0))
        right = int(w - np.argmax(cols[::-1] > 0) - 1)
        top = int(np.argmax(rows > 0))
        bottom = int(h - np.argmax(rows[::-1] > 0) - 1)
    except Exception:
        return (float('nan'),) * 4
    left_pct = 100.0 * left / w
    right_pct = 100.0 * (w - right - 1) / w
    top_pct = 100.0 * top / h
    bottom_pct = 100.0 * (h - bottom - 1) / h
    return (left_pct, right_pct, top_pct, bottom_pct)


def estimate_signature(bin_inv: np.ndarray) -> float:
    # Heuristic: look in bottom 25% for a large, elongated, curvy component
    h, w = bin_inv.shape
    roi = bin_inv[int(0.75 * h):, :]
    if roi.size == 0:
        return 0.0
    cnts, _ = cv2.findContours((roi * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    scores = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        elong = max(bw, 1) / max(bh, 1)
        per = cv2.arcLength(c, True)
        score = (area / (w * h * 0.25)) * 0.6 + (min(elong, 8) / 8) * 0.4
        scores.append(score)
    if not scores:
        return 0.0
    return float(np.clip(np.max(scores), 0, 1))


def measure_all(gray: np.ndarray, bin_inv: np.ndarray) -> Measures:
    return Measures(
        letter_height_px=estimate_letter_height(bin_inv),
        ink_coverage=estimate_ink_coverage(bin_inv),
        avg_slant_deg=estimate_slant(gray, bin_inv),
        word_gap_px=estimate_word_spacing(bin_inv),
        line_gap_px=estimate_line_spacing(bin_inv),
        margins_pct=estimate_margins(bin_inv),
        signature_likelihood=estimate_signature(bin_inv),
    )

# -------------------------
# Interpretatie (rule-based)
# -------------------------

def bucketize(value: float, thresholds: List[float], labels: List[str]) -> str:
    if np.isnan(value):
        return "onbekend"
    idx = 0
    for t in thresholds:
        if value <= t:
            break
        idx += 1
    idx = min(idx, len(labels)-1)
    return labels[idx]


def interpret(meas: Measures, dpi_guess: int = 200) -> Dict[str, str]:
    # Converteer pixels naar mm-schatting obv DPI-guess
    px2mm = 25.4 / max(dpi_guess, 72)
    letter_mm = meas.letter_height_px * px2mm if not math.isnan(meas.letter_height_px) else float('nan')
    word_mm = meas.word_gap_px * px2mm if not math.isnan(meas.word_gap_px) else float('nan')
    line_mm = meas.line_gap_px * px2mm if not math.isnan(meas.line_gap_px) else float('nan')

    size_cat = bucketize(letter_mm, [2.5, 4.0], ["klein", "gemiddeld", "groot"])  # x-hoogte ~2-4mm
    press_cat = bucketize(meas.ink_coverage, [0.03, 0.07], ["licht", "normaal", "zwaar"])  # grove proxy
    slant_cat = (
        "linkshellend" if not math.isnan(meas.avg_slant_deg) and meas.avg_slant_deg < -5 else
        "rechtshellend" if not math.isnan(meas.avg_slant_deg) and meas.avg_slant_deg > 5 else
        "verticaal"
    )
    wordspace_cat = bucketize(word_mm, [3.0, 6.0], ["nauw", "gemiddeld", "ruim"])  # mm
    linespace_cat = bucketize(line_mm, [4.0, 8.0], ["nauw", "gemiddeld", "ruim"])  # mm

    left, right, top, bottom = meas.margins_pct
    margin_left_cat = bucketize(left, [3.0, 8.0], ["smal", "gemiddeld", "breed"]) if not math.isnan(left) else "onbekend"
    margin_right_cat = bucketize(right, [3.0, 8.0], ["smal", "gemiddeld", "breed"]) if not math.isnan(right) else "onbekend"

    sig_cat = (
        "waarschijnlijk aanwezig" if meas.signature_likelihood > 0.35 else
        "niet duidelijk"
    )

    # Mapping naar traditionele duidingen (niet-wetenschappelijk)
    meanings = {
        "lettergrootte": {
            "klein": "detailgericht, geconcentreerd, soms gereserveerd",
            "gemiddeld": "in balans tussen overzicht en detail",
            "groot": "expressief, behoefte aan ruimte en zichtbaarheid",
        },
        "schrijfdruk": {
            "licht": "gevoelig, energiebesparend, mogelijk snel vermoeid",
            "normaal": "stabiele energie en emotionele balans",
            "zwaar": "intens, volhardend, sterke indruk willen maken",
        },
        "hellingshoek": {
            "linkshellend": "terughoudendheid, reflectie, behoedzaamheid",
            "verticaal": "rationeel, zelfbeheerst, objectief",
            "rechtshellend": "sociaal, spontaan, naar buiten gericht",
        },
        "woordafstand": {
            "nauw": "neiging tot nabijheid en betrokkenheid",
            "gemiddeld": "gezonde afbakening, nette ordening",
            "ruim": "behoefte aan eigen ruimte en overzicht",
        },
        "regelafstand": {
            "nauw": "drukke geest, snel schakelen",
            "gemiddeld": "ritme en structuur",
            "ruim": "ademruimte, bedachtzaamheid",
        },
        "marges": {
            "links_smal": "impulsief starten, snel van de blokken",
            "links_breed": "voorzichtig begin, planmatig",
            "rechts_smal": "drang om af te ronden, resultaatgericht",
            "rechts_breed": "tijd nemen, afronding laten rijpen",
        },
    }

    interp: Dict[str, str] = {
        "lettergrootte": f"{size_cat} → {meanings['lettergrootte'].get(size_cat, '')}",
        "schrijfdruk": f"{press_cat} → {meanings['schrijfdruk'].get(press_cat, '')}",
        "hellingshoek": f"{slant_cat} → {meanings['hellingshoek'].get(slant_cat, '')}",
        "woordafstand": f"{wordspace_cat} → {meanings['woordafstand'].get(wordspace_cat, '')}",
        "regelafstand": f"{linespace_cat} → {meanings['regelafstand'].get(linespace_cat, '')}",
        "marge_links": f"{margin_left_cat} → " + (
            meanings['marges']['links_smal'] if margin_left_cat=="smal" else
            meanings['marges']['links_breed'] if margin_left_cat=="breed" else "neutraal"
        ),
        "marge_rechts": f"{margin_right_cat} → " + (
            meanings['marges']['rechts_smal'] if margin_right_cat=="smal" else
            meanings['marges']['rechts_breed'] if margin_right_cat=="breed" else "neutraal"
        ),
        "handtekening": sig_cat,
        "_mm": {
            "letter_mm": letter_mm,
            "word_mm": word_mm,
            "line_mm": line_mm,
        }
    }
    return interp


# -------------------------
# Profiel-generator
# -------------------------

def make_profile(interp: Dict[str, str]) -> str:
    size = interp["lettergrootte"].split(" → ")[0]
    press = interp["schrijfdruk"].split(" → ")[0]
    slant = interp["hellingshoek"].split(" → ")[0]
    wordsp = interp["woordafstand"].split(" → ")[0]
    linesp = interp["regelafstand"].split(" → ")[0]
    leftm = interp["marge_links"].split(" → ")[0]
    rightm = interp["marge_rechts"].split(" → ")[0]

    pieces = []
    pieces.append(
        "Alsof de pen een seismograaf is van uw binnenwereld, zo tekent dit handschrift een kaart van gewoontes en verlangens. "
        f"De letters zijn {size}, de druk is {press}, en de hellingshoek is overwegend {slant}. "
        f"De woorden staan {wordsp} van elkaar, de regels {linesp}; de marge links is {leftm} en rechts {rightm}."
    )

    # Voeg interpretatieve draad toe
    meanings = {
        "klein": "U zoekt nuance en beheersing; detail is geen decor, maar richtingaanwijzer.",
        "gemiddeld": "U balanceert tussen nabijheid en afstand, actie en reflectie.",
        "groot": "U beweegt ruim en vrij—zichtbaarheid is voor u ook verbinding.",
        "licht": "Uw energie stroomt zuinig en verfijnd; u kiest zorgvuldig waar u gewicht legt.",
        "normaal": "Er is een rustige continuïteit in uw inzet en emotie.",
        "zwaar": "U gaat ergens voor, laat sporen achter—volharding is uw handtekening.",
        "linkshellend": "Het hart kijkt eerst naar binnen; het verleden levert context.",
        "verticaal": "Zelfbeheersing en nuchter kijken zijn uw kompas.",
        "rechtshellend": "U leunt naar de ander; het contact trekt u vooruit.",
        "nauw": "U zoekt nabijheid en intensiteit in het moment.",
        "ruim": "U creëert ademruimte en overzicht, ook wanneer het druk wordt.",
        "smal": "Snelle start, scherpe afronding; u houdt van vaart.",
        "breed": "U gunt het begin en einde hun ritueel; tijd is partner, geen tegenstander.",
    }

    for k in [size, press, slant, wordsp, linesp, leftm, rightm]:
        if k in meanings:
            pieces.append(meanings[k])

    pieces.append(
        "Samen geweven klinkt hier de stem van iemand die zichzelf observeert terwijl hij handelt; "
        "iemand die ritme zoekt tussen hoofd en hart, tussen precisie en beweging."
    )

    # Reflectieve vragen
    reflect = [
        "Waar vraagt uw leven nu om meer ruimte—en waar juist om verfijning?",
        "Op welke momenten kiest u voor vaart, en wanneer verdient afronden meer tijd?",
        "Welke relatie wil u actiever voeden: die met uzelf (reflectie) of die met de ander (expressie)?",
    ]

    return "\n\n".join(pieces) + "\n\n" + "\n".join(f"• {q}" for q in reflect)


# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.set_page_config(page_title="Grafologie uit afbeelding", page_icon="🖋️", layout="wide")
    st.title("🖋️ Grafologische analyse uit een handschriftafbeelding")
    st.caption("Niet-wetenschappelijke, heuristische analyse. Focus op vorm/stijl, niet op inhoud.")

    with st.sidebar:
        st.header("Instellingen")
        resize_max = st.slider("Max breedte/hoogte (px)", 800, 2400, 1600, 100)
        dpi_guess = st.slider("DPI schatting (voor mm)", 100, 400, 200, 25)

    file = st.file_uploader("Upload een foto/scan met handschrift (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if not file:
        st.info("⬆️ Upload een afbeelding om te starten.")
        return

    img = Image.open(file).convert("RGB")
    gray, bin_inv, vis = preprocess(img, resize_max=resize_max)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.subheader("Origineel")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("Grijs + contrast")
        st.image(gray, clamp=True, use_container_width=True)
    with col3:
        st.subheader("Binarisatie (inkt in wit)")
        st.image((bin_inv*255).astype(np.uint8), use_container_width=True)

    meas = measure_all(gray, bin_inv)
    interp = interpret(meas, dpi_guess=dpi_guess)

    st.markdown("---")
    st.header("1) Objectieve observatie")
    px2mm = 25.4 / max(dpi_guess, 72)

    obs_lines = []
    if not math.isnan(meas.letter_height_px):
        obs_lines.append(f"• Lettergrootte (schatting): {meas.letter_height_px*px2mm:.1f} mm")
    if not math.isnan(meas.avg_slant_deg):
        obs_lines.append(f"• Hellingshoek (mediaan): {meas.avg_slant_deg:+.1f}°")
    if not math.isnan(meas.word_gap_px):
        obs_lines.append(f"• Gem. woordafstand: {meas.word_gap_px*px2mm:.1f} mm")
    if not math.isnan(meas.line_gap_px):
        obs_lines.append(f"• Gem. regelafstand: {meas.line_gap_px*px2mm:.1f} mm")
    l, r, t, b = meas.margins_pct
    if not math.isnan(l):
        obs_lines.append(f"• Marges: links {l:.1f}%, rechts {r:.1f}%, boven {t:.1f}%, onder {b:.1f}%")
    obs_lines.append(f"• Schrijfdruk (proxy inktdekking): {meas.ink_coverage*100:.1f}%")
    obs_lines.append(f"• Handtekening-indicatie: {interp['handtekening']}")

    st.write("\n".join(obs_lines))

    st.header("2) Grafologische interpretatie (traditioneel)")
    inter_table = {
        "Lettergrootte": interp["lettergrootte"],
        "Schrijfdruk": interp["schrijfdruk"],
        "Hellingshoek": interp["hellingshoek"],
        "Woordafstand": interp["woordafstand"],
        "Regelafstand": interp["regelafstand"],
        "Marge links": interp["marge_links"],
        "Marge rechts": interp["marge_rechts"],
    }
    st.table({k: [v] for k, v in inter_table.items()})

    st.header("3) Persoonlijkheidsprofiel (verhalend)")
    st.write(make_profile(interp))

    st.header("4) Reflectieve vragen")
    st.markdown("\n".join([
        "• Waar vraagt uw leven nu om meer ruimte—en waar juist om verfijning?",
        "• Op welke momenten kiest u voor vaart, en wanneer verdient afronden meer tijd?",
        "• Welke relatie wilt u actiever voeden: die met uzelf (reflectie) of die met de ander (expressie)?",
    ]))

    st.markdown("---")
    with st.expander("Disclaimer"):
        st.caption(
            "Deze app gebruikt heuristieken en traditionele grafologische duidingen. "
            "Het resultaat is interpretatief en niet wetenschappelijk gevalideerd. Focus op vorm en stijl, niet op inhoud."
        )


if __name__ == "__main__":
    main()

