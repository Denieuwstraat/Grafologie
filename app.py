# grafologie_app.py
from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image, ImageOps
import cv2
import streamlit as st

# -------------------------
# Styling
# -------------------------

CSS = """
<style>
.badge {display:inline-block;padding:0.25rem 0.5rem;border-radius:999px;background:#EEF2FF;color:#1E40AF;font-size:0.85rem;margin-right:6px}
.kpi {display:flex;gap:12px;flex-wrap:wrap;margin-bottom:0.5rem}
.card {border:1px solid #e5e7eb;border-radius:12px;padding:16px;background:#fff}
.small {opacity:0.75}
.hr {height:1px;background:#e5e7eb;border:none;margin:12px 0}
.section-title {margin: 0.2rem 0 0.6rem; font-weight: 700; font-size: 1.1rem;}
</style>
"""

# -------------------------
# Dataclasses
# -------------------------

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
# Helpers
# -------------------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def bytes_from_pil(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# -------------------------
# Auto-parameter selectie
# -------------------------

def estimate_noise_level(gray: np.ndarray) -> float:
    # Variantie van Laplacian ≈ scherpte/ruis-indicatie
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def try_adaptive(gray_eq: np.ndarray, block: int, C: int) -> np.ndarray:
    block = max(3, block | 1)
    bin_ = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, C)
    return (255 - bin_) // 255  # ink=1

def auto_params(gray: np.ndarray) -> Tuple[bool, int, int]:
    """
    Kies denoise + (block, C) door simpel gridsearch op
    - inktdekking target ~ 2%–12% (schrijfdruk-proxy)
    - edge-density van inktpixels (meer duidelijke randen is beter)
    """
    mean, std = gray.mean(), gray.std()
    # Voor-fuik: denoise bij hogere ruis/scherpte
    denoise = estimate_noise_level(gray) > 60 or std > 45

    gray_eq = cv2.equalizeHist(gray)

    blocks = [21, 35, 51]
    Cs = [5, 9, 13, 17]
    target = 0.06  # gewenste inktfractie
    best = None
    best_score = -1e9

    for b in blocks:
        for c in Cs:
            bin_inv = try_adaptive(gray_eq, b, c)
            cov = bin_inv.mean()
            if cov <= 0.001:
                continue
            # rand-dichtheid (hoeveel randen binnen inktgebied)
            edges = cv2.Canny(gray_eq, 30, 90)
            ink_edges = (edges > 0) & (bin_inv.astype(bool))
            edge_density = ink_edges.sum() / (bin_inv.sum() + 1e-6)

            # score: dicht bij target + voldoende randen, straft te veel/te weinig inkt
            score = -abs(cov - target) * 5.0 + edge_density * 2.5
            # lichte bias voor middelgrote blokken
            if b == 35:
                score += 0.2
            if score > best_score:
                best_score = score
                best = (b, c)

    # Fallbacks
    if best is None:
        best = (35, 12)
    block, C = best

    # Kleine correctie o.b.v. helderheid
    if mean > 170:  # erg licht
        C = min(C + 2, 25)
    elif mean < 110:  # erg donker
        C = max(C - 2, 3)

    return denoise, block, C

# -------------------------
# Preprocess & features
# -------------------------

@st.cache_data(show_spinner=False)
def preprocess_auto(img_bytes: bytes, resize_max: int = 1600) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Automatische pre-processing:
    - transpositie (EXIF)
    - resize
    - denoise-keuze
    - automatische adaptive-threshold parameters
    Return: (gray_eq, bin_inv, vis_bgr, meta_params)
    """
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_pil = ImageOps.exif_transpose(img_pil)

    w, h = img_pil.size
    if max(w, h) > resize_max:
        scale = resize_max / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    cv = pil_to_cv(img_pil)
    gray0 = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

    denoise, block, C = auto_params(gray0)
    gray = cv2.fastNlMeansDenoising(gray0, None, 7, 7, 21) if denoise else gray0

    gray_eq = cv2.equalizeHist(gray)
    bin_inv = try_adaptive(gray_eq, block, C)
    vis = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    meta = {"denoise": int(denoise), "block": int(block), "C": int(C)}
    return gray_eq, bin_inv.astype(np.uint8), vis, meta


def _connected_components_stats(bin_inv: np.ndarray):
    mask = (bin_inv * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num_labels, labels, stats, centroids


def estimate_letter_height(bin_inv: np.ndarray) -> float:
    _, _, stats, _ = _connected_components_stats(bin_inv)
    hs = []
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = stats[i]
        if 5 <= h <= 200 and 5 <= w <= 200 and 20 <= area <= 2000:
            hs.append(h)
    return float(np.median(hs)) if hs else float('nan')


def estimate_ink_coverage(bin_inv: np.ndarray) -> float:
    return float(bin_inv.mean())


def estimate_slant(gray: np.ndarray, bin_inv: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    edge_thresh = np.percentile(mag, 82)
    mask = (mag > edge_thresh) & (bin_inv.astype(bool))
    if mask.sum() < 100:
        return float('nan')
    ang = np.arctan2(gy[mask], gx[mask])
    ang = ((ang + np.pi/2 + np.pi) % np.pi) - np.pi/2
    deg = np.degrees(np.median(ang))
    return float(np.clip(deg, -45, 45))


def horizontal_projection(bin_inv: np.ndarray) -> np.ndarray:
    return bin_inv.sum(axis=1)

def vertical_projection(bin_inv: np.ndarray) -> np.ndarray:
    return bin_inv.sum(axis=0)


def estimate_line_spacing(bin_inv: np.ndarray) -> float:
    proj = horizontal_projection(bin_inv)
    if proj.max() == 0:
        return float('nan')
    thresh = np.percentile(proj, 60)
    lines = (proj > thresh).astype(np.uint8)
    gaps, run = [], 0
    for v in lines:
        if v == 0:
            run += 1
        elif run:
            gaps.append(run); run = 0
    return float(np.median(gaps)) if gaps else float('nan')


def estimate_word_spacing(bin_inv: np.ndarray) -> float:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    words = cv2.morphologyEx(bin_inv * 255, cv2.MORPH_CLOSE, kernel)
    proj = horizontal_projection((words // 255).astype(np.uint8))
    if proj.max() == 0:
        return float('nan')
    rows = np.where(proj > np.percentile(proj, 60))[0]
    if len(rows) < 3:
        return float('nan')
    step = max(1, len(rows) // 10)
    sample_rows = rows[::step]
    gaps_all = []
    for r in sample_rows:
        row = words[r, :]
        is_ink, gap = row > 0, 0
        for px in is_ink:
            if not px:
                gap += 1
            elif gap:
                gaps_all.append(gap); gap = 0
    gaps_all = [g for g in gaps_all if 3 <= g <= 200]
    return float(np.median(gaps_all)) if gaps_all else float('nan')


def estimate_margins(bin_inv: np.ndarray) -> Tuple[float, float, float, float]:
    h, w = bin_inv.shape
    cols = vertical_projection(bin_inv)
    rows = horizontal_projection(bin_inv)
    if cols.max() == 0 or rows.max() == 0:
        return (float('nan'),) * 4
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
    h, w = bin_inv.shape
    roi = bin_inv[int(0.75 * h):, :]
    if roi.size == 0:
        return 0.0
    contours = cv2.findContours((roi * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0] if len(contours) == 2 else contours[1]
    if not cnts:
        return 0.0
    scores = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        elong = max(bw, 1) / max(bh, 1)
        score = (area / (w * h * 0.25)) * 0.6 + (min(elong, 8) / 8) * 0.4
        scores.append(score)
    return float(np.clip(np.max(scores), 0, 1)) if scores else 0.0


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
# Interpretatie & profiel
# -------------------------

def bucketize(value: float, thresholds: List[float], labels: List[str]) -> str:
    if np.isnan(value):
        return "onbekend"
    idx = 0
    for t in thresholds:
        if value <= t:
            break
        idx += 1
    return labels[min(idx, len(labels) - 1)]

def interpret(meas: Measures, dpi_guess: int = 200) -> Dict[str, object]:
    px2mm = 25.4 / max(dpi_guess, 72)
    letter_mm = meas.letter_height_px * px2mm if not math.isnan(meas.letter_height_px) else float('nan')
    word_mm = meas.word_gap_px * px2mm if not math.isnan(meas.word_gap_px) else float('nan')
    line_mm = meas.line_gap_px * px2mm if not math.isnan(meas.line_gap_px) else float('nan')

    size_cat = bucketize(letter_mm, [2.5, 4.0], ["klein", "gemiddeld", "groot"])
    press_cat = bucketize(meas.ink_coverage, [0.03, 0.07], ["licht", "normaal", "zwaar"])
    if not math.isnan(meas.avg_slant_deg):
        slant_cat = "linkshellend" if meas.avg_slant_deg < -5 else ("rechtshellend" if meas.avg_slant_deg > 5 else "verticaal")
    else:
        slant_cat = "verticaal"
    wordspace_cat = bucketize(word_mm, [3.0, 6.0], ["nauw", "gemiddeld", "ruim"])
    linespace_cat = bucketize(line_mm, [4.0, 8.0], ["nauw", "gemiddeld", "ruim"])

    left, right, top, bottom = meas.margins_pct
    margin_left_cat = bucketize(left, [3.0, 8.0], ["smal", "gemiddeld", "breed"]) if not math.isnan(left) else "onbekend"
    margin_right_cat = bucketize(right, [3.0, 8.0], ["smal", "gemiddeld", "breed"]) if not math.isnan(right) else "onbekend"

    sig_cat = "waarschijnlijk aanwezig" if meas.signature_likelihood > 0.35 else "niet duidelijk"

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

    interp: Dict[str, object] = {
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


def make_profile(interp: Dict[str, object], style: str = "Mystiek", length: str = "Middel") -> str:
    size = str(interp["lettergrootte"]).split(" → ")[0]
    press = str(interp["schrijfdruk"]).split(" → ")[0]
    slant = str(interp["hellingshoek"]).split(" → ")[0]
    wordsp = str(interp["woordafstand"]).split(" → ")[0]
    linesp = str(interp["regelafstand"]).split(" → ")[0]
    leftm = str(interp["marge_links"]).split(" → ")[0]
    rightm = str(interp["marge_rechts"]).split(" → ")[0]

    base_map = {
        "klein": "U zoekt nuance en beheersing; detail is richtinggevend.",
        "gemiddeld": "U balanceert tussen nabijheid en afstand, actie en reflectie.",
        "groot": "U beweegt ruim en vrij—zichtbaarheid als vorm van verbinding.",
        "licht": "Uw energie stroomt zuinig en verfijnd; u kiest bewust waar u gewicht legt.",
        "normaal": "Een rustige continuïteit in inzet en emotie.",
        "zwaar": "U laat sporen achter—volharding is uw handtekening.",
        "linkshellend": "Eerst naar binnen, het verleden als context.",
        "verticaal": "Zelfbeheersing en nuchter kijken als kompas.",
        "rechtshellend": "U leunt naar de ander; contact trekt u vooruit.",
        "nauw": "Neiging tot nabijheid en intensiteit in het moment.",
        "ruim": "U creëert ademruimte en overzicht wanneer het druk wordt.",
        "smal": "Snelle start en scherpe afronding; liefde voor vaart.",
        "breed": "Begin en einde krijgen een ritueel; tijd is partner.",
    }

    cues = []
    for k in [size, press, slant, wordsp, linesp, leftm, rightm]:
        if k in base_map:
            cues.append(base_map[k])

    body_mystiek = (
        "Alsof de pen een seismograaf is van uw binnenwereld, tekent dit handschrift een stille landkaart van gewoontes en verlangens. "
        + " ".join(cues)
        + " Samen klinkt hier iemand die ritme zoekt tussen hoofd en hart, precisie en beweging."
    )
    body_nuchter = (
        "Het handschrift wijst op consistente gewoontes en duidelijke voorkeuren. "
        + " ".join(cues)
        + " In samenhang oogt dit als een evenwicht tussen structuur en flexibiliteit."
    )
    body_poetisch = (
        "De lijnen ademen; de marge laat licht vallen. "
        + " ".join(cues)
        + " In dit ritme wordt een stem hoorbaar die niet schreeuwt, maar draagt."
    )
    body_coach = (
        "Sterktes die opvallen: " + ", ".join(cues[:3]) + ". "
        "Kans om te groeien: benut contrast tussen nabijheid en ademruimte bewuster in drukke fases."
    )

    style_map = {
        "Mystiek": body_mystiek,
        "Nuchter analytisch": body_nuchter,
        "Poëtisch": body_poetisch,
        "Coaching": body_coach,
    }

    text = style_map.get(style, body_mystiek)
    if length == "Kort":
        text = textwrap.shorten(text, width=350, placeholder="…")
    elif length == "Lang":
        text += " Waar precisie en beweging elkaar kruisen, ontstaat de stijl die u onderscheidt—niet luid, wel helder."
    return text


# -------------------------
# UI helpers
# -------------------------

def kpi_row(items: List[Tuple[str, str]]):
    html = ["<div class='kpi'>"]
    for label, value in items:
        html.append(f"<div class='card'><div class='small'>{label}</div><div><strong>{value}</strong></div></div>")
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)

def get_image_bytes() -> Optional[bytes]:
    """
    Hoofdscherm: upload of camera naast elkaar.
    """
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Upload een foto/scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with c2:
        cam = st.camera_input("Of maak een foto met je camera")
    if cam:
        return cam.getvalue()
    if up:
        return up.read()
    return None

# -------------------------
# App
# -------------------------

def main():
    st.set_page_config(page_title="Grafologie uit afbeelding", page_icon="🖋️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("🖋️ Grafologische analyse (automatische instellingen)")
    st.caption("Niet-wetenschappelijke, heuristische analyse. Focus op vorm/stijl, niet op inhoud.")

    # Enige keuzes die je nog maakt: stijl en lengte (in hoofdscherm).
    col_style, col_len = st.columns([2, 1])
    with col_style:
        style = st.selectbox("Profiel-stijl", ["Mystiek", "Nuchter analytisch", "Poëtisch", "Coaching"], index=0)
    with col_len:
        length = st.selectbox("Tekstlengte", ["Kort", "Middel", "Lang"], index=1)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    img_bytes = get_image_bytes()
    if not img_bytes:
        st.info("📷 Upload een afbeelding of maak een foto om te starten.")
        return

    # Automatische pre-processing (geen handmatige instellingen meer nodig)
    gray, bin_inv, vis, meta = preprocess_auto(img_bytes, resize_max=1600)

    # Metingen & interpretatie
    # DPI-guessed conversie blijft 200 (stabiel) – kan later slim geschat worden o.b.v. letterhoogte.
    dpi_guess = 200
    meas = measure_all(gray, bin_inv)
    interp = interpret(meas, dpi_guess=dpi_guess)
    px2mm = 25.4 / max(dpi_guess, 72)

    # -------------------------
    # 1) PERSOONLIJKHEIDSPROFIEL (als eerste)
    # -------------------------
    st.markdown("<div class='section-title'>Persoonlijkheidsprofiel</div>", unsafe_allow_html=True)
    profile_text = make_profile(interp, style=style, length=length)
    st.markdown(f"<div class='card'><div>{profile_text}</div></div>", unsafe_allow_html=True)

    # Downloadbaar rapport
    # Chips voor korte samenvatting
    chips = [
        ("Lettergrootte", str(interp["lettergrootte"]).split(" → ")[0]),
        ("Druk", str(interp["schrijfdruk"]).split(" → ")[0]),
        ("Slant", str(interp["hellingshoek"]).split(" → ")[0]),
        ("Woord", str(interp["woordafstand"]).split(" → ")[0]),
        ("Regel", str(interp["regelafstand"]).split(" → ")[0]),
        ("Links", str(interp["marge_links"]).split(" → ")[0]),
        ("Rechts", str(interp["marge_rechts"]).split(" → ")[0]),
    ]
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
    obs_lines.append(f"• Schrijfdruk (proxy inktdekking): {meas.ink_coverage*100:.2f}%")
    obs_lines.append(f"• Handtekening-indicatie: {interp['handtekening']}")

    inter_table = {
        "Lettergrootte": interp["lettergrootte"],
        "Schrijfdruk": interp["schrijfdruk"],
        "Hellingshoek": interp["hellingshoek"],
        "Woordafstand": interp["woordafstand"],
        "Regelafstand": interp["regelafstand"],
        "Marge links": interp["marge_links"],
        "Marge rechts": interp["marge_rechts"],
    }

    md_report = (
        f"# Grafologische analyse\n\n"
        f"## Persoonlijkheidsprofiel\n{profile_text}\n\n"
        f"## Overzicht (samenvatting)\n{', '.join([f'{k}: {v}' for k, v in chips])}\n\n"
        f"## Objectieve observatie\n{chr(10).join(obs_lines)}\n\n"
        f"## Interpretatie\n" + "\n".join([f"- **{k}**: {v}" for k, v in inter_table.items()]) + "\n"
    )
    st.download_button("⬇️ Download rapport (Markdown)", data=md_report, file_name="grafologie_rapport.md")

    # -------------------------
    # 2) OVERZICHT (KPI's & chips)
    # -------------------------
    st.markdown("<div class='section-title'>Overzicht</div>", unsafe_allow_html=True)
    kpi_row([
        ("Lettergrootte", f"{(meas.letter_height_px*px2mm):.1f} mm" if not math.isnan(meas.letter_height_px) else "–"),
        ("Hellingshoek", f"{meas.avg_slant_deg:+.1f}°" if not math.isnan(meas.avg_slant_deg) else "–"),
        ("Woordafstand", f"{(meas.word_gap_px*px2mm):.1f} mm" if not math.isnan(meas.word_gap_px) else "–"),
        ("Regelafstand", f"{(meas.line_gap_px*px2mm):.1f} mm" if not math.isnan(meas.line_gap_px) else "–"),
        ("Schrijfdruk (proxy)", f"{meas.ink_coverage*100:.2f}%"),
    ])
    st.markdown(" ".join([f"<span class='badge'>{k}: <strong>{v}</strong></span>" for k, v in chips]), unsafe_allow_html=True)

    # -------------------------
    # 3) OBJECTIEVE OBSERVATIE
    # -------------------------
    st.markdown("<div class='section-title'>Objectieve observatie</div>", unsafe_allow_html=True)
    st.write("\n".join(obs_lines))

    # -------------------------
    # 4) GRAFOLOGISCHE INTERPRETATIE
    # -------------------------
    st.markdown("<div class='section-title'>Grafologische interpretatie</div>", unsafe_allow_html=True)
    st.table({k: [v] for k, v in inter_table.items()})

    # -------------------------
    # 5) Beeldmateriaal
    # -------------------------
    st.markdown("<div class='section-title'>Beeldmateriaal</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    with col1:
        st.subheader("Origineel")
        st.image(img_pil, use_container_width=True)
    with col2:
        st.subheader("Grijs + contrast")
        st.image(gray, clamp=True, use_container_width=True)
    with col3:
        st.subheader("Binarisatie (inkt in wit)")
        st.image((bin_inv * 255).astype(np.uint8), use_container_width=True)

    st.markdown("---")
    with st.expander("Technische details (automatisch gekozen)"):
        st.json({"denoise": bool(meta["denoise"]), "adaptive_block": meta["block"], "adaptive_C": meta["C"], "dpi_guess": dpi_guess})

    with st.expander("Disclaimer"):
        st.caption(
            "Deze app gebruikt heuristieken en traditionele grafologische duidingen. "
            "Het resultaat is interpretatief en niet wetenschappelijk gevalideerd. Focus op vorm en stijl, niet op inhoud."
        )

if __name__ == "__main__":
    main()
