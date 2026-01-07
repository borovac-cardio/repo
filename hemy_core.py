# hemy_core.py
# Hemy – Hemodynamic RHC Calculator (Core)
# Author: Josip A. Borovac, MD, PhD

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

HUFNER = 1.34
DYNE_PER_WU = 80.0
RVSWI_FACTOR = 0.0136  # RVSWI = SVI*(mPAP-RAP)*0.0136

DEFAULT_INSTITUTION = "Department of Cardiovascular Diseases, University Hospital of Split"


def safe_div(n: float, d: float) -> float:
    return n / d if abs(d) > 1e-12 else float("nan")


def is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    if height_cm <= 0 or weight_kg <= 0:
        return float("nan")
    return math.sqrt((height_cm * weight_kg) / 3600.0)


def mean_from_sys_dia(sys_p: float, dia_p: float) -> float:
    return dia_p + (sys_p - dia_p) / 3.0


def map_from_sbp_dbp(sbp: float, dbp: float) -> float:
    return dbp + (sbp - dbp) / 3.0


def hb_gL_to_gdL(hb_g_L: float) -> Tuple[float, float, bool]:
    """Returns (hb_g_L_corrected, hb_g_dL, corrected_flag).
    If user accidentally entered g/dL (e.g., 14), auto-convert to g/L (140)."""
    corrected = False
    hb = hb_g_L
    if hb_g_L > 0 and hb_g_L < 40:
        hb = hb_g_L * 10.0
        corrected = True
    return hb, hb / 10.0, corrected


def o2_content_ml_per_dl(hb_g_dl: float, sat_frac: float) -> float:
    return HUFNER * hb_g_dl * sat_frac


def pick_mixed_venous_sat(
    pa: Optional[float],
    ra: Optional[float],
    rv: Optional[float],
    svc: Optional[float],
    ivc: Optional[float],
) -> Tuple[float, str]:
    if pa is not None:
        return pa, "PA"
    if ra is not None:
        return ra, "RA"
    if svc is not None and ivc is not None:
        return (2.0 * ivc + 1.0 * svc) / 3.0, "weighted(2/3 IVC + 1/3 SVC)"
    if rv is not None:
        return rv, "RV"
    return 75.0, "default(75%)"


def pvr_severity(pvr_wu: float) -> str:
    if is_nan(pvr_wu):
        return "N/A"
    if pvr_wu >= 5.0:
        return "SEVERE (≥5 WU)"
    if pvr_wu > 2.0:
        return "ELEVATED (>2 WU)"
    return "NORMAL (≤2 WU)"


def interpret_ph(mpap: float, pcwp: float, pvr_wu: float) -> str:
    if is_nan(mpap) or is_nan(pcwp) or is_nan(pvr_wu):
        return "Unable to classify PH (missing/invalid inputs)."
    if mpap <= 20:
        return f"No PH by hemodynamics: mPAP {mpap:.1f} mmHg (≤ 20)."
    if pcwp <= 15:
        if pvr_wu > 2:
            return f"Pre-capillary PH: mPAP>20, PCWP≤15, PVR>2 (PVR={pvr_wu:.2f})."
        return "PH with PCWP≤15 but PVR≤2 (borderline/flow-related; interpret clinically)."
    # pcwp > 15
    if pvr_wu > 2:
        return f"Combined post- and pre-capillary PH (CpcPH): PCWP>15 and PVR>2 (PVR={pvr_wu:.2f})."
    return f"Isolated post-capillary PH (IpcPH): PCWP>15 and PVR≤2 (PVR={pvr_wu:.2f})."


def treatment_block(mpap: float, pcwp: float, pvr_wu: float) -> str:
    # High-level haemodynamic-phenotype suggestions (final decisions depend on PH group + full work-up).
    if is_nan(mpap) or is_nan(pcwp) or is_nan(pvr_wu):
        return "Treatment: insufficient data for phenotype-based suggestions."

    lines = ["Treatment options (high-level, phenotype-based):"]
    if mpap <= 20:
        lines.append("- No haemodynamic PH: treat underlying condition; follow clinically.")
        return "\n".join(lines)

    lines.append("- General: diuretics if congested; oxygen if hypoxaemic; rehab when stable; manage comorbidities; consider PH expert-centre referral.")

    # Pre-cap
    if pcwp <= 15 and pvr_wu > 2:
        lines.append("- Pre-capillary PH: complete work-up to define PH group (PAH vs lung/hypoxia vs CTEPH vs other) before targeted therapy.")
        lines.append("- If PAH confirmed: risk-based therapy (often initial dual oral combination; escalate by reassessment).")
        lines.append("- If high-risk PAH/severe haemodynamics: consider parenteral prostacyclin strategy in expert centre; consider transplant pathway if inadequate response.")
        lines.append("- If CTEPH: anticoagulation + refer to CTEPH team (PEA if operable; BPA if not; medical therapy for inoperable/persistent).")
        return "\n".join(lines)

    # Post-cap
    if pcwp > 15 and pvr_wu <= 2:
        lines.append("- IpcPH (PH-LHD): optimize left-heart disease/valvular strategy first; PAH drugs generally not recommended.")
        return "\n".join(lines)

    if pcwp > 15 and pvr_wu > 2:
        lines.append("- CpcPH: optimize left-heart disease first; refer to HF/PH expert centre if advanced HF/RV dysfunction.")
        if pvr_wu >= 5:
            lines.append("- PVR ≥ 5 WU suggests severe pulmonary vascular disease; prioritize expert-centre management and advanced HF pathways where appropriate.")
        lines.append("- Targeted PAH drugs are not routinely recommended in PH-LHD; any use should be individualized in expert centre.")
        return "\n".join(lines)

    lines.append("- Mixed/uncertain pattern: reassess volume status, repeat/confirm haemodynamics if needed, and complete diagnostic work-up.")
    return "\n".join(lines)


def compute_qpqs(hb_g_dl: float, sao2: float, svo2: float, pa_sat: Optional[float]) -> Tuple[float, str]:
    """Qp/Qs via O2 content method; assumes pulmonary venous sat ≈ max(98%, SaO2)."""
    if pa_sat is None:
        return float("nan"), "N/A (PA sat missing)"

    spv = max(98.0, sao2)
    spv = min(spv, 100.0)

    ca = o2_content_ml_per_dl(hb_g_dl, sao2 / 100.0)
    cv = o2_content_ml_per_dl(hb_g_dl, svo2 / 100.0)
    cpa = o2_content_ml_per_dl(hb_g_dl, pa_sat / 100.0)
    cpv = o2_content_ml_per_dl(hb_g_dl, spv / 100.0)

    denom = cpv - cpa
    if abs(denom) < 1e-9:
        return float("nan"), f"N/A (Cpv≈Cpa; SpvO2 assumed {spv:.1f}%)"

    qpqs = (ca - cv) / denom
    return qpqs, f"SpvO2 assumed {spv:.1f}%"


def interpret_shunt(qpqs: float) -> str:
    if is_nan(qpqs):
        return "Shunt: unable to determine (Qp/Qs N/A)."
    if qpqs > 1.05:
        if qpqs < 1.5:
            sev = "Non-significant/small (Qp/Qs < 1.5)."
        elif qpqs <= 2.0:
            sev = "Moderate (Qp/Qs 1.5–2.0)."
        else:
            sev = "Significant/large (Qp/Qs > 2.0)."
        return f"Shunt: Left-to-right suggested. Severity: {sev}"
    if qpqs < 0.95:
        sev = "Moderate (0.80–0.95)." if qpqs >= 0.80 else "Significant (<0.80)."
        return f"Shunt: Right-to-left suggested. Severity: {sev}"
    return "Shunt: no significant shunt suggested (Qp/Qs ~ 1)."


def compute_hemy(inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    # Required basics
    height_cm = float(inputs["height_cm"])
    weight_kg = float(inputs["weight_kg"])
    hb_in = float(inputs["hb_g_L"])
    sao2 = float(inputs["sao2"])
    hr = float(inputs["hr"])
    ra_mean = float(inputs["ra_mean"])
    pcwp = float(inputs["pcwp"])
    pa_sys = float(inputs["pa_sys"])
    pa_dia = float(inputs["pa_dia"])

    # Optional sats
    svc = inputs.get("svc")
    ivc = inputs.get("ivc")
    ra_sat = inputs.get("ra_sat")
    rv_sat = inputs.get("rv_sat")
    pa_sat = inputs.get("pa_sat")

    # Optional systemic pressures
    sbp = inputs.get("sbp")
    dbp = inputs.get("dbp")

    # Optional VO2
    vo2_measured = inputs.get("vo2_measured")

    bsa = bsa_mosteller(height_cm, weight_kg)
    hb_g_L, hb_g_dl, hb_corrected = hb_gL_to_gdL(hb_in)

    mpap = mean_from_sys_dia(pa_sys, pa_dia)
    svo2, svo2_source = pick_mixed_venous_sat(pa_sat, ra_sat, rv_sat, svc, ivc)

    if vo2_measured is None:
        vo2 = 3.5 * weight_kg
        vo2_source = "estimated (3.5 mL/kg/min × weight)"
    else:
        vo2 = float(vo2_measured)
        vo2_source = "measured"

    ca = o2_content_ml_per_dl(hb_g_dl, sao2 / 100.0)
    cv = o2_content_ml_per_dl(hb_g_dl, svo2 / 100.0)
    av_diff = max(ca - cv, 1e-9)
    co = (vo2 / av_diff) / 10.0

    ci = safe_div(co, bsa)
    sv = safe_div(co * 1000.0, hr)
    svi = safe_div(sv, bsa)

    tpg = mpap - pcwp
    dpg = pa_dia - pcwp
    pvr_wu = safe_div((mpap - pcwp), co)
    pvr_dyn = pvr_wu * DYNE_PER_WU
    pvri = pvr_wu * bsa if not is_nan(bsa) else float("nan")

    papi = safe_div((pa_sys - pa_dia), ra_mean)
    rap_pcwp = safe_div(ra_mean, pcwp)
    pac = safe_div(sv, (pa_sys - pa_dia))
    rvswi = svi * (mpap - ra_mean) * RVSWI_FACTOR

    map_mmHg = None
    svr_wu = None
    cpo = None
    if sbp is not None and dbp is not None:
        map_mmHg = map_from_sbp_dbp(float(sbp), float(dbp))
        svr_wu = safe_div((map_mmHg - ra_mean), co)
        cpo = (map_mmHg * co) / 451.0

    qpqs, qpqs_note = compute_qpqs(hb_g_dl, sao2, svo2, pa_sat)
    shunt_text = interpret_shunt(qpqs)

    ph_class = interpret_ph(mpap, pcwp, pvr_wu)
    treat = treatment_block(mpap, pcwp, pvr_wu)

    results = {
        "bsa": bsa,
        "hb_g_L": hb_g_L,
        "hb_g_dl": hb_g_dl,
        "hb_corrected": hb_corrected,
        "mpap": mpap,
        "svo2": svo2,
        "svo2_source": svo2_source,
        "vo2": vo2,
        "vo2_source": vo2_source,
        "co": co,
        "ci": ci,
        "sv": sv,
        "svi": svi,
        "tpg": tpg,
        "dpg": dpg,
        "pvr_wu": pvr_wu,
        "pvr_dyn": pvr_dyn,
        "pvri": pvri,
        "pvr_severity": pvr_severity(pvr_wu),
        "papi": papi,
        "rap_pcwp": rap_pcwp,
        "pac": pac,
        "rvswi": rvswi,
        "map": map_mmHg,
        "svr_wu": svr_wu,
        "cpo": cpo,
        "qpqs": qpqs,
        "qpqs_note": qpqs_note,
        "shunt_text": shunt_text,
        "ph_class": ph_class,
        "treatment": treat,
    }

    report = build_report(inputs, results)
    return results, report


def build_report(inputs: Dict[str, Any], r: Dict[str, Any]) -> str:
    def fmt(x, nd=2):
        if x is None or is_nan(x):
            return "N/A"
        return f"{x:.{nd}f}"

    patient = inputs.get("patient_name", "") or ""
    pid = inputs.get("patient_id", "") or ""
    operator = inputs.get("operator_name", "") or ""
    institution = inputs.get("institution", DEFAULT_INSTITUTION) or DEFAULT_INSTITUTION
    run_ts = inputs.get("run_timestamp", "")

    lines = []
    lines.append("Hemy – RHC Hemodynamics Report")
    lines.append(f"Run timestamp: {run_ts}")
    if patient:
        lines.append(f"Patient: {patient}")
    if pid:
        lines.append(f"Patient ID: {pid}")
    if institution:
        lines.append(f"Institution: {institution}")
    if operator:
        lines.append(f"Operator: {operator}")
    lines.append("")

    if r["hb_corrected"]:
        lines.append("NOTE: Hb input looked like g/dL; auto-converted to g/L.")
        lines.append("")

    lines.append(f"Height: {inputs['height_cm']} cm | Weight: {inputs['weight_kg']} kg | BSA: {fmt(r['bsa'],2)} m²")
    lines.append(f"Hb: {fmt(r['hb_g_L'],0)} g/L (= {fmt(r['hb_g_dl'],1)} g/dL)")
    lines.append(f"SaO2: {inputs['sao2']}% | SvO2 used: {fmt(r['svo2'],1)}% (source: {r['svo2_source']})")
    lines.append(f"VO2: {fmt(r['vo2'],0)} mL/min ({r['vo2_source']})")
    lines.append("")

    lines.append("Flow / pump:")
    lines.append(f"  CO (Fick): {fmt(r['co'],2)} L/min")
    lines.append(f"  CI: {fmt(r['ci'],2)} L/min/m²")
    lines.append(f"  SV: {fmt(r['sv'],0)} mL/beat")
    lines.append(f"  SVI: {fmt(r['svi'],1)} mL/beat/m²")
    if r["cpo"] is not None:
        lines.append(f"  CPO: {fmt(r['cpo'],2)} W")
    lines.append("")

    lines.append("Pressures / pulmonary:")
    lines.append(f"  RAP(mean): {inputs['ra_mean']} mmHg")
    lines.append(f"  PA: {inputs['pa_sys']}/{inputs['pa_dia']} mmHg | mPAP(auto): {fmt(r['mpap'],1)} mmHg")
    lines.append(f"  PCWP(mean): {inputs['pcwp']} mmHg")
    lines.append(f"  TPG: {fmt(r['tpg'],1)} mmHg")
    lines.append(f"  DPG: {fmt(r['dpg'],1)} mmHg")
    lines.append(f"  PVR: {fmt(r['pvr_wu'],2)} WU ({r['pvr_severity']})")
    lines.append(f"       {fmt(r['pvr_dyn'],0)} dyn·s/cm⁵ | PVRI: {fmt(r['pvri'],2)} WU·m²")
    lines.append(f"  PAPi: {fmt(r['papi'],2)}")
    lines.append(f"  RAP/PCWP: {fmt(r['rap_pcwp'],2)}")
    lines.append(f"  PA compliance (SV/PP): {fmt(r['pac'],2)} mL/mmHg")
    lines.append(f"  RVSWI: {fmt(r['rvswi'],1)} g·m/m²/beat")
    lines.append("")

    lines.append("Shunt (Qp/Qs):")
    lines.append(f"  Qp/Qs: {fmt(r['qpqs'],2)} ({r['qpqs_note']})")
    lines.append(f"  {r['shunt_text']}")
    lines.append("")

    lines.append("Final PH hemodynamic phenotype:")
    lines.append(f"  {r['ph_class']}")
    lines.append("")
    lines.append("Treatment summary:")
    lines.append(r["treatment"])
    lines.append("")
    lines.append("NOTE: Treatment section is high-level; definitive decisions depend on PH group (1–5) and full diagnostic work-up.")
    return "\n".join(lines)
