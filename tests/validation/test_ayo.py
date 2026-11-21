"""jaxedith against the AYO columns of the calibration workbook.

The workbook prints, for five stars at 500 and 1000 nm, every input and every
count rate of the AYO exposure-time calculation. Each column is rebuilt as a
:class:`ScalarCase` from its printed inputs, and the rates and the science time
are compared with the printed cells.

The workbook prints seven significant figures of its inputs, and each rate is
a product of about eight of them, so rounding of the printed inputs alone
bounds the agreement near 5e-6: the rate and time budgets used here are 1e-5.
"""

import ineedvalidation as vv
import pytest
from reference.scalar_case import from_workbook_cells, half_ulp, workbook_columns

RTOL_RATE = 1e-5
RTOL_TIME = 1e-5
SPECKLE_FRACTION = 0.029  # the workbook's CR_NF is 0.029 times the leakage rate
COLUMNS = {key: cells for key, cells in workbook_columns().items() if key[2] == "AYO"}
IDS = [f"{star}-{wl}nm" for star, wl, _ in COLUMNS]

# Columns whose printed detector rate disagrees with the printed detector inputs
# (CR_bd over n_pix * (DC + CIC / t_photon)): the cell is stale against its own
# inputs. Pinned as printed over formula so a change in either shows.
STALE_DETECTOR = {
    ("HIP 77052", 500, "AYO"): 0.949111,
    ("HIP 77052", 1000, "AYO"): 0.779763,
    ("HIP 32439", 1000, "AYO"): 1.000052,
}


def _values(cells):
    return {label: value for label, (value, _) in cells.items()}


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="ayo")
@pytest.mark.parametrize("key", list(COLUMNS), ids=IDS)
def test_rates_agree_with_the_printed_cells(key):
    """Fails if a planet, leakage, zodi or exozodi rate departs from the workbook cell.

    Basis: rtol 1e-5, from the rounding of the printed inputs (seven figures,
    eight factors); measured agreement is below 1e-6 on every cell. The
    workbook's own full-precision formula cells for the planet and leakage
    rates are matched at the same budget.
    """
    cells = COLUMNS[key]
    v = _values(cells)
    terms = from_workbook_cells(v).terms()
    assert terms["planet"] == pytest.approx(v["CR_p"], rel=RTOL_RATE)
    assert terms["star_leakage"] == pytest.approx(v["CR_bs"], rel=RTOL_RATE)
    assert terms["local_zodi"] == pytest.approx(v["CR_bz"], rel=RTOL_RATE)
    assert terms["exozodi"] == pytest.approx(v["CR_bez"], rel=RTOL_RATE)
    assert terms["planet"] == pytest.approx(v["check_CR_p"], rel=RTOL_RATE)
    assert terms["star_leakage"] == pytest.approx(v["check_CR_bs"], rel=RTOL_RATE)
    # The noise floor is the speckle fraction times the leakage rate.
    assert v["CR_NF"] == pytest.approx(
        SPECKLE_FRACTION * v["CR_bs"], rel=2 * half_ulp(*cells["CR_NF"]) + 1e-6
    )


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="ayo")
@pytest.mark.parametrize("key", list(COLUMNS), ids=IDS)
def test_detector_rate_follows_the_printed_detector_inputs(key):
    """Fails if jaxedith's detector rate is not n_pix * (DC + CIC / t_photon).

    Basis: the jaxedith side is exact arithmetic of the printed inputs (rtol
    1e-12). The printed cell agrees to 1e-5 except in three columns whose cell
    is stale against its own inputs; those are pinned as findings with the
    printed-over-formula ratio at rtol 1e-6.
    """
    v = _values(COLUMNS[key])
    case = from_workbook_cells(v)
    formula = v["det_npix"] * (v["det_DC"] + v["det_CIC"] / v["t_photon_count"])
    assert v["det_RN"] == 0.0
    assert case.detector_rate(v["t_photon_count"]) == pytest.approx(formula, rel=1e-12)
    ratio = v["CR_bd"] / formula
    if key in STALE_DETECTOR:
        assert ratio == pytest.approx(STALE_DETECTOR[key], rel=1e-6)
    else:
        assert ratio == pytest.approx(1.0, rel=1e-5)


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="ayo")
@pytest.mark.parametrize("key", list(COLUMNS), ids=IDS)
def test_science_time_agrees_with_the_workbook_formula(key):
    """Fails if the continuous exposure time departs from the workbook's time cell.

    Basis: rtol 1e-5 (the matched time budget; measured at most 7.4e-6, from
    the printed inputs). The frame time is the one the printed detector rate
    implies, so a stale detector cell does not enter; the stray-light rate the
    workbook adds is injected as a stellar-companion rate; speckle fraction
    0.029, factor 2, compared with the workbook's full-precision time cell and
    with its printed time (which differs from that cell by up to 7e-6).
    """
    v = _values(COLUMNS[key])
    case = from_workbook_cells(v)
    frame = v["det_CIC"] / (v["CR_bd"] / v["det_npix"] - v["det_DC"])
    result = case.exposure_time(
        v["SNR"], frame, speckle=SPECKLE_FRACTION, companions=v.get("CR_bstray", 0.0)
    )
    assert float(result.time_s[0]) == pytest.approx(
        v["check_t_science_ayo"], rel=RTOL_TIME
    )
    assert v["t_science"] == pytest.approx(v["check_t_science_ayo"], rel=RTOL_TIME)
