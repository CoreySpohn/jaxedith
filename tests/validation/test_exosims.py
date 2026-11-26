"""jaxedith against the EXOSIMS columns of the calibration workbook.

The EXOSIMS detection equation puts the planet rate in the numerator only
through the speckle term and not in the variance, which the shared calculator
does not express; it stays a reference equation (``tests/reference/etc.py``)
fed with jaxedith's rates. The characterization equation is the calculator's
equation with a background factor of 1, and is compared directly.

Budgets: 2e-5 on the rates (the workbook prints six to seven figures of its
inputs and the EXOSIMS columns round the leakage density), 1e-5 on times.
"""

import ineedvalidation as vv
import pytest
from hwoutils.testing import ulps
from reference.etc import (
    exptime_from_rates_ayo,
    exptime_from_rates_exosims_char,
    exptime_from_rates_exosims_det,
)
from reference.scalar_case import from_workbook_cells, workbook_columns

RTOL_RATE = 2e-5
RTOL_TIME = 1e-5
SPECKLE_FRACTION = 0.029
COLUMNS = {
    key: cells for key, cells in workbook_columns().items() if key[2] == "EXOSIMS"
}
IDS = [f"{star}-{wl}nm" for star, wl, _ in COLUMNS]

# Printed exozodi rate over the rate from the printed exozodi surface
# brightness: the same for every star at a wavelength (HIP 32439 at 500 nm is
# the one column that agrees). Pinned as a finding: the printed rate is not
# reproducible from the printed F_exozodi.
EXOZODI_RATIO = {500: 0.995444, 1000: 1.191110}
# Printed science time over the workbook's own formula cell, at 500 nm, where
# the printed time was left stale.
STALE_TIME = {
    "HIP 32439": 1.003324,
    "HIP 79672": 1.000601,
    "HIP 26779": 1.004427,
    "HIP 113283": 0.997950,
}


def _values(cells):
    return {label: value for label, (value, _) in cells.items()}


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="exosims")
@pytest.mark.float64_only(reason="EXOSIMS workbook validation reference values")
@pytest.mark.parametrize("key", list(COLUMNS), ids=IDS)
def test_rates_agree_with_the_printed_cells(key):
    """Fails if a planet, leakage or zodi rate departs from the EXOSIMS column.

    Basis: rtol 2e-5 from the printed-input rounding; measured at most 8e-6.
    The exozodi rate is the finding below, not a budget item.
    """
    v = _values(COLUMNS[key])
    terms = from_workbook_cells(v).terms()
    assert terms["planet"] == pytest.approx(v["CR_p"], rel=RTOL_RATE)
    assert terms["star_leakage"] == pytest.approx(v["CR_bs"], rel=RTOL_RATE)
    assert terms["local_zodi"] == pytest.approx(v["CR_bz"], rel=RTOL_RATE)


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="exosims")
@pytest.mark.float64_only(reason="EXOSIMS workbook validation reference values")
@pytest.mark.parametrize("key", list(COLUMNS), ids=IDS)
def test_printed_exozodi_rate_is_not_reproducible_from_its_surface_brightness(key):
    """Pins the EXOSIMS exozodi cell against jaxedith: a finding, not a budget item.

    Basis: the ratio printed/jaxedith is constant over stars at each
    wavelength (0.995444 at 500 nm, 1.19111 at 1000 nm; HIP 32439 at 500 nm is
    1 to 1e-6), pinned at rtol 2e-5 so a change in either side shows.
    """
    star, wavelength, _ = key
    v = _values(COLUMNS[key])
    ratio = v["CR_bez"] / from_workbook_cells(v).terms()["exozodi"]
    expected = (
        1.0 if (star, wavelength) == ("HIP 32439", 500) else EXOZODI_RATIO[wavelength]
    )
    assert ratio == pytest.approx(expected, rel=2e-5)


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="exosims")
@pytest.mark.float64_only(reason="EXOSIMS workbook validation reference values")
@pytest.mark.parametrize("key", list(COLUMNS), ids=IDS)
def test_detection_time_from_jaxedith_rates_matches_the_workbook_formula(key):
    """Fails if the EXOSIMS detection time from jaxedith's rates departs from the cell.

    Basis: rtol 1e-5 on the workbook's full-precision time cell. The printed
    exozodi and detector cells are used for those two terms, since the first
    is the finding above and the second is the workbook's own; the leakage,
    zodi and planet rates are jaxedith's. The printed time differs from its
    formula cell at 500 nm for four stars (pinned, rtol 1e-5).
    """
    star, wavelength, _ = key
    v = _values(COLUMNS[key])
    terms = from_workbook_cells(v).terms()
    background = terms["star_leakage"] + terms["local_zodi"] + v["CR_bez"] + v["CR_bd"]
    background += v.get("CR_bstray", 0.0)
    time = float(
        exptime_from_rates_exosims_det(
            terms["planet"],
            background,
            SPECKLE_FRACTION * terms["star_leakage"],
            v["SNR"],
        )
    )
    assert time == pytest.approx(v["check_t_science_exosims"], rel=RTOL_TIME)
    stale = STALE_TIME.get(star) if wavelength == 500 else None
    ratio = v["t_science"] / v["check_t_science_exosims"]
    assert ratio == pytest.approx(stale or 1.0, rel=RTOL_TIME)


@vv.case("etc-inversion", "code-verification", basis="exact-algebra")
@pytest.mark.parametrize("key", list(COLUMNS)[:4], ids=IDS[:4])
def test_calculator_factor_selects_the_ayo_or_exosims_characterization_equation(key):
    """Fails if the calculator's background factor does not reduce to the two equations.

    Basis: 16 rounding operations cover the two variance and time calculations.
    Six more cover the squared rates, amplified by the condition of subtracting
    the speckle-floor square from the planet-rate square.
    Factor 2 is the AYO equation and factor 1 the EXOSIMS characterization one.
    """
    v = _values(COLUMNS[key])
    case = from_workbook_cells(v)
    frame = v["t_photon_count"]
    terms = case.terms()
    background = (
        terms["star_leakage"]
        + terms["local_zodi"]
        + terms["exozodi"]
        + case.detector_rate(frame)
    )
    floor = SPECKLE_FRACTION * terms["star_leakage"]
    planet_sq = float(terms["planet"]) ** 2
    floor_sq = (v["SNR"] * float(floor)) ** 2
    condition = (planet_sq + floor_sq) / (planet_sq - floor_sq)
    rtol = float(ulps(16, 1.0) + condition * ulps(6, 1.0))
    for factor, reference in (
        (
            2.0,
            exptime_from_rates_ayo(terms["planet"], background, floor, v["SNR"], 2.0),
        ),
        (
            1.0,
            exptime_from_rates_exosims_char(
                terms["planet"], background, floor, v["SNR"]
            ),
        ),
    ):
        result = case.exposure_time(
            v["SNR"], frame, speckle=SPECKLE_FRACTION, factor=factor
        )
        assert float(result.time_s[0]) == pytest.approx(
            float(reference), rel=rtol, abs=0
        )
