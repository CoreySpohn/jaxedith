"""jaxedith against pyEDITH, on constant coronagraph curves.

Cross-code benchmark of the AYO and pyEDITH arithmetic: the same scalar
inputs go into pyEDITH (generated once, stored in ``tests/data``) and into
``jaxedith.rates`` with ``optixstuff.exposure_time``, and every count rate and
the science time are compared. No coronagraph model is involved: the curves
are constants, so the tests exercise the product of area, throughput, flux,
bandwidth and aperture, the unit conversions, the detector term and the time
equation.

The matched budget of the benchmark allows 1e-6 on count rates (the scale
chain of area, throughput and bandwidth agrees to that) and 1e-5 on the
science time; the measured agreement is at float64 rounding, orders of
magnitude below both.
"""

import math

import ineedvalidation as vv
import numpy as np
import pytest
from reference.etc import exptime_from_rates_ayo
from reference.legacy import legacy_pyedith_noise_rate
from reference.scalar_case import ScalarCase, read_csv

pytestmark = pytest.mark.float64_only(reason="pyEDITH validation reference values")

RTOL_RATE = 1e-6
RTOL_TIME = 1e-5
CASES = read_csv("pyedith_cases.csv")
DUMP = {row["name"]: float(row["value"]) for row in read_csv("pyedith_dump.csv")}
OCCUPANCY = 6.73


def _case(row):
    """A ScalarCase from one generated row, and the row's floats."""
    f = {k: float(v) for k, v in row.items() if k != "case"}
    case = ScalarCase(
        wavelength_nm=f["wavelength_nm"],
        bandwidth_nm=f["bandwidth_nm"],
        diameter_m=f["diameter_m"],
        area_cm2=f["area_cm2"],
        total_throughput=f["total_throughput"],
        qe=f["qe"] * f["survival"],
        star_flux=f["flux_star"],
        contrast=f["flux_planet"] / f["flux_star"],
        core_throughput=f["core_throughput"],
        core_area_lod2=f["core_area_lod2"],
        leakage_density_lod2=f["leakage_density_lod2"],
        sky_transmission=f["sky_transmission"],
        zodi_sb=f["zodi_sb"],
        exozodi_sb=f["exozodi_sb"],
        dark=f["dark_e_per_pix_s"],
        cic=f["cic_e_per_pix_frame"],
        read_noise=f["read_noise_e"],
        n_pix=f["n_pix"],
        separation_arcsec=f["separation_lod"] * f["lod_arcsec"],
    )
    return case, f


def _post_processing(f):
    """Post-processing arguments pyEDITH's floor corresponds to."""
    return dict(
        speckle=f["speckle_residual_fraction"],
        exozodi=0.0 if math.isinf(f["ez_ppf"]) else 1.0 / f["ez_ppf"],
    )


def _ids(rows):
    return [row["case"] for row in rows]


STAR_ROWS = [row for row in CASES if row["case"].startswith("HIP")]
CONVENTION_ROWS = {
    row["case"]: row for row in CASES if not row["case"].startswith("HIP")
}


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
def test_toy_dump_reproduces_every_printed_rate_and_the_science_time():
    """Fails if a count rate or the science time departs from pyEDITH's printed dump.

    Basis: the dump prints seven significant figures, so printing alone bounds
    the residual near 5e-7; rtol 1e-5 on the science time (the measured residual
    from the rounding is 2e-7) and 1e-5 on the rates. The inputs are the dump's
    own: flux zero point, stellar and planet flux, zodi and exozodi surface
    brightness, throughput, constant coronagraph, detector pixel count of 100
    that the printed detector rate and the 1 s photon-counting time fix.
    """
    d = DUMP
    diameter = d["observatory.telescope.diameter"]
    unobscured = 1.0 - 0.121  # the toy telescope's default, not printed
    pixscale = d["observatory.coronagraph.pixscale"]  # lambda/D per pixel
    separation_au = d["scene.separation"] * d["scene.dist"]
    f0 = d["scene.F0"]
    omega = d["observatory.coronagraph.omega_lod"]
    qe = d["observatory.detector.QE*observatory.detector.dQE"]
    t_photon = d["t_photon_count"]
    n_pix = d["CRbd"] / (
        d["observatory.detector.DC"] + d["observatory.detector.CIC"] / t_photon
    )
    assert n_pix == pytest.approx(100.0, rel=1e-3)  # fixed by the printed detector rate
    n_pix = float(round(n_pix))
    case = ScalarCase(
        wavelength_nm=d["observation.wavelength"] * 1000.0,
        bandwidth_nm=d["deltalambda_nm"],
        diameter_m=diameter,
        area_cm2=math.pi * (diameter * 100.0 / 2) ** 2 * unobscured,
        total_throughput=d["observatory.total_throughput"],
        qe=qe,
        star_flux=f0 * d["Fs_over_F0"],
        contrast=d["scene.Fp_over_Fs"],
        core_throughput=d["observatory.coronagraph.photometric_aperture_throughput"],
        core_area_lod2=omega,
        leakage_density_lod2=d["observatory.coronagraph.Istar"] / pixscale**2,
        sky_transmission=d["observatory.coronagraph.skytrans"],
        zodi_sb=f0 * d["scene.Fzodi_list"],
        exozodi_sb=f0 * d["scene.Fexozodi_list"] / separation_au**2,
        dark=d["observatory.detector.DC"],
        cic=d["observatory.detector.CIC"],
        read_noise=d["observatory.detector.RN"],
        n_pix=n_pix,
        separation_arcsec=d["scene.separation"],
    )
    terms = case.terms()
    np.testing.assert_allclose(terms["planet"], d["CRp"], rtol=RTOL_RATE * 10)
    np.testing.assert_allclose(terms["star_leakage"], d["CRbs"], rtol=RTOL_RATE * 10)
    np.testing.assert_allclose(terms["local_zodi"], d["CRbz"], rtol=RTOL_RATE * 10)
    np.testing.assert_allclose(terms["exozodi"], d["CRbez"], rtol=RTOL_RATE * 10)
    np.testing.assert_allclose(case.detector_rate(t_photon), d["CRbd"], rtol=1e-12)
    # The floor, as the dump prints it, is the noise floor over the leakage.
    fraction = (
        d["observatory.coronagraph.noisefloor"] / d["observatory.coronagraph.Istar"]
    )
    np.testing.assert_allclose(
        fraction * terms["star_leakage"], d["CRnf"], rtol=RTOL_RATE * 10
    )
    science_time = (
        d["observation.exptime"] / d["observatory.coronagraph.nrolls"]
        - d["observatory.telescope.toverhead_fixed"]
    ) / d["observatory.telescope.toverhead_multi"]
    result = case.exposure_time(
        d["observation.SNR"],
        t_photon,
        speckle=fraction,
        factor=d["observation.CRb_multiplier"],
    )
    np.testing.assert_allclose(result.time_s, science_time, rtol=RTOL_TIME)
    assert bool(result.valid.all())


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
@pytest.mark.parametrize("row", STAR_ROWS, ids=_ids(STAR_ROWS))
def test_calibration_stars_agree_on_every_rate_and_the_science_time(row):
    """Fails if a count rate, the detector term or the time departs from pyEDITH.

    Basis: both sides are float64 products of the same constants, so the
    residual is rounding; rtol 1e-6 on rates and 1e-5 on the time, the matched
    budget. A larger gap would be a named convention difference, not noise.
    """
    case, f = _case(row)
    terms = case.terms()
    np.testing.assert_allclose(terms["planet"], f["CR_p"], rtol=RTOL_RATE)
    np.testing.assert_allclose(terms["star_leakage"], f["CR_bs"], rtol=RTOL_RATE)
    np.testing.assert_allclose(terms["local_zodi"], f["CR_bz"], rtol=RTOL_RATE)
    np.testing.assert_allclose(terms["exozodi"], f["CR_bez"], rtol=RTOL_RATE)
    np.testing.assert_allclose(
        case.detector_rate(f["t_photon_count"]), f["CR_bd"], rtol=RTOL_RATE
    )
    post = _post_processing(f)
    floor = math.hypot(
        post["speckle"] * terms["star_leakage"], post["exozodi"] * terms["exozodi"]
    )
    np.testing.assert_allclose(floor, f["CR_NF"], rtol=RTOL_RATE)
    result = case.exposure_time(f["snr"], f["t_photon_count"], **post)
    np.testing.assert_allclose(result.time_s, f["t_science"], rtol=RTOL_TIME)
    assert bool(result.valid.all())


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
def test_exozodi_residual_enters_the_floor_as_pyedith_adds_it():
    """Fails if a finite exozodi subtraction factor is not added in quadrature.

    Basis: one case with ez_ppf = 200, where the exozodi term dominates the
    floor; rtol 1e-6 on the floor and 1e-5 on the time, as for the stars.
    """
    case, f = _case(CONVENTION_ROWS["exozodi floor"])
    assert f["ez_ppf"] == 200.0
    terms = case.terms()
    post = _post_processing(f)
    assert (
        post["exozodi"] * terms["exozodi"]
        > 100 * post["speckle"] * terms["star_leakage"]
    )
    result = case.exposure_time(f["snr"], f["t_photon_count"], **post)
    np.testing.assert_allclose(result.time_s, f["t_science"], rtol=RTOL_TIME)


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
def test_read_noise_differs_by_the_two_frame_times_not_by_arithmetic():
    """Fails if the read-noise gap to pyEDITH is anything but the predicted one.

    Basis: exact algebra, rtol 1e-12. pyEDITH divides the read variance by its
    read time and the clock-induced charge by the photon-counting time;
    optixstuff has one frame time for both. The predicted gap is
    ``n_pix RN**2 (1 / t_frame - 1 / t_read)`` on the detector rate, carried to
    the time through the AYO equation; pyEDITH's own rate is reproduced by the
    legacy reference to 1e-12 first, so the gap is a convention, not an error.
    """
    case, f = _case(CONVENTION_ROWS["read noise"])
    assert f["read_noise_e"] > 0
    t_frame, t_read = f["t_photon_count"], f["read_time_s"]
    legacy = legacy_pyedith_noise_rate(
        1.0 / (OCCUPANCY * t_frame),
        n_pix=f["n_pix"],
        dark_current_e_per_pixel_s=f["dark_e_per_pix_s"],
        read_noise_e_rms=f["read_noise_e"],
        read_frame_time_s=t_read,
        clock_induced_charge_e_per_pixel_frame=f["cic_e_per_pix_frame"],
    )
    np.testing.assert_allclose(legacy, f["CR_bd"], rtol=1e-12)
    ours = case.detector_rate(t_frame)
    gap = f["n_pix"] * f["read_noise_e"] ** 2 * (1.0 / t_frame - 1.0 / t_read)
    np.testing.assert_allclose(ours - legacy, gap, rtol=1e-12)
    np.testing.assert_allclose(
        ours / legacy,
        1.0 + gap / f["CR_bd"],
        rtol=1e-12,
    )
    # The time: the AYO equation on pyEDITH's rates with our detector term.
    terms = case.terms()
    background = terms["star_leakage"] + terms["local_zodi"] + terms["exozodi"] + ours
    expected = exptime_from_rates_ayo(
        terms["planet"], background, f["CR_NF"], f["snr"], 2.0
    )
    result = case.exposure_time(f["snr"], t_frame, **_post_processing(f))
    np.testing.assert_allclose(result.time_s, expected, rtol=1e-12)
    assert float(result.time_s[0]) > f["t_science"]
