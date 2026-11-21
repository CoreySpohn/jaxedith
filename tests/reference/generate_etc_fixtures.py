"""Regenerate the CSV fixtures of the exposure-time validation suite.

Three files are written beside this script:

``pyedith_dump.csv``
    The inputs and outputs of the toy-coronagraph dump shipped with pyEDITH
    (``pyedith_validation.txt``), transcribed at the seven significant figures
    the dump prints.
``pyedith_cases.csv``
    pyEDITH run live on the five calibration stars at 500 and 1000 nm with
    constant coronagraph curves taken from the calibration workbook, plus two
    convention cases (nonzero read noise, a finite exozodi subtraction factor).
    Every number is kept at full float64 precision.
``ayo_workbook.csv``
    The AYO and EXOSIMS columns of the calibration workbook, one cell per row,
    with the number of significant digits the cell prints.

pyEDITH and its workbook are read from an installed pyEDITH checkout. pyEDITH
pins ``hwoutils<2`` and ``yippy<3``, so it cannot join the library's test
environment; the fixtures are generated once and committed. Usage::

    python generate_etc_fixtures.py [PYEDITH_CHECKOUT]
"""

import csv
import io
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from importlib import metadata
from pathlib import Path

import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parents[1] / "data"
STARS = ("HIP 32439", "HIP 77052", "HIP 79672", "HIP 26779", "HIP 113283")
# Workbook columns (zero-based) of each code at each wavelength.
COLUMNS = {
    ("AYO", 500): 3,
    ("EXOSIMS", 500): 5,
    ("AYO", 1000): 9,
    ("EXOSIMS", 1000): 11,
}
LAMBDA_LABEL, BANDWIDTH_LABEL = "λ", "Δλ"
PIXEL_SCALE_LOD = 0.25  # the 0.25 lambda/D pixel of the workbook's printed intensities
NOISE_FLOOR_FACTOR = 0.029  # the workbook's noise floor over leaked starlight
SQUARE_METERS_PER_SQUARE_CENTIMETER = 1e-4


def checkout_path(argument):
    """Return the pyEDITH checkout, from the argument or the imported package."""
    if argument:
        return Path(argument).resolve()
    import pyEDITH

    return Path(pyEDITH.__file__).resolve().parents[2]


def git_head(path):
    """Return the checkout's commit, or an empty string outside a git tree."""
    done = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip()


def significant_digits(value):
    """Return how many significant digits a cell's shortest repr prints."""
    text = format(Decimal(repr(float(value))), "e")
    mantissa = text.split("e")[0].replace("-", "").replace(".", "")
    return max(1, len(mantissa.rstrip("0")))


def write_csv(path, header_lines, fieldnames, rows):
    """Write rows under ``#`` header lines; the reader skips them."""
    with open(path, "w", newline="") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def workbook_cells(path):
    """Yield (star, wavelength, code, label, value, is_formula) for numeric cells."""
    cached = openpyxl.load_workbook(path, data_only=True)
    formulas = openpyxl.load_workbook(path, data_only=False)
    for star in STARS:
        sheet = f"{star} (Detection)"
        for (code, wavelength), column in COLUMNS.items():
            for row_c, row_f in zip(
                cached[sheet].iter_rows(), formulas[sheet].iter_rows(), strict=True
            ):
                label = row_c[0].value
                value = row_c[column].value
                if label is None or not isinstance(value, (int, float)):
                    continue
                if label == "Solar longitude":
                    continue
                label = {
                    LAMBDA_LABEL: "lambda",
                    BANDWIDTH_LABEL: "bandwidth",
                    "Check on t_science (AYO equation)": "check_t_science_ayo",
                    "Check on t_science (EXOSIMS equation)": "check_t_science_exosims",
                }.get(label, str(label))
                label = label.replace("Check on ", "check_")
                is_formula = str(row_f[column].value).startswith("=")
                yield star, wavelength, code, label, float(value), is_formula


def write_workbook_csv(checkout, path_xlsx):
    rows = []
    for star, wavelength, code, label, value, is_formula in workbook_cells(path_xlsx):
        rows.append(
            dict(
                star=star,
                wavelength_nm=wavelength,
                code=code,
                label=label,
                value=repr(value),
                digits=0 if is_formula else significant_digits(value),
            )
        )
    write_csv(
        HERE / "ayo_workbook.csv",
        [
            "Cells of the AYO and EXOSIMS columns of the ETC calibration workbook,",
            "ETC_cal_detect.xlsx as shipped with pyEDITH, sheets '<star> (Detection)'.",
            f"pyEDITH commit {git_head(checkout)}.",
            "digits: significant digits the cell prints; 0 marks a formula cell stored",
            "at full precision. check_* rows are the workbook's own formula cells that",
            "recompute a rate or the science time from the printed cells.",
        ],
        ["star", "wavelength_nm", "code", "label", "value", "digits"],
        rows,
    )
    return len(rows)


def write_dump_csv(checkout):
    """Transcribe the printed validation dump into one name,value row per quantity."""
    text = (checkout / "pyedith_validation.txt").read_text()
    rows = []
    for name, value in re.findall(r"^([^\n:]+): (?:max )?value: ([^\s,]+)", text, re.M):
        rows.append(dict(name=name.strip(), value=value))
    write_csv(
        HERE / "pyedith_dump.csv",
        [
            "Quantities printed by pyEDITH's validation dump (pyedith_validation.txt),",
            "toy coronagraph, 500 nm, S/N 7, at the seven significant figures it "
            "prints.",
            f"pyEDITH commit {git_head(checkout)}.",
            "Names are the dump's own, qualified by section where the dump "
            "repeats them.",
        ],
        ["name", "value"],
        rows,
    )
    return len(rows)


def star_inputs(checkout, star, cells):
    """Return the HPIC row of a star, for pyEDITH's own zodi and exozodi models."""
    path = checkout / "tests/validation_tests_battery/full_HPIC.txt"
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            if row["hip_name"] == star.split()[1]:
                return row
    raise KeyError(star)


def pyedith_parameters(cells, hpic, case):
    """Build pyEDITH's toy-model input dictionary from workbook cells.

    The toy coronagraph is constant everywhere; its parameters are solved so
    that the leakage density, core throughput, core area and sky transmission
    equal the workbook's printed values.
    """
    from hwoutils.conversions import lambda_d_to_arcsec

    wavelength_nm = cells["lambda"]
    diameter = cells["D"]
    lod_arcsec = float(lambda_d_to_arcsec(1.0, wavelength_nm, diameter))
    pixel = PIXEL_SCALE_LOD**2
    lyot = cells["skytrans"] * pixel
    peak = 0.025 * lyot  # the toy coronagraph's fixed PSF peak
    params = {
        "wavelength": np.array([wavelength_nm / 1000.0]),
        "snr": np.array([cells["SNR"]]),
        "CRb_multiplier": 2.0,
        "photometric_aperture_radius": float(np.sqrt(cells["omega_core"] / np.pi)),
        "distance": float(hpic["sy_dist"]),
        "stellar_radius": float(hpic["st_rad"]),
        "Lstar": 10 ** float(hpic["st_lum"]),
        "magV": float(hpic["sy_vmag"]),
        "ra": float(hpic["ra"]),
        "dec": float(hpic["dec"]),
        "nzodis": cells["nzodis"],
        "separation": cells["sp"] * lod_arcsec,
        "F0": np.array([cells["F_0"]]),
        "mag": np.array([-2.5 * np.log10(cells["F_star"] / cells["F_0"])]),
        "delta_mag": np.array([-2.5 * np.log10(cells["F_p"] / cells["F_star"])]),
        "observatory_preset": "ToyModel",
        "observing_mode": "IMAGER",
        "filter_name": "calibration",
        "filter_center": wavelength_nm / 1000.0,
        "filter_bandwidth": cells["bandwidth"] / wavelength_nm,
        "diameter": diameter,
        "unobscured_area": cells["A"] * 1e-4 / (np.pi * (diameter / 2) ** 2),
        "T_optical": np.array([cells["T_optical"]]),
        "T_contamination": 1.0,
        "QE": np.array([cells["QE"]]),
        "dQE": np.array([cells["dQE"]]),
        "DC": np.array([cells["det_DC"]]),
        "RN": np.array([cells["det_RN"]]),
        "CIC": np.array([cells["det_CIC"]]),
        "tread": np.array([cells["det_tread"]]),
        "pixscale": PIXEL_SCALE_LOD,
        "TLyot": lyot,
        "Tcore": cells["T_core"],
        "contrast": cells["I_star"] * pixel / peak,
        "noisefloor_factor": NOISE_FLOOR_FACTOR,
        "coronagraph_bandwidth": 0.2,
        # Fixed as in the workbook (pyEDITH's validation mode): the constant toy
        # curve puts pyEDITH's own estimate at zero separation, where the exozodi
        # diverges.
        "det_npix_input": cells["det_npix"],
        "t_photon_count_input": cells["t_photon_count"],
    }
    params.update(case.get("override", {}))
    return params


def run_pyedith(params):
    """Run pyEDITH once and return the quantities the comparison needs."""
    from pyEDITH import (
        AstrophysicalScene,
        Observation,
        Observatory,
        parse_input,
        set_verbosity,
    )
    from pyEDITH.exposure_time_calculator import calculate_exposure_time_or_snr
    from pyEDITH.units import DIMENSIONLESS, FRAME, SECOND

    set_verbosity(level="error")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        parameters = parse_input.parse_parameters(params)
        (band,) = parse_input.parse_filters(parameters)
        observation = Observation()
        observation.load_configuration(parameters, filter=band)
        observation.set_output_arrays()
        observation.validate_configuration()
        scene = AstrophysicalScene()
        scene.load_configuration(parameters)
        scene.calculate_zodi_exozodi(parameters)
        scene.validate_configuration()
        scene.regrid_spectra(observation)
        observatory = Observatory()
        observatory.create_observatory(parse_input.get_observatory_config(parameters))
        observatory.load_configuration(parameters, observation, scene)
        observatory.validate_configuration()
        # Only the EAC detector reads the validation-mode inputs; set them as it does.
        observatory.detector.det_npix_input = (
            parameters["det_npix_input"] * DIMENSIONLESS
        )
        observatory.detector.t_photon_count_input = (
            parameters["t_photon_count_input"] * SECOND / FRAME
        )
        calculate_exposure_time_or_snr(
            observation, scene, observatory, ETC_validation=True, mode="exposure_time"
        )
    vv = observation.validation_variables

    def number(name):
        return float(np.asarray(getattr(vv[name], "value", vv[name])).ravel()[0])

    return dict(
        total_throughput=float(
            np.asarray(observatory.total_throughput.value).ravel()[0]
        ),
        area_cm2=number("A_cm"),
        bandwidth_nm=number("deltalambda_nm"),
        flux_star=number("Fs_over_F0"),
        flux_planet=number("Fp"),
        zodi_sb=number("Fzodi"),
        exozodi_sb=number("Fexozodi"),
        lod_arcsec=number("lod_arcsec"),
        n_pix=number("det_npix"),
        t_photon_count=number("t_photon_count"),
        ez_ppf=float(np.asarray(scene.ez_PPF.value).ravel()[0]),
        CR_p=number("CRp"),
        CR_bs=number("CRbs"),
        CR_bz=number("CRbz"),
        CR_bez=number("CRbez"),
        CR_bth=number("CRbth"),
        CR_bd=number("CRbd"),
        CR_NF=number("CRnf"),
        t_science=number("sciencetime"),
    )


def write_cases_csv(checkout, path_xlsx):
    cells_by = {}
    for star, wavelength, code, label, value, _ in workbook_cells(path_xlsx):
        if code == "AYO":
            cells_by.setdefault((star, wavelength), {})[label] = value
    cases = [
        dict(name=f"{star} {wl} nm", star=star, wavelength=wl, override={})
        for star in STARS
        for wl in (500, 1000)
    ]
    cases += [
        dict(
            name="read noise",
            star="HIP 32439",
            wavelength=500,
            override={"RN": np.array([0.8])},
        ),
        dict(
            name="exozodi floor",
            star="HIP 32439",
            wavelength=500,
            override={"ez_PPF": 200.0},
        ),
    ]
    rows = []
    for case in cases:
        cells = cells_by[(case["star"], case["wavelength"])]
        params = pyedith_parameters(
            cells, star_inputs(checkout, case["star"], cells), case
        )
        out = run_pyedith(params)
        coronagraph = dict(
            core_throughput=cells["T_core"],
            core_area_lod2=cells["omega_core"],
            leakage_density_lod2=cells["I_star"],
            sky_transmission=cells["skytrans"] * PIXEL_SCALE_LOD**2,
            speckle_residual_fraction=NOISE_FLOOR_FACTOR,
        )
        rows.append(
            {
                "case": case["name"],
                "wavelength_nm": cells["lambda"],
                "diameter_m": cells["D"],
                "snr": cells["SNR"],
                "separation_lod": cells["sp"],
                "dark_e_per_pix_s": cells["det_DC"],
                "cic_e_per_pix_frame": cells["det_CIC"],
                "read_noise_e": float(params["RN"][0]),
                "read_time_s": cells["det_tread"],
                "qe": cells["QE"],
                "survival": cells["dQE"],
                **coronagraph,
                **out,
            }
        )
    rows = [
        {k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()}
        for r in rows
    ]
    write_csv(
        HERE / "pyedith_cases.csv",
        [
            "pyEDITH run on constant coronagraph curves solved from the AYO column "
            "of the",
            "ETC calibration workbook (500 and 1000 nm), with pyEDITH's own detector",
            "pixel count and photon-counting time, zodi and exozodi surface "
            "brightness.",
            f"pyEDITH {metadata.version('pyEDITH')} commit {git_head(checkout)}; "
            f"astropy {metadata.version('astropy')}.",
            "Fluxes are per cm^2, per nm, per second (per arcsec^2 for the sky terms);",
            "rates are electrons/s; area_cm2 and total_throughput are pyEDITH's own.",
            "ez_ppf inf means exozodi subtracted to the Poisson limit.",
        ],
        list(rows[0]),
        rows,
    )
    return len(rows)


def main(argument=None):
    checkout = checkout_path(argument)
    path_xlsx = checkout / "tests/validation_tests_battery/ETC_cal_detect.xlsx"
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)  # pyEDITH may write its working files into the cwd
        print("workbook cells", write_workbook_csv(checkout, path_xlsx))
        print("dump rows", write_dump_csv(checkout))
        print("pyEDITH cases", write_cases_csv(checkout, path_xlsx))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
