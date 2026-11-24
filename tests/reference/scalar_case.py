"""One scalar exposure-time case as plain numbers, and the objects that realize it.

The community calculators (AYO, pyEDITH, EXOSIMS) take a handful of numbers:
fluxes per cm^2, a coronagraph that is constant over separation, a detector.
:class:`ScalarCase` holds those numbers in the calculators' own units and
builds the scene, instrument, curves and sky profile that make
``jaxedith.rates`` read exactly them, so a comparison tests the arithmetic and
the unit conversions, not a coronagraph model.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
from hwoutils.conversions import arcsec_to_rad, lambda_d_to_arcsec
from hwoutils.epoch import Epoch
from optixstuff import (
    ConstantThroughput,
    Detector,
    Instrument,
    PerformanceCurves,
    PostProcessing,
    Quadrature,
    SampledProfile,
    SimplePrimary,
)
from optixstuff.apertures import Circular
from optixstuff.grids import Frame
from optixstuff.spectral import Band, Resolution
from skyscapes import PrecomputedZodi
from skyscapes.disk import AbstractDisk
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import FlatStar, Planet, Scene

import jaxedith

DATA = Path(__file__).resolve().parents[1] / "data"
CM2_PER_M2 = 1e4
# The planet is read in photons; the disk and zodi radiances are per steradian.
SR_PER_ARCSEC2 = float(arcsec_to_rad(1.0)) ** 2


def read_csv(name):
    """Read a fixture under ``tests/data``, skipping its ``#`` header lines."""
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


class ConstantDisk(AbstractDisk):
    """A disk of the same photon radiance everywhere, ph/s/m^2/nm/sr."""

    radiance: float
    flux_scale: float = 1.0

    def _photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
    ):
        """Return the constant radiance, wavelength axes first."""
        wave = jnp.asarray(wavelength_nm)
        sky = jnp.broadcast_shapes(jnp.shape(east_arcsec), jnp.shape(north_arcsec))
        return jnp.full(wave.shape + sky, self.radiance)


@dataclass(frozen=True)
class ScalarCase:
    """The inputs of a scalar calculator, in its units.

    ``star_flux`` is per cm^2 per nm per second at the telescope, ``zodi_sb``
    and ``exozodi_sb`` the surface brightness at the aperture in the same units
    per arcsec^2, ``contrast`` the planet over the star, ``core_throughput``,
    ``core_area_lod2`` (lambda/D squared), ``leakage_density_lod2``
    (lambda/D to the minus two) and ``sky_transmission`` the constant
    coronagraph, and ``n_pix`` the detector pixels in the aperture.
    """

    wavelength_nm: float
    bandwidth_nm: float
    diameter_m: float
    area_cm2: float
    total_throughput: (
        float  # optics times QE and the other yields, as the codes print it
    )
    qe: float  # the detector's share of it
    star_flux: float
    contrast: float
    core_throughput: float
    core_area_lod2: float
    leakage_density_lod2: float
    sky_transmission: float
    zodi_sb: float
    exozodi_sb: float
    dark: float
    cic: float
    read_noise: float
    n_pix: float
    separation_arcsec: float

    @property
    def lod_arcsec(self):
        """Lambda over D in arcsec at the band centre."""
        return float(lambda_d_to_arcsec(1.0, self.wavelength_nm, self.diameter_m))

    @property
    def optics(self):
        """Optical throughput that, with the detector's QE, gives the total."""
        return self.total_throughput / self.qe

    def band(self):
        """The one band, centred on the wavelength."""
        half = self.bandwidth_nm / 2
        return Band(self.wavelength_nm - half, self.wavelength_nm + half)

    def quadrature(self):
        """One-node quadrature over the band."""
        return Quadrature.build((self.band(),), order=1)

    def frame(self):
        """The frame whose lambda/D the curves and the sky profile use."""
        return Frame(self.wavelength_nm, self.diameter_m, "circumscribed")

    def curves(self):
        """Constant coronagraph curves; variance moments equal the means."""
        flat = lambda value: np.full(2, value)  # noqa: E731
        area = self.core_area_lod2
        density = self.leakage_density_lod2
        return PerformanceCurves(
            np.array([0.0, 40.0]),
            self.frame(),
            Circular(0.7),
            flat(self.core_throughput),
            flat(area),
            flat(density * area),
            flat(density),
            flat(density * area / self.core_throughput),
            1.0,
            30.0,
            throughput_variance=flat(self.core_throughput),
            core_area_variance_lod2=flat(area),
            leakage_fraction_variance=flat(density * area),
            leakage_density_variance_lod2=flat(density),
        )

    def sky(self):
        """The constant sky transmission."""
        return SampledProfile(
            np.array([0.0, 40.0]),
            np.full(2, self.sky_transmission),
            quantity="uniform_sky_response",
            frame=self.frame(),
            interpolation="linear",
        )

    def scene(self):
        """A flat star, a uniform local zodi and a constant disk."""
        grid = GridPhysicalModel(
            jnp.array([100.0, 3000.0]),
            jnp.array([0.0, 180.0]),
            jnp.full((1, 2, 2), 1e-10),
        )
        per_m2_sr = CM2_PER_M2 / SR_PER_ARCSEC2
        return Scene.from_classical(
            FlatStar(flux_phot_per_nm_m2=self.star_flux * CM2_PER_M2),
            (Planet(jnp.ones(1), grid),),
            star_mass_Msun=1.0,
            planet_mass_Msun=3e-6,
            dist_pc=10.0,
            epoch=Epoch(2451545.0),
            ra_deg=100.0,
            dec_deg=-20.0,
            a_AU=jnp.array([1.0]),
            e=0.0,
            W_rad=0.0,
            i_rad=0.0,
            w_rad=0.0,
            M0_rad=0.0,
            disk=ConstantDisk(self.exozodi_sb * per_m2_sr),
            local_zodi=PrecomputedZodi(
                np.array([100.0, 3000.0]), np.full(2, self.zodi_sb * per_m2_sr)
            ),
        )

    def detector(self, *, read_noise=None):
        """The detector whose pixel count in the aperture is ``n_pix``."""
        pixel = self.lod_arcsec * math.sqrt(self.core_area_lod2 / self.n_pix)
        return Detector(
            pixel_scale_arcsec=pixel,
            shape=(8, 8),
            qe=self.qe,
            dark_current_e_per_pixel_s=self.dark,
            clock_induced_charge_e_per_pixel_frame=self.cic,
            read_noise_e_rms=self.read_noise if read_noise is None else read_noise,
        )

    def instrument(self, **keywords):
        """Primary, optics throughput and the detector."""
        primary = SimplePrimary(
            circumscribed_diameter_m=self.diameter_m,
            collecting_area_m2=self.area_cm2 / CM2_PER_M2,
        )
        return Instrument(
            primary,
            (ConstantThroughput(self.optics),),
            None,
            self.detector(**keywords),
        )

    def rates(self):
        """The aperture rates in photons/s/nm at the case's separation."""
        scene, quadrature = self.scene(), self.quadrature()
        resolution = Resolution.build(quadrature, *scene.spectral_tables())
        return jaxedith.rates(
            self.curves(),
            scene,
            self.instrument(),
            quadrature,
            self.separation_arcsec,
            0.0,
            contrast=self.contrast,
            time_d=0.0,
            sources=scene.sources(quadrature, resolution, 0.0),
            uniform_sky=self.sky(),
        )

    def electrons(self, *, companions=0.0):
        """The aperture's electron rates, with an optional stellar-companion rate.

        ``companions`` (electrons/s) is added to the mean and the variance
        record, as a third light source the analytic reader does not produce.
        """
        record = self.instrument().detector.electrons(self.rates())
        if companions:
            record = eqx.tree_at(
                lambda r: (r.stellar_companions, r.variance.stellar_companions),
                record,
                (jnp.full((1,), companions), jnp.full((1,), companions)),
            )
        return record

    def terms(self, *, companions=0.0):
        """Planet, leakage, zodi and exozodi electron rates as floats."""
        record = self.electrons(companions=companions)
        return {
            name: float(getattr(record, name)[0])
            for name in ("planet", "star_leakage", "local_zodi", "exozodi")
        }

    def detector_rate(self, frame_time_s, **keywords):
        """The detector's variance per live second, electrons/s, with no sources.

        Dark current, clock-induced charge and read noise over the aperture's
        pixels for frames of ``frame_time_s``, as the exposure-time calculator
        counts them: the quantity the scalar codes print as the detector term.
        """
        detector = self.instrument(**keywords).detector
        quiet = ox.Rates(
            (self.band(),),
            "electron",
            planet=jnp.zeros(1),
            n_pix=jnp.array([self.n_pix]),
        )
        _, variance = detector.mean_and_variance(quiet, ox.FramePlan(frame_time_s, 1))
        return float(variance[0]) / frame_time_s

    def exposure_time(
        self,
        snr,
        frame_time_s,
        *,
        speckle=0.0,
        exozodi=0.0,
        factor=2.0,
        companions=0.0,
        **keywords,
    ):
        """Continuous live time at ``snr``, with the post-processing arguments."""
        detector = self.instrument(**keywords).detector
        result = ox.exposure_time(
            self.electrons(companions=companions),
            detector,
            snr,
            frame_time_s=frame_time_s,
            post_processing=PostProcessing(
                speckle, exozodi, background_variance_factor=factor
            ),
        )
        return result


def from_workbook_cells(cells, *, separation_lod=None):
    """Build a case from one workbook column's cells, keyed by label.

    Fluxes are the printed ones; the sky transmission is the printed value
    per (lambda/D)^2 of a 0.25 lambda/D pixel times that pixel's area, which
    is the transmission the printed zodi and exozodi rates imply.
    """
    qe = cells["QE"] * cells["dQE"]
    wavelength = cells["lambda"]
    diameter = cells["D"]
    lod = float(lambda_d_to_arcsec(1.0, wavelength, diameter))
    separation = cells["sp"] if separation_lod is None else separation_lod
    return ScalarCase(
        wavelength_nm=wavelength,
        bandwidth_nm=cells["bandwidth"],
        diameter_m=diameter,
        area_cm2=cells["A"],
        total_throughput=cells["T_optical"] * qe,
        qe=qe,
        star_flux=cells["F_star"],
        contrast=cells["F_p"] / cells["F_star"],
        core_throughput=cells["T_core"],
        core_area_lod2=cells["omega_core"],
        leakage_density_lod2=cells["I_star"],
        sky_transmission=cells["skytrans"] * 0.25**2,
        zodi_sb=cells["F_zodi"],
        exozodi_sb=cells["F_exozodi"],
        dark=cells["det_DC"],
        cic=cells["det_CIC"],
        read_noise=cells["det_RN"],
        n_pix=cells["det_npix"],
        separation_arcsec=separation * lod,
    )


def workbook_columns():
    """Return {(star, wavelength_nm, code): {label: (value, digits)}} from the fixture.

    ``digits`` is the significant digits the cell prints, 0 for a formula cell
    stored at full precision.
    """
    columns = {}
    for row in read_csv("ayo_workbook.csv"):
        key = (row["star"], int(row["wavelength_nm"]), row["code"])
        columns.setdefault(key, {})[row["label"]] = (
            float(row["value"]),
            int(row["digits"]),
        )
    return columns


def half_ulp(value, digits):
    """Relative half unit in the last printed digit of a cell; 1e-15 if a formula."""
    if digits == 0 or value == 0:
        return 1e-15
    exponent = math.floor(math.log10(abs(value)))
    return 0.5 * 10.0 ** (exponent - digits + 1) / abs(value)
