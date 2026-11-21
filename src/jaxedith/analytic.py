"""Analytic aperture rates from performance curves, a scene and an instrument.

:func:`rates` evaluates the scalar terms of the AYO and pyEDITH exposure-time
calculators with the read of the image path: star leakage, local zodi, exozodi
and planet rates are products of a collecting area, a throughput, a source
radiance and the aperture, evaluated once per spectral node. The result is an
``optixstuff.Rates`` that ``optixstuff.snr`` and ``optixstuff.exposure_time``
reduce to a signal-to-noise ratio or a time.

Thermal emission and stellar-companion light are zero in this release, as in
the image path; they return when the instrument describes the warm telescope
and the scene holds a companion source.
"""

import math

import jax.numpy as jnp
from hwoutils.conversions import arcsec_to_rad, lambda_d_to_arcsec
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike
from optixstuff import (
    AbstractDetector,
    Instrument,
    PerformanceCurves,
    Quadrature,
    Rates,
    SampledProfile,
)
from skyscapes import Scene

# A photon-counting EMCCD is run at 1 / 6.73 electrons per pixel per frame.
_OCCUPANCY_DIVISOR = 6.73


def _check_telescope(curves, instrument):
    """Raise when a record was measured for another telescope diameter."""
    frame = curves.frame
    supplied = instrument.primary.diameter(frame.diameter_convention)
    if isinstance(supplied, Tracer):
        return
    if not math.isclose(float(supplied), frame.diameter_m, rel_tol=1e-6):
        raise ValueError(
            f"primary {frame.diameter_convention} diameter {float(supplied)} m "
            f"differs from the curves' lambda/D frame ({frame.diameter_m} m); "
            "request curves for this telescope"
        )


def _moments(record, separation, squared):
    """Return throughput, core area and leakage density, as means or squared weights."""
    suffix = "_variance" if squared else ""
    return (
        record.evaluate(f"throughput{suffix}", separation),
        record.evaluate(f"core_area{suffix}_lod2", separation),
        record.evaluate(f"leakage_density{suffix}_lod2", separation),
    )


def _terms(moments, gain, star, zodi, disk, sky, contrast, solid_angle):
    """Return the five node fields of one band, each of shape (Q,).

    ``moments`` is (throughput, core area, leakage density) in the same
    statistic, mean or squared weight, and ``solid_angle`` the aperture in sr.
    """
    throughput, area, density = moments
    flux = gain * star
    return dict(
        planet=flux * contrast * throughput,
        star_leakage=flux * density * area,
        local_zodi=gain * zodi * sky * solid_angle,
        exozodi=gain * disk * sky * solid_angle,
    )


def rates(
    curves: PerformanceCurves | tuple[PerformanceCurves, ...],
    scene: Scene,
    instrument: Instrument,
    quadrature: Quadrature,
    east_arcsec: ArrayLike,
    north_arcsec: ArrayLike,
    contrast: ArrayLike,
    *,
    time_d: ArrayLike,
    uniform_sky: SampledProfile | None,
) -> Rates:
    """Return one planet's aperture rates in photons/s/nm, fields ``(W, Q)``.

    The planet sits ``east_arcsec`` east and ``north_arcsec`` north of the
    star, the order and meaning of ``coronagraphoto.aperture_rates``, with
    planet-to-star ``contrast``: a scalar, or shape ``(W, Q)``. Per node, with
    collecting area ``A``, system throughput ``T``, star photon flux density
    ``F`` and ``s`` the separation in lambda/D at the band centre:

    - ``planet = A T F contrast throughput(s)``
    - ``star_leakage = A T F leakage_density(s) core_area(s)``
    - ``local_zodi = A T L_zodi sky(s) Omega``
    - ``exozodi = A T L_disk sky(s) Omega``

    with ``Omega = core_area(s) (lambda_c / D)**2`` the aperture solid angle,
    ``L_zodi`` the scene's local-zodi radiance and ``L_disk`` the disk radiance
    at the aperture centre. ``n_pix`` is ``Omega`` over the detector pixel
    solid angle, and ``variance`` holds the same terms from the squared-weight
    moments of the curves. Detector yield is not applied: the record is
    photons, and ``detector.electrons`` applies QE and survival once.

    ``curves`` is one ``PerformanceCurves`` or a tuple with one per band of
    ``quadrature``, measured for this telescope and star (the curves are not
    broadened here). The separation is read at the band-centre lambda/D for
    every node of the band, so the aperture is fixed in angle within a band.
    A separation beyond ``curves.data_extent_lod`` gives NaN rates.
    ``uniform_sky`` is the sky transmission profile, required when the scene
    has a disk or a local zodi. The instrument's coronagraph slot and speckle
    field are not read, and neither is any channel multiplier.

    ``time_d`` is days from the scene epoch. The function is a pure scalar
    read; batch it with ``jax.vmap`` over position and contrast.
    """
    bands = quadrature.bands
    per_band = tuple(curves) if isinstance(curves, tuple) else (curves,) * len(bands)
    if len(per_band) != len(bands):
        raise ValueError("curves must be one record or one per band of the quadrature")
    if uniform_sky is None and (scene.disk is not None or scene.local_zodi is not None):
        raise ValueError("uniform_sky is required when the scene has a disk or zodi")
    for record in {id(record): record for record in per_band}.values():
        _check_telescope(record, instrument)

    time_d = jnp.asarray(time_d)
    east, north = jnp.asarray(east_arcsec), jnp.asarray(north_arcsec)
    nodes = quadrature.nodes_nm

    def per_node(values):
        return jnp.broadcast_to(values, nodes.shape)

    gain = per_node(
        instrument.primary.collecting_area_m2 * instrument.system_throughput(nodes)
    )
    star = per_node(
        scene.star.photon_flux_density(nodes, time_d, dist_pc=scene.dist_pc)
    )
    disk = per_node(scene.photon_radiance_at(nodes, time_d, east, north))
    zodi = per_node(
        0.0
        if scene.local_zodi is None
        else scene.local_zodi.photon_radiance(nodes, scene.sky_frame, time_d)
    )
    contrast = jnp.broadcast_to(jnp.asarray(contrast), nodes.shape)
    pixel_arcsec2 = instrument.detector.pixel_scale_arcsec**2
    angle = jnp.hypot(east, north)

    mean, variance, n_pix, n_pix_variance = [], [], [], []
    for w, (band, record) in enumerate(zip(bands, per_band, strict=True)):
        lod_arcsec = lambda_d_to_arcsec(1.0, band.center_nm, record.frame.diameter_m)
        separation = angle / lod_arcsec
        sky = 1.0 if uniform_sky is None else uniform_sky.evaluate(separation)
        outside = separation > record.data_extent_lod
        for squared, fields_out, pixels_out in (
            (False, mean, n_pix),
            (True, variance, n_pix_variance),
        ):
            throughput, area, density = _moments(record, separation, squared)
            solid_angle = area * arcsec_to_rad(lod_arcsec) ** 2
            fields = _terms(
                (throughput, area, density),
                gain[w],
                star[w],
                zodi[w],
                disk[w],
                sky,
                contrast[w],
                solid_angle,
            )
            fields_out.append(
                {name: jnp.where(outside, jnp.nan, v) for name, v in fields.items()}
            )
            pixels = area * lod_arcsec**2 / pixel_arcsec2
            pixels_out.append(jnp.where(outside, jnp.nan, pixels))

    def stack(fields):
        return {name: jnp.stack([band[name] for band in fields]) for name in fields[0]}

    return Rates(
        quadrature,
        "photon",
        n_pix=jnp.stack(n_pix),
        variance=Rates(
            quadrature, "photon", n_pix=jnp.stack(n_pix_variance), **stack(variance)
        ),
        **stack(mean),
    )


def photon_counting_frame_time(rates: Rates, detector: AbstractDetector) -> Array:
    """Return the photon-counting frame time per band, shape ``(W,)`` seconds.

    The frame time at which a photon-counting EMCCD sees a mean of 1/6.73
    electrons per aperture pixel per frame, ``1 / (6.73 r)`` with ``r`` the
    electron rate of all sources, planet included, per aperture pixel (the
    cadence pyEDITH takes for clock-induced charge). ``rates`` is one aperture
    in photons/s/nm or electrons/s with ``n_pix`` set.
    """
    if rates.n_pix is None:
        raise ValueError("rates must carry n_pix, the aperture's pixel count")
    electrons = detector.electrons(rates) if rates.carrier == "photon" else rates
    per_pixel = electrons.total() / electrons.n_pix
    return 1.0 / (_OCCUPANCY_DIVISOR * per_pixel)
