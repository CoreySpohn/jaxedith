"""Analytic aperture rates from performance curves, a scene and an instrument.

:func:`rates` evaluates the scalar terms of the AYO and pyEDITH exposure-time
calculators with the read of the image path: star leakage, local zodi, exozodi
and planet rates are products of a collecting area, a throughput, a source
radiance and the aperture, evaluated once per spectral node. The result is an
``optixstuff.Rates`` that ``optixstuff.snr`` and ``optixstuff.exposure_time``
reduce to a signal-to-noise ratio or a time.

Thermal emission is read from ``Instrument.warm_emission`` and is zero when
the instrument has none. Stellar-companion light is zero in this release, as in
the image path; it returns when the scene holds a companion source.
"""

import jax
import jax.numpy as jnp
import optixstuff as ox
from hwoutils.conversions import arcsec_to_rad
from jax import Array
from jax.typing import ArrayLike
from optixstuff import (
    AbstractDetector,
    ExposureTime,
    Instrument,
    PerformanceCurves,
    PostProcessing,
    Quadrature,
    Rates,
    SampledProfile,
)
from skyscapes import Scene

# A photon-counting EMCCD is run at 1 / 6.73 electrons per pixel per frame.
_OCCUPANCY_DIVISOR = 6.73
_READ_FIELDS = ("planet", "star_leakage", "local_zodi", "exozodi", "thermal")


def _moments(record, separation, squared):
    """Return throughput, core area and leakage density, as means or squared weights."""
    suffix = "_variance" if squared else ""
    return (
        record.evaluate(f"throughput{suffix}", separation),
        record.evaluate(f"core_area{suffix}_lod2", separation),
        record.evaluate(f"leakage_density{suffix}_lod2", separation),
    )


def _terms(moments, gain, star, zodi, disk, sky, contrast, thermal, solid_angle):
    """Return the node fields of one band, each of shape (Q,).

    ``moments`` is (throughput, core area, leakage density) in the same
    statistic, mean or squared weight, ``thermal`` the warm telescope's photon
    rate per steradian and ``solid_angle`` the aperture in sr.
    """
    throughput, area, density = moments
    flux = gain * star
    return dict(
        planet=flux * contrast * throughput,
        star_leakage=flux * density * area,
        local_zodi=gain * zodi * sky * solid_angle,
        exozodi=gain * disk * sky * solid_angle,
        thermal=thermal * solid_angle,
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
    - ``thermal = A_geo e B Omega``

    with ``Omega = core_area(s) (lambda_c / D)**2`` the aperture solid angle,
    ``L_zodi`` the scene's local-zodi radiance and ``L_disk`` the disk radiance
    at the aperture centre. ``thermal`` is the warm telescope's own emission,
    pyEDITH's ``calculate_CRbth``: ``B`` the Planck photon radiance at the
    temperature of ``instrument.warm_emission``, ``e`` its effective emissivity
    (warm emissivity times cold transmission) and ``A_geo`` the geometric
    collecting area, with no system throughput and no flux scale (see
    ``Instrument.thermal_photon_rate_per_sr``). It is zero for an instrument
    without ``warm_emission``, the default. ``optixstuff`` owns the
    description; ``jaxedith`` and the image path of ``coronagraphoto`` both
    fill the term.
    ``n_pix`` is ``Omega`` over the detector pixel solid angle, and
    ``variance`` holds the same terms from the squared-weight moments of the
    curves. Detector yield is not applied: the record is
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
    read; batch it with ``jax.vmap`` over position and contrast, or read a
    whole scene with :func:`scene_rates`.
    """
    bands = quadrature.bands
    per_band = tuple(curves) if isinstance(curves, tuple) else (curves,) * len(bands)
    if len(per_band) != len(bands):
        raise ValueError("curves must be one record or one per band of the quadrature")
    if uniform_sky is None and (scene.disk is not None or scene.local_zodi is not None):
        raise ValueError("uniform_sky is required when the scene has a disk or zodi")
    for record in {id(record): record for record in per_band}.values():
        ox.check_primary_matches_frame(
            instrument.primary, record.frame, source="curves"
        )

    time_d = jnp.asarray(time_d)
    east, north = jnp.asarray(east_arcsec), jnp.asarray(north_arcsec)
    nodes = quadrature.nodes_nm

    def per_node(values):
        return jnp.broadcast_to(values, nodes.shape)

    gain = per_node(instrument.effective_area_m2(nodes))
    star = per_node(
        scene.star.photon_flux_density(nodes, time_d, dist_pc=scene.dist_pc)
    )
    disk = per_node(scene.photon_radiance_at(nodes, time_d, east, north))
    zodi = per_node(
        0.0
        if scene.local_zodi is None
        else scene.local_zodi.photon_radiance(nodes, scene.sky_frame, time_d)
    )
    thermal = per_node(instrument.thermal_photon_rate_per_sr(nodes))
    contrast = jnp.broadcast_to(jnp.asarray(contrast), nodes.shape)
    pixel_arcsec2 = instrument.detector.pixel_scale_arcsec**2
    angle = jnp.hypot(east, north)

    mean, variance, n_pix, n_pix_variance = [], [], [], []
    for w, (band, record) in enumerate(zip(bands, per_band, strict=True)):
        lod_arcsec = record.frame.lod_to_arcsec(band.center_nm)
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
                thermal[w],
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


def scene_rates(
    curves: PerformanceCurves | tuple[PerformanceCurves, ...],
    scene: Scene,
    instrument: Instrument,
    quadrature: Quadrature,
    time_d: ArrayLike,
    *,
    uniform_sky: SampledProfile | None,
) -> Rates:
    """Return every planet of a scene at every time, in photons/s/nm.

    A ``jax.vmap`` of :func:`rates` over the scene's planets and the times:
    positions come from ``scene.positions(time_d)`` and each planet's contrast
    from ``scene.contrasts`` at every spectral node of ``quadrature``. It
    follows ``coronagraphoto.scene_aperture_rates``, so the two reads swap:
    ``time_d`` has shape ``(T,)``, days from the scene epoch, and the returned
    ``Rates`` has the band and node axes leading and the batch axes trailing.
    Fields have shape ``(W, Q, K, T)``, ``n_pix`` has shape ``(W, K, T)``, and
    ``variance`` holds the squared-weight rates. Pass it to
    ``optixstuff.snr`` or ``optixstuff.exposure_time``, which return one value
    per band, planet and time. A planet with ``scene.valid`` False reads zero
    in every field, so its exposure time is invalid, not a number. A scene with
    a batch of systems is not read; map this function over them.

    ``curves`` and ``uniform_sky`` are as in :func:`rates`. Every planet is
    read with the same aperture rule as :func:`rates`, so a planet beyond
    ``curves.data_extent_lod`` reads NaN.
    """
    time_d = jnp.asarray(time_d)
    if time_d.ndim != 1:
        raise ValueError("time_d must have shape (T,)")
    if scene.batch_shape:
        raise ValueError("scene_rates reads one system; vmap it over a batch")
    east, north = scene.positions(time_d)
    contrast = jax.vmap(
        jax.vmap(lambda wavelength: scene.contrasts(wavelength, time_d))
    )(quadrature.nodes_nm)

    def read(east_arcsec, north_arcsec, contrast, time):
        record = rates(
            curves,
            scene,
            instrument,
            quadrature,
            east_arcsec,
            north_arcsec,
            contrast,
            time_d=time,
            uniform_sky=uniform_sky,
        )
        return tuple(
            (*(getattr(r, name) for name in _READ_FIELDS), r.n_pix)
            for r in (record, record.variance)
        )

    over_times = jax.vmap(read, in_axes=(0, 0, -1, 0), out_axes=-1)
    mean, variance = jax.vmap(over_times, in_axes=(0, 0, -2, None), out_axes=-2)(
        east, north, contrast, time_d
    )

    def build(values, variance=None):
        *fields, n_pix = (jnp.where(scene.valid[:, None], v, 0.0) for v in values)
        return Rates(
            quadrature,
            "photon",
            **dict(zip(_READ_FIELDS, fields, strict=True)),
            n_pix=n_pix,
            variance=variance,
        )

    return build(mean, build(variance))


def planet_exposure_time(
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
    target_snr: ArrayLike,
    frame_time_s: ArrayLike,
    post_processing: PostProcessing,
) -> ExposureTime:
    """Return the live time reaching ``target_snr`` for one planet.

    The one-call entry point for the scalar calculators, for the single planet
    at ``east_arcsec``, ``north_arcsec`` with ``contrast``; the scene's other
    planets are not read (:func:`scene_rates` reads all of them). It is
    ``optixstuff.exposure_time(rates(...), instrument.detector, target_snr,
    frame_time_s=frame_time_s, post_processing=post_processing)`` and does no
    arithmetic of its own; see those functions for every argument and for the
    meaning of ``valid``.
    """
    record = rates(
        curves,
        scene,
        instrument,
        quadrature,
        east_arcsec,
        north_arcsec,
        contrast,
        time_d=time_d,
        uniform_sky=uniform_sky,
    )
    return ox.exposure_time(
        record,
        instrument.detector,
        target_snr,
        frame_time_s=frame_time_s,
        post_processing=post_processing,
    )


def planet_snr(
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
    time_s: ArrayLike,
    frame_time_s: ArrayLike,
    post_processing: PostProcessing,
) -> Array:
    """Return one planet's S/N per band after ``time_s`` live seconds.

    The planet is the one at ``east_arcsec``, ``north_arcsec`` with
    ``contrast``; the scene's other planets are not read. It is
    ``optixstuff.snr(rates(...), instrument.detector, time_s,
    frame_time_s=frame_time_s, post_processing=post_processing)`` and does no
    arithmetic of its own.
    """
    record = rates(
        curves,
        scene,
        instrument,
        quadrature,
        east_arcsec,
        north_arcsec,
        contrast,
        time_d=time_d,
        uniform_sky=uniform_sky,
    )
    return ox.snr(
        record,
        instrument.detector,
        time_s,
        frame_time_s=frame_time_s,
        post_processing=post_processing,
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
