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

import equinox as eqx
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
    Resolution,
    SampledProfile,
)
from skyscapes import Scene
from skyscapes.scene import Sources

# A photon-counting EMCCD is run at 1 / 6.73 electrons per pixel per frame.
_OCCUPANCY_DIVISOR = 6.73
_READ_FIELDS = ("planet", "star_leakage", "local_zodi", "exozodi", "thermal")


def _moments(record, separation, diameter, squared):
    """Return throughput, core area and leakage density, as means or squared weights.

    ``diameter`` is the star's angular diameter in lambda/D at each node, which
    the leakage density is read at.
    """
    suffix = "_variance" if squared else ""
    return (
        record.evaluate(f"throughput{suffix}", separation),
        record.evaluate(f"core_area{suffix}_lod2", separation),
        record.evaluate(f"leakage_density{suffix}_lod2", separation, diameter),
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


def _at_time(sources: Sources, index) -> Sources:
    """Return ``sources`` holding only the time at ``index``, which may be traced."""
    held = jnp.take(sources.unscaled_star, index, axis=-1)[..., None]
    sources = eqx.tree_at(lambda s: s.unscaled_star, sources, held)
    if sources.local_zodi is not None:
        zodi = jnp.take(sources.local_zodi, index, axis=-1)[..., None]
        sources = eqx.tree_at(lambda s: s.local_zodi, sources, zodi)
    return sources


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
    sources: Sources,
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
    ``quadrature``, measured for this telescope. The leakage is read at the
    star's angular diameter (``sources.star_diameter_arcsec``) in lambda/D at
    each node, so a record tabulated over stellar diameter is read at the
    star's size. A record of one profile that carries its diameter
    (``stellar_diameters_lod`` set) raises for a star of a different diameter.
    A record with ``stellar_diameters_lod=None`` has no provenance to check and
    is read for any star unchecked; the caller vouches for it. The separation
    and the aperture solid angle are read at the band centre, the diameter at
    each node.

    The function reads no source spectrum. ``F``, ``L_zodi`` and the star the
    disk is defined against are the node-effective rows of ``sources``, from
    ``scene.sources(quadrature, resolution, time_d)`` at the one time read
    here, so a tabulated star or local zodi is integrated exactly over each
    band however few nodes it has. ``contrast`` is the star-weighted node
    contrast, ``sources.contrast`` of the planet and time, for a scene read;
    a scalar is the natural input of an exposure-time calculation and is exact
    for a contrast that does not vary across the band. Build the quadrature
    with ``Quadrature.build(bands, *models, field_lod=0)``: the curves are read
    at the band-centre lambda/D for every node, so the response sets no
    extra nodes. Under a different ``field_lod`` the result is the same to
    quadrature accuracy and costs more nodes.

    The star leakage is read from curves that are functions of separation
    alone, so the position angle of the planet is never used. What
    ``leakage_density`` means at a separation depends on how the record was
    made. For a rotationally symmetric (1D) product it is the aperture
    leakage, the same at every angle. For a 2D product prepared by yippy with
    its default rule (``aperture_ring_mean``) it is the exact-coverage aperture
    averaged over position angle at that separation, the roll-averaged
    leakage and not the value at the planet's own angle; with ``aperture_mean``
    it is the aperture on the +x axis only. Where the leakage at a specific
    planet position angle and roll is wanted, read it from the image or
    aperture-record route (``coronagraphoto.aperture_record``), which agrees
    with a roll average of this read and gives the single-angle value.

    The separation is read at the band-centre lambda/D for
    every node of the band, so the aperture is fixed in angle within a band.
    A planet where ``curves.supported(separation)`` is False (beyond the first
    or last sample, or its aperture leaving ``data_extent_lod``) reads zero in
    every field with ``Rates.supported`` False, so ``optixstuff.snr`` and
    ``exposure_time`` return NaN there. The curves' support is a disk of radius
    ``data_extent_lod`` and does not know the detector.
    ``uniform_sky`` is the sky transmission profile, required when the scene
    has a disk or a local zodi. The instrument's coronagraph slot and speckle
    field are not read, and neither is any channel multiplier.

    ``time_d`` is days from the scene epoch. The function is a pure scalar
    read; batch it with ``jax.vmap`` over position and contrast, or read a
    whole scene with :func:`scene_rates`.

    Args:
        curves: Performance curves measured for this telescope, one
            record or a tuple with one per band of ``quadrature``.
        scene: Scene holding the star, the planets and any disk or local zodi.
        instrument: Instrument supplying the collecting area, detector pixel
            scale and any warm emission.
        quadrature: Spectral bands and their wavelength nodes in nm.
        east_arcsec: Planet offset east of the star in arcsec.
        north_arcsec: Planet offset north of the star in arcsec.
        contrast: Planet-to-star flux ratio, a scalar or shape ``(W, Q)``.
        time_d: Time in days from the scene epoch.
        sources: Node-effective sources of ``scene`` on ``quadrature`` at
            ``time_d`` alone (shape ``T = 1``), from ``scene.sources``.
        uniform_sky: Sky transmission profile read at the separation in
            lambda/D, or None. Required when the scene has a disk or a local
            zodi.

    Returns:
        Aperture rates in photons/s/nm with fields of shape ``(W, Q)``,
        ``n_pix`` of shape ``(W,)``, the squared-weight rates in ``variance``
        and ``supported`` False for a planet the curves do not support (past their
        samples or extent).
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
    held = sources.unscaled_star.shape
    if held != (*nodes.shape, 1):
        raise ValueError(
            f"sources hold star rows of shape {held}; rates reads one time on "
            f"the nodes {nodes.shape}, so pass scene.sources(quadrature, "
            "resolution, time_d) for this quadrature and a single time"
        )

    def per_node(values):
        return jnp.broadcast_to(values, nodes.shape)

    gain = per_node(instrument.effective_area_m2(nodes))
    star = per_node(sources.star[..., 0])
    disk = per_node(
        scene.photon_radiance_at(nodes, time_d, east, north, star=sources.node_star())
    )
    zodi = per_node(0.0 if sources.local_zodi is None else sources.local_zodi[..., 0])
    thermal = per_node(instrument.thermal_photon_rate_per_sr(nodes))
    contrast = jnp.broadcast_to(jnp.asarray(contrast), nodes.shape)
    pixel_arcsec2 = instrument.detector.pixel_scale_arcsec**2
    angle = jnp.hypot(east, north)

    mean, variance, n_pix, n_pix_variance, supported = [], [], [], [], []
    for w, (band, record) in enumerate(zip(bands, per_band, strict=True)):
        lod_arcsec = record.frame.lod_to_arcsec(band.center_nm)
        separation = angle / lod_arcsec
        # The star spans a different number of lambda/D at each node.
        diameter = sources.star_diameter_arcsec / record.frame.lod_to_arcsec(nodes[w])
        outside = ~record.supported(separation)
        # Read inside the table so values and gradients stay finite; the
        # unsupported reads are zeroed below.
        separation = jnp.clip(
            separation, record.separation_lod[0], record.separation_lod[-1]
        )
        sky = 1.0 if uniform_sky is None else uniform_sky.evaluate(separation)
        for squared, fields_out, pixels_out in (
            (False, mean, n_pix),
            (True, variance, n_pix_variance),
        ):
            throughput, area, density = _moments(record, separation, diameter, squared)
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
                {name: jnp.where(outside, 0.0, v) for name, v in fields.items()}
            )
            pixels = area * lod_arcsec**2 / pixel_arcsec2
            pixels_out.append(jnp.where(outside, 0.0, pixels))
        supported.append(~outside)

    def stack(fields):
        return {name: jnp.stack([band[name] for band in fields]) for name in fields[0]}

    flag = jnp.stack(supported)
    return Rates(
        quadrature,
        "photon",
        n_pix=jnp.stack(n_pix),
        variance=Rates(
            quadrature,
            "photon",
            n_pix=jnp.stack(n_pix_variance),
            supported=flag,
            **stack(variance),
        ),
        supported=flag,
        **stack(mean),
    )


def scene_rates(
    curves: PerformanceCurves | tuple[PerformanceCurves, ...],
    scene: Scene,
    instrument: Instrument,
    quadrature: Quadrature,
    time_d: ArrayLike,
    *,
    resolution: Resolution,
    uniform_sky: SampledProfile | None,
) -> Rates:
    """Return every planet of a scene at every time, in photons/s/nm.

    A ``jax.vmap`` of :func:`rates` over the scene's planets and the times:
    positions come from ``scene.positions(time_d)``. The sources are read once,
    outside the map, with ``scene.sources(quadrature, resolution, time_d)``,
    and each planet's contrast is ``sources.contrast`` at every spectral node
    of ``quadrature``, so a tabulated star or planet is integrated exactly over
    each band and nothing is recomputed per planet or time. Build the
    resolution on the host with
    ``Resolution.build(quadrature, *scene.spectral_tables())``, and the
    quadrature with ``field_lod=0`` (see :func:`rates`). It follows
    ``coronagraphoto.scene_aperture_rates``, so the two reads swap:
    ``time_d`` has shape ``(T,)``, days from the scene epoch, and the returned
    ``Rates`` has the band and node axes leading and the batch axes trailing.
    Fields have shape ``(W, Q, K, T)``, ``n_pix`` has shape ``(W, K, T)``, and
    ``variance`` holds the squared-weight rates. Pass it to
    ``optixstuff.snr`` or ``optixstuff.exposure_time``, which return one value
    per band, planet and time. A planet with ``scene.valid`` False reads zero
    in every field, so its exposure time is invalid, not a number. A scene with
    a batch of systems is not read; map this function over them.

    ``curves`` and ``uniform_sky`` are as in :func:`rates`. Every planet is
    read with the same aperture rule as :func:`rates`, so a planet the curves
    do not support reads zero with ``supported`` False, shape
    ``(W, K, T)``.

    Args:
        curves: Performance curves measured for this telescope, one
            record or a tuple with one per band of ``quadrature``.
        scene: Scene holding the star, the planets and any disk or local zodi.
        instrument: Instrument supplying the collecting area, detector pixel
            scale and any warm emission.
        quadrature: Spectral bands and their wavelength nodes in nm.
        time_d: Times in days from the scene epoch, shape ``(T,)``.
        resolution: Operators for the scene's tables on ``quadrature``.
        uniform_sky: Sky transmission profile read at the separation in
            lambda/D, or None. Required when the scene has a disk or a local
            zodi.

    Returns:
        Aperture rates in photons/s/nm with fields of shape ``(W, Q, K, T)``,
        ``n_pix`` and ``supported`` of shape ``(W, K, T)``, and the
        squared-weight rates in ``variance``.
    """
    time_d = jnp.asarray(time_d)
    if time_d.ndim != 1:
        raise ValueError("time_d must have shape (T,)")
    if scene.batch_shape:
        raise ValueError("scene_rates reads one system; vmap it over a batch")
    ephemeris = scene.ephemeris(time_d)
    east, north = scene.positions(time_d, ephemeris=ephemeris)
    sources = scene.sources(quadrature, resolution, time_d, ephemeris=ephemeris)
    contrast = sources.contrast

    def read(east_arcsec, north_arcsec, contrast, time, index):
        record = rates(
            curves,
            scene,
            instrument,
            quadrature,
            east_arcsec,
            north_arcsec,
            contrast,
            time_d=time,
            sources=_at_time(sources, index),
            uniform_sky=uniform_sky,
        )
        return tuple(
            (*(getattr(r, name) for name in _READ_FIELDS), r.n_pix, r.supported)
            for r in (record, record.variance)
        )

    over_times = jax.vmap(read, in_axes=(0, 0, -1, 0, 0), out_axes=-1)
    mean, variance = jax.vmap(over_times, in_axes=(0, 0, -2, None, None), out_axes=-2)(
        east, north, contrast, time_d, jnp.arange(time_d.size)
    )

    def build(values, variance=None):
        *fields, n_pix, supported = values
        fields = [jnp.where(scene.valid[:, None], v, 0.0) for v in fields]
        n_pix = jnp.where(scene.valid[:, None], n_pix, 0.0)
        return Rates(
            quadrature,
            "photon",
            **dict(zip(_READ_FIELDS, fields, strict=True)),
            n_pix=n_pix,
            variance=variance,
            supported=supported,
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
    sources: Sources,
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

    Args:
        curves: Performance curves measured for this telescope, one
            record or a tuple with one per band of ``quadrature``.
        scene: Scene holding the star, the planets and any disk or local zodi.
        instrument: Instrument supplying the collecting area, detector pixel
            scale and any warm emission.
        quadrature: Spectral bands and their wavelength nodes in nm.
        east_arcsec: Planet offset east of the star in arcsec.
        north_arcsec: Planet offset north of the star in arcsec.
        contrast: Planet-to-star flux ratio, a scalar or shape ``(W, Q)``.
        time_d: Time in days from the scene epoch.
        sources: Node-effective sources of ``scene`` on ``quadrature`` at
            ``time_d`` alone, from ``scene.sources``; see :func:`rates`.
        uniform_sky: Sky transmission profile read at the separation in
            lambda/D, or None. Required when the scene has a disk or a local
            zodi.
        target_snr: Target S/N, a scalar or shape ``(W,)``.
        frame_time_s: Seconds per detector frame, a scalar or shape ``(W,)``.
        post_processing: Residual fractions and background variance factor of
            the speckle and exozodi subtraction.

    Returns:
        Continuous live seconds per band, shape ``(W,)``, with NaN and
        ``valid`` False where the target cannot be reached or the read is not
        ``Rates.supported``.
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
        sources=sources,
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
    sources: Sources,
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

    Args:
        curves: Performance curves measured for this telescope, one
            record or a tuple with one per band of ``quadrature``.
        scene: Scene holding the star, the planets and any disk or local zodi.
        instrument: Instrument supplying the collecting area, detector pixel
            scale and any warm emission.
        quadrature: Spectral bands and their wavelength nodes in nm.
        east_arcsec: Planet offset east of the star in arcsec.
        north_arcsec: Planet offset north of the star in arcsec.
        contrast: Planet-to-star flux ratio, a scalar or shape ``(W, Q)``.
        time_d: Time in days from the scene epoch.
        sources: Node-effective sources of ``scene`` on ``quadrature`` at
            ``time_d`` alone, from ``scene.sources``; see :func:`rates`.
        uniform_sky: Sky transmission profile read at the separation in
            lambda/D, or None. Required when the scene has a disk or a local
            zodi.
        time_s: Live exposure time in seconds, a scalar or shape ``(W,)``.
        frame_time_s: Seconds per detector frame, a scalar or shape ``(W,)``.
        post_processing: Residual fractions and background variance factor of
            the speckle and exozodi subtraction.

    Returns:
        S/N per band, shape ``(W,)``, NaN where the read is not
        ``Rates.supported``.
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
        sources=sources,
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
    """Return the photon-counting frame time per band, shape ``(W, *batch)`` seconds.

    The frame time at which a photon-counting EMCCD sees a mean of 1/6.73
    electrons per aperture pixel per frame, ``1 / (6.73 r)`` with ``r`` the
    electron rate of all sources, planet included, per aperture pixel (the
    cadence pyEDITH takes for clock-induced charge). ``rates`` is one aperture
    in photons/s/nm or electrons/s with ``n_pix`` set. The batch axes are those
    of ``rates``: ``()`` for one aperture, ``(K, T)`` for a record from
    :func:`scene_rates`. ``optixstuff.snr`` and ``optixstuff.exposure_time``
    take a frame time that is a scalar or ``(W,)``, so a cadence per planet is a
    ``jax.vmap`` of the calculator over the batch axes.

    Args:
        rates: Aperture rates of one or more apertures, in photons/s/nm or
            electrons/s, with ``n_pix`` set.
        detector: Detector whose yield converts photon rates to electrons.

    Returns:
        Frame time in seconds per band, shape ``(W, *batch)``.
    """
    if rates.n_pix is None:
        raise ValueError("rates must carry n_pix, the aperture's pixel count")
    electrons = detector.electrons(rates) if rates.carrier == "photon" else rates
    per_pixel = electrons.total() / electrons.n_pix
    return 1.0 / (_OCCUPANCY_DIVISOR * per_pixel)
