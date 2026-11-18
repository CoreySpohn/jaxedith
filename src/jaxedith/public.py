"""Read photon-native scenes into one aperture's spectral photon budget."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.conversions import arcsec_to_au, arcsec_to_rad
from hwoutils.planck import planck_photon_radiance
from hwoutils.spectral import Band, Quadrature, spectral_knots
from optixstuff import ElectronRates, FramePlan, PhotonRates, aperture_binding
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from zodi.jax import AYOExozodi, AYOZodi

_DEFAULT_ZODI = AYOZodi()
_DEFAULT_EXOZODI = AYOExozodi()


class ApertureRates(eqx.Module):
    """Photon-rate integrands and detector pixel counts for a weighted aperture.

    Mean rates use the aperture weights; Poisson variance rates use their
    squares. Pixel counts have one value per band and use the same two
    area moments in detector pixels.
    """

    mean: PhotonRates
    variance: PhotonRates
    n_pix: jax.Array
    n_pix_variance: jax.Array


def _bands(band_or_bands):
    return (band_or_bands,) if isinstance(band_or_bands, Band) else tuple(band_or_bands)


def aperture_quadrature(
    system,
    optical_path,
    band_or_bands,
    *,
    quadrature_order=4,
    zodi=_DEFAULT_ZODI,
    exozodi=_DEFAULT_EXOZODI,
    companion_flux_density=None,
) -> Quadrature:
    """Prepare shared host quadrature, split at every declared spectral table.

    Includes star, planet, optics, local zodi, analytic exozodi, QE and
    nearest-band dispatch breakpoints; disk knots only for exozodi="scene".
    Custom tabulated models must expose
    quadrature_knots_nm (property or method) or wavelengths_nm. Build before
    tracing model arrays and pass the result to aperture_rates and detector.
    """
    scene_disk = (
        system.disk if isinstance(exozodi, str) and exozodi == "scene" else None
    )
    knots = spectral_knots(
        system.star,
        system.planets,
        scene_disk,
        optical_path,
        zodi,
        exozodi,
        companion_flux_density,
    )
    return Quadrature.build(_bands(band_or_bands), knots, quadrature_order)


def _position(system, time_d, planet_index, separation_arcsec, position_angle_deg):
    """Select one scene planet or a background-only sky position."""
    if (planet_index is None) == (separation_arcsec is None):
        raise ValueError("supply exactly one of planet_index or separation_arcsec")
    if planet_index is not None:
        if not isinstance(planet_index, jax.core.Tracer) and not (
            0 <= planet_index < system.n_planets
        ):
            raise IndexError("planet_index is outside the system")
        times = jnp.atleast_1d(time_d)
        ephemeris = system.ephemeris(times)
        east, north = system.positions(times, ephemeris=ephemeris)[:, planet_index, 0]
        # A traced index cannot raise; out of range it yields a NaN position
        # (and so NaN rates) instead of JAX's silent clamp to the last planet.
        inside = (planet_index >= 0) & (planet_index < system.n_planets)
        east = jnp.where(inside, east, jnp.nan)
        north = jnp.where(inside, north, jnp.nan)
        return east, north, ephemeris
    angle = jnp.deg2rad(position_angle_deg)
    return separation_arcsec * jnp.sin(angle), separation_arcsec * jnp.cos(angle), None


def _matched_rates(
    prepared, detector, incident, separation, diameter, time_d, frame_time_s
):
    """Apply one band-integrated electron-space filter to all node integrands.

    Incident point-source fields are photons/s/nm. Diffuse fields are per sr,
    with thermal emission entering uniformly downstream of the coronagraph.
    """
    if frame_time_s is None:
        frame_time_s = eqx.error_if(
            jnp.asarray(1.0),
            (detector.clock_induced_charge_e_per_pixel_frame != 0)
            | (detector.read_noise_e_rms != 0),
            "Matched requires frame_time_s for detector CIC or read noise",
        )
    else:
        frame_time_s = jnp.asarray(frame_time_s)
        frame_time_s = eqx.error_if(
            frame_time_s,
            ~jnp.isfinite(frame_time_s) | (frame_time_s <= 0),
            "frame_time_s must be finite and positive",
        )
    quadrature = prepared.quadrature
    _, detector_variance = detector.mean_and_variance(
        ElectronRates(thermal=jnp.zeros(quadrature.nodes_nm.shape[:1])),
        FramePlan(frame_time_s, 1),
    )
    detected = (
        quadrature.weights_nm
        * detector.quantum_efficiency(quadrature.nodes_nm)
        * detector.survival_fraction
    )
    shot = detected * detector.excess_noise_factor_sq
    diffuse = incident.local_zodi + incident.exozodi + incident.stellar_companions
    native_sr = jnp.stack(
        [ref.stamps.pixel_solid_angle_sr for ref in prepared.references]
    )
    detector_pixels = native_sr / arcsec_to_rad(detector.pixel_scale_arcsec) ** 2
    uniform = (
        detector_variance / frame_time_s * detector_pixels
        + jnp.sum(shot * incident.thermal, axis=1) * native_sr
    )
    moments = []
    for i, ref in enumerate(prepared.references):
        moments.append(
            prepared.response(
                ref,
                separation / ref.arcsec_per_lod,
                time_d,
                diameter / ref.nodes[0].stellar_arcsec_per_lod,
                A=shot[i] * incident.star_leakage[i],
                B=shot[i] * diffuse[i],
                D=uniform[i],
                planet_spectrum=detected[i] * incident.planet[i],
            )
        )
    m = jax.tree.map(lambda *values: jnp.stack(values), *moments)

    def photons(tau, leak, sky_sr, n_pix):
        return PhotonRates(
            planet=incident.planet * tau,
            star_leakage=incident.star_leakage * leak,
            local_zodi=incident.local_zodi * sky_sr,
            exozodi=incident.exozodi * sky_sr,
            stellar_companions=incident.stellar_companions * sky_sr,
            thermal=incident.thermal * (native_sr * n_pix)[:, None],
        )

    return ApertureRates(
        photons(m.tau1, m.L1, m.Omega1, m.n_pix1),
        photons(m.tau2, m.L2, m.Omega2, m.n_pix2),
        m.n_pix1 * detector_pixels,
        m.n_pix2 * detector_pixels,
    )


def aperture_rates(
    system,
    optical_path,
    band_or_bands,
    *,
    time_d,
    planet_index=None,
    separation_arcsec=None,
    position_angle_deg=0.0,
    rule=DEFAULT_APERTURE_RULE,
    quadrature_order=4,
    zodi=_DEFAULT_ZODI,
    exozodi=_DEFAULT_EXOZODI,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
    reference_wavelength_nm=None,
    quadrature=None,
    prepared_aperture: aperture_binding.PreparedAperture | None = None,
    frame_time_s=None,
    temperature_K=270.0,
    eps_warm_T_cold=0.0,
    companion_flux_density=None,
) -> ApertureRates:
    """Return mean and variance integrands in photons/s/nm, with shape (W, Q).

    Select one planet by flattened scene index, or a background-only aperture
    by separation_arcsec and astronomical position_angle_deg (north to east).
    time_d is scalar days from the scene epoch. Bands are static Band objects.
    The rule and its solid angle are fixed at each band's centre, or at the
    supplied reference_wavelength_nm, a static Python float: aperture
    preparation runs on the host, so it cannot be traced. Pass
    prepared_aperture to jit or vmap over reads. Separations use the primary
    Frame with the performance record's diameter convention.

    Exozodi defaults to zodi.jax.AYOExozodi(): analytic radiance at projected physical
    separation system.dist_pc * separation_arcsec, also for background-only
    apertures. It evaluates one scalar per node, independent of system.disk.
    Supply exozodi=None for zero, or exozodi="scene" for disk light.
    Pass the same model to aperture_quadrature when preparing it separately.

    In scene mode, disk light approximates the footprint integral by centre
    radiance times aperture solid angle. The disk is evaluated directly at
    the aperture's sky position. No disk gives zero.

    Thermal light retains Planck emission times warm emissivity and cold
    transmission, without system throughput. companion_flux_density(wave,
    time_d) supplies photons/s/m^2/nm per reference lod^2, retaining the legacy
    stray-light area law; None means zero. These are node integrands too.
    One call describes one optical path; n_channels and npix_multiplier are
    legacy metadata, not implicit photon or pixel multipliers.

    Matched builds one inverse-variance filter per band using detected
    stellar leakage, local and exozodiacal light, companion surface brightness,
    thermal emission, and detector noise. The planet spectrum sets the signal
    template. Its shot noise enters only the returned variance. Supply the
    same positive frame_time_s used by the exposure plan when CIC or read
    noise is nonzero. Supply a scalar or one duration per band.
    Recalculate the filter when that frame time changes.
    """
    if jnp.ndim(time_d) != 0:
        raise ValueError("time_d must be scalar; use vmap for multiple times")
    bands = _bands(band_or_bands)
    if prepared_aperture is not None:
        if quadrature is not None and quadrature is not prepared_aperture.quadrature:
            raise ValueError("use the prepared aperture's quadrature")
        quadrature = prepared_aperture.quadrature
    if quadrature is None:
        quadrature = aperture_quadrature(
            system,
            optical_path,
            bands,
            quadrature_order=quadrature_order,
            zodi=zodi,
            exozodi=exozodi,
            companion_flux_density=companion_flux_density,
        )
    if quadrature.nodes_nm.shape[0] != len(bands):
        raise ValueError("quadrature must have one row per band")
    if prepared_aperture is None:
        prepared_aperture = aperture_binding.prepare_aperture(
            optical_path,
            bands,
            quadrature,
            rule=rule,
            reference_wavelength_nm=reference_wavelength_nm,
        )
    wave = quadrature.nodes_nm
    east, north, ephemeris = _position(
        system, time_d, planet_index, separation_arcsec, position_angle_deg
    )
    separation = jnp.hypot(east, north)
    star = system.star.photon_flux_density(wave, time_d, dist_pc=system.dist_pc)
    if planet_index is None:
        planet = jnp.zeros_like(wave)
    else:
        planet = jax.vmap(
            lambda w: system.planet_flux_densities(
                w, jnp.atleast_1d(time_d), ephemeris=ephemeris
            )[planet_index, 0]
        )(wave.reshape(-1)).reshape(wave.shape)
        # _position marks an out-of-range traced index with a NaN position.
        planet = jnp.where(jnp.isnan(east), jnp.nan, planet)
    if exozodi is None:
        disk = jnp.zeros_like(wave)
    elif isinstance(exozodi, str):
        if exozodi != "scene":
            raise ValueError('exozodi must be a radiance model, "scene", or None')
        disk = system.photon_radiance_at(wave, time_d, east, north)
    else:
        disk = exozodi.photon_radiance(
            wave, separation_au=arcsec_to_au(separation, system.dist_pc)
        )
    local = (
        jnp.zeros_like(wave)
        if zodi is None
        else zodi.photon_radiance(
            wave,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
    )
    diameter = system.star.angular_diameter_arcsec(system.dist_pc)
    area_m2 = optical_path.primary.collecting_area_m2
    throughput = jax.vmap(optical_path.system_throughput)(wave.reshape(-1)).reshape(
        wave.shape
    )
    common = area_m2 * throughput
    thermal_phot_per_s_m2_nm_sr = planck_photon_radiance(wave, temperature_K)
    companion = (
        None if companion_flux_density is None else companion_flux_density(wave, time_d)
    )

    if prepared_aperture.references[0].stamps is not None:
        reference_sr = jnp.stack(
            [
                arcsec_to_rad(ref.arcsec_per_lod) ** 2
                for ref in prepared_aperture.references
            ]
        )
        incident = PhotonRates(
            planet=common * planet,
            star_leakage=common * star,
            local_zodi=common * local,
            exozodi=common * disk,
            thermal=area_m2 * eps_warm_T_cold * thermal_phot_per_s_m2_nm_sr,
            stellar_companions=(
                0.0 if companion is None else common * companion / reference_sr[:, None]
            ),
        )
        return _matched_rates(
            prepared_aperture,
            optical_path.detector,
            incident,
            separation,
            diameter,
            time_d,
            frame_time_s,
        )
    responses, areas, variance_areas, scales = [], [], [], []
    for reference in prepared_aperture.references:
        scale = reference.arcsec_per_lod
        sep = separation / scale
        areas.append(reference.record.evaluate("core_area_lod2", sep))
        variance_areas.append(reference.record.evaluate("core_area_variance_lod2", sep))
        scales.append(scale)
        responses.append(
            jnp.stack(
                [
                    prepared_aperture.response(
                        node,
                        separation / node.arcsec_per_lod,
                        time_d,
                        diameter / node.stellar_arcsec_per_lod,
                    )
                    for node in reference.nodes
                ],
                axis=1,
            )
        )
    tau, leak, sky, tau_variance, leak_variance, sky_variance = jnp.stack(
        responses, axis=1
    )
    scales = jnp.stack(scales)
    areas = jnp.stack(areas)
    variance_areas = jnp.stack(variance_areas)

    def photon_rates(tau, leak, area, sky):
        background = common * sky
        solid_angle_sr = (area * arcsec_to_rad(scales) ** 2)[:, None]
        return PhotonRates(
            planet=common * tau * planet,
            star_leakage=common * leak * star,
            local_zodi=background * local * solid_angle_sr,
            exozodi=background * disk * solid_angle_sr,
            thermal=area_m2
            * eps_warm_T_cold
            * thermal_phot_per_s_m2_nm_sr
            * solid_angle_sr,
            stellar_companions=(
                0.0 if companion is None else background * area[:, None] * companion
            ),
        )

    pixels_per_lod2 = (scales / optical_path.detector.pixel_scale_arcsec) ** 2
    return ApertureRates(
        mean=photon_rates(tau, leak, areas, sky),
        variance=photon_rates(
            tau_variance, leak_variance, variance_areas, sky_variance
        ),
        n_pix=areas * pixels_per_lod2,
        n_pix_variance=variance_areas * pixels_per_lod2,
    )
