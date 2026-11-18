"""Read photon-native scenes into one aperture's spectral photon budget."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.grids import PixelGrid
from hwoutils.planck import planck_photon_radiance
from hwoutils.spectral import Band, Quadrature
from optixstuff import ElectronRates, FramePlan, PhotonRates, aperture_binding
from optixstuff.apertures import DEFAULT_APERTURE_RULE
from zodi.jax import AYOZodi

_DEFAULT_ZODI = AYOZodi()


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


def _position(scene, time_d, planet_index, separation_arcsec, position_angle_deg):
    """Select one scene planet or a background-only sky position."""
    if (planet_index is None) == (separation_arcsec is None):
        raise ValueError("supply exactly one of planet_index or separation_arcsec")
    if planet_index is not None:
        if not isinstance(planet_index, jax.core.Tracer) and not (
            0 <= planet_index < scene.n_planets
        ):
            raise IndexError("planet_index is outside the scene")
        times = jnp.atleast_1d(time_d)
        ephemeris = scene.ephemeris(times)
        east, north = scene.positions(times, ephemeris=ephemeris)[:, planet_index, 0]
        # A traced index cannot raise; out of range it yields a NaN position
        # (and so NaN rates) instead of JAX's silent clamp to the last planet.
        inside = (planet_index >= 0) & (planet_index < scene.n_planets)
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
        ElectronRates(quadrature.bands),
        FramePlan(frame_time_s, 1),
    )
    yield_fraction = (
        detector.quantum_efficiency(quadrature.nodes_nm) * detector.survival_fraction
    )
    detected = quadrature.weights_nm * yield_fraction
    shot = detected * detector.excess_noise_factor_sq
    diffuse = incident.local_zodi + incident.exozodi + incident.stellar_companions
    native_sr = jnp.stack(
        [ref.stamps.pixel_solid_angle_sr for ref in prepared.references]
    )
    detector_grid = PixelGrid.regular(
        detector.shape, detector.pixel_scale_arcsec, "arcsec"
    )
    detector_pixels = native_sr / detector_grid.cell_area("sr")
    uniform = (
        detector_variance / frame_time_s * detector_pixels
        + quadrature.integrate(yield_fraction * incident.thermal)
        * detector.excess_noise_factor_sq
        * native_sr
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
            quadrature,
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
    scene,
    instrument,
    sampling,
    *,
    time_d,
    planet_index=None,
    separation_arcsec=None,
    position_angle_deg=0.0,
    rule=DEFAULT_APERTURE_RULE,
    quadrature_order=4,
    zodi=_DEFAULT_ZODI,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
    reference_wavelength_nm=None,
    frame_time_s=None,
    temperature_K=270.0,
    eps_warm_T_cold=0.0,
    companion_flux_density=None,
) -> ApertureRates:
    """Return mean and variance integrands in photons/s/nm, with shape (W, Q).

    Select one planet by flattened scene index, or a background-only aperture
    by separation_arcsec and astronomical position_angle_deg (north to east).
    time_d is scalar days from the scene epoch. Supply a Band or sequence of
    bands to build quadrature from the participating models, a Quadrature
    to reuse sampling, or a PreparedAperture to reuse the optical response.
    The rule and its solid angle are fixed at each band's centre, or at the
    supplied reference_wavelength_nm, a static Python float: aperture
    preparation runs on the host, so it cannot be traced. Pass
    a PreparedAperture to jit or vmap over reads. Separations use the primary
    Frame with the performance record's diameter convention.

    Exozodi is scene.disk, evaluated directly at the aperture's sky position
    and approximating the footprint integral by centre radiance times aperture
    solid angle. No disk gives zero. For the analytic AYO exozodi, build the
    scene with disk=skyscapes.disk.AYOExozodiDisk(); it evaluates at the
    projected physical separation scene.dist_pc * separation_arcsec, also for
    background-only apertures. Include the same disk in Quadrature.build when
    preparing the sampling separately.

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
    if isinstance(sampling, aperture_binding.PreparedAperture):
        prepared_aperture = sampling
    else:
        if isinstance(sampling, Quadrature):
            quadrature = sampling
        else:
            bands = (sampling,) if isinstance(sampling, Band) else tuple(sampling)
            models = [scene.star, instrument]
            if planet_index is not None:
                models.extend(scene.planets)
            models.extend(
                model
                for model in (zodi, scene.disk, companion_flux_density)
                if model is not None
            )
            quadrature = Quadrature.build(bands, *models, order=quadrature_order)
        prepared_aperture = aperture_binding.prepare_aperture(
            instrument,
            quadrature,
            rule=rule,
            reference_wavelength_nm=reference_wavelength_nm,
        )
    quadrature = prepared_aperture.quadrature
    wave = quadrature.nodes_nm
    east, north, ephemeris = _position(
        scene, time_d, planet_index, separation_arcsec, position_angle_deg
    )
    separation = jnp.hypot(east, north)
    star = scene.star.photon_flux_density(wave, time_d, dist_pc=scene.dist_pc)
    if planet_index is None:
        planet = jnp.zeros_like(wave)
    else:
        planet = jax.vmap(
            lambda w: scene.planet_flux_densities(
                w, jnp.atleast_1d(time_d), ephemeris=ephemeris
            )[planet_index, 0]
        )(wave.reshape(-1)).reshape(wave.shape)
        # _position marks an out-of-range traced index with a NaN position.
        planet = jnp.where(jnp.isnan(east), jnp.nan, planet)
    disk = scene.photon_radiance_at(wave, time_d, east, north)
    local = (
        jnp.zeros_like(wave)
        if zodi is None
        else zodi.photon_radiance(
            wave,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
    )
    diameter = scene.star.angular_diameter_arcsec(scene.dist_pc)
    area_m2 = instrument.primary.collecting_area_m2
    throughput = jax.vmap(instrument.system_throughput)(wave.reshape(-1)).reshape(
        wave.shape
    )
    common = area_m2 * throughput
    thermal_phot_per_s_m2_nm_sr = planck_photon_radiance(wave, temperature_K)
    companion = (
        None if companion_flux_density is None else companion_flux_density(wave, time_d)
    )

    if prepared_aperture.references[0].stamps is not None:
        reference_sr = jnp.stack(
            [ref.lod_cell_solid_angle_sr for ref in prepared_aperture.references]
        )
        incident = PhotonRates(
            quadrature,
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
            instrument.detector,
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
        solid_angle_sr = jnp.stack(
            [
                ref.solid_angle_sr(a)
                for ref, a in zip(prepared_aperture.references, area, strict=True)
            ]
        )[:, None]
        return PhotonRates(
            quadrature,
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

    pixels_per_lod2 = (scales / instrument.detector.pixel_scale_arcsec) ** 2
    return ApertureRates(
        mean=photon_rates(tau, leak, areas, sky),
        variance=photon_rates(
            tau_variance, leak_variance, variance_areas, sky_variance
        ),
        n_pix=areas * pixels_per_lod2,
        n_pix_variance=variance_areas * pixels_per_lod2,
    )
