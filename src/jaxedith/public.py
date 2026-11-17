"""Read photon-native scenes into one aperture's spectral photon budget."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_au, arcsec_to_rad
from hwoutils.grids import PixelGrid, SkyOrientation
from hwoutils.map_coordinates import map_coordinates
from hwoutils.planck import planck_photon_radiance
from hwoutils.spectral import Band, Quadrature, spectral_knots
from optixstuff import MultiBandCoronagraph, PhotonRates, YippyCoronagraph
from optixstuff.coronagraph import AbstractScalarCoronagraph, AbstractTableCoronagraph
from yippy import DEFAULT_APERTURE_RULE, Circular, HalfMaximum, Matched
from zodi.jax import AYOExozodi, AYOZodi

_DEFAULT_ZODI = AYOZodi()
_DEFAULT_EXOZODI = AYOExozodi()


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
        return east, north, ephemeris
    angle = jnp.deg2rad(position_angle_deg)
    return separation_arcsec * jnp.sin(angle), separation_arcsec * jnp.cos(angle), None


def _sky_grid(east, north):
    """One host pixel whose centre is the requested east/north sky position."""
    return PixelGrid.regular(
        (1, 1),
        1.0,
        "arcsec",
        center=(-float(east), -float(north)),
        orientation=SkyOrientation(0.0, parity=1),
    )


def _disk_radiance(system, wave, time_d, east, north, disk_grid):
    """Sample centre radiance, optionally interpolating a caller's fixed grid."""
    if system.disk is None:
        return jnp.zeros_like(wave)
    grid = _sky_grid(east, north) if disk_grid is None else disk_grid
    image = system.photon_radiance(wave, time_d, grid=grid)
    if disk_grid is None:
        return image.data[..., 0, 0]
    x, y = grid.orientation.sky_to_detector(east, north)
    px, py = grid.unit_to_pixel(x, y)
    coordinates = jnp.array([[py], [px]])
    values = image.data.reshape((-1, *grid.shape))
    return jax.vmap(
        lambda plane: map_coordinates(
            plane, coordinates, order=1, mode="constant", cval=jnp.nan
        )[0]
    )(values).reshape(wave.shape)


class PreparedAperture(eqx.Module):
    """Host-prepared geometry and records for one fixed angular aperture.

    Rebuild when the instrument, bands, quadrature or reference rule changes.
    Scene fluxes, positions, stellar diameters and time remain runtime inputs
    to aperture_rates. Node adapters retain the prepared leakage weights.
    """

    quadrature: Quadrature
    references: tuple
    nodes: tuple
    leakage_models: tuple
    stellar_scales: tuple


def prepare_aperture(
    optical_path,
    band_or_bands,
    quadrature,
    *,
    rule=DEFAULT_APERTURE_RULE,
    reference_wavelength_nm=None,
) -> PreparedAperture:
    """Prepare node rules and curves on the host, before tracing the reader.

    An angular radius is theta = r_ref * lambda_ref / D.
    At a node r_q = theta / (lambda_q / D) = r_ref * lambda_ref / lambda_q.
    Likewise s_q = s_arcsec / frame.lod_to_arcsec(lambda_q).
    Omega = A_ref * (lambda_ref / D)^2 stays fixed, not A_ref*(lambda_q/D)^2.

    Scalar coronagraph constants describe the calibrated fixed-angular
    aperture here: their area is in reference lambda/D squared. They have
    no PSF dilation law; their wavelength-dependent scalar methods are kept.
    HalfMaximum and Matched require an upstream frozen-template dilation API
    for polychromatic table models; changing a threshold is not dilation.
    """
    bands = _bands(band_or_bands)
    waves = np.asarray(quadrature.nodes_nm)
    if waves.shape[0] != len(bands):
        raise ValueError("quadrature must have one row per band")
    coro = optical_path.coronagraph
    models = coro.bands if isinstance(coro, MultiBandCoronagraph) else (coro,)

    def index(w):
        return (
            int(np.argmin(abs(np.asarray(coro.band_centers_nm) - w)))
            if isinstance(coro, MultiBandCoronagraph)
            else 0
        )

    def curves(model, aperture_rule):
        try:
            return model, model.curves(aperture_rule)
        except ValueError:
            if not isinstance(model, YippyCoronagraph):
                raise
            # The adapter constructor is the upstream host preparation API.
            prepared = YippyCoronagraph(backend=model._backend, rules=(aperture_rule,))
            return prepared, prepared.curves(aperture_rule)

    references, nodes, leakage_models, stellar_scales = [], [], [], []
    for band, row in zip(bands, waves, strict=True):
        ref = (
            band.center_nm
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )
        if isinstance(rule, (HalfMaximum, Matched)) and any(
            w != ref and not isinstance(models[index(w)], AbstractScalarCoronagraph)
            for w in row
        ):
            raise NotImplementedError(
                f"{type(rule).__name__} needs yippy/optixstuff support for "
                "dilating a frozen reference aperture/template at each separation"
            )
        _, reference = curves(models[index(ref)], rule)
        frame = optical_path.primary.frame(ref, reference.frame.diameter_convention)
        references.append((index(ref), ref, frame.lod_to_arcsec(ref), reference))
        prepared_row, model_row, scale_row = [], [], []
        for w in row:
            model = models[index(w)]
            scalar = isinstance(model, AbstractScalarCoronagraph)
            if scalar or w == ref:
                scaled = rule
            elif isinstance(rule, Circular):
                # replace keeps every other rule field, such as the centre.
                scaled = replace(rule, radius_lod=rule.radius_lod * ref / float(w))
            else:
                raise TypeError("rule must be Circular, HalfMaximum or Matched")
            leakage_model, record = curves(model, scaled)
            node_frame = optical_path.primary.frame(w, record.frame.diameter_convention)
            # Diameter is in the stellar product frame, independent of the primary.
            if not isinstance(leakage_model, AbstractTableCoronagraph):
                stellar_frame = leakage_model.grid.frame
            else:
                try:
                    stellar_frame = leakage_model.stellar_map(
                        w, 0.0, grid=None
                    ).grid.frame
                except NotImplementedError:
                    stellar_frame = leakage_model.grid.frame
            prepared_row.append((index(w), scaled, record, node_frame.lod_to_arcsec(w)))
            model_row.append(leakage_model)
            scale_row.append(stellar_frame.lod_to_arcsec(w))
        nodes.append(tuple(prepared_row))
        leakage_models.append(tuple(model_row))
        stellar_scales.append(tuple(scale_row))
    return PreparedAperture(
        quadrature,
        tuple(references),
        tuple(nodes),
        tuple(leakage_models),
        tuple(stellar_scales),
    )


def _node_response(coro, record, separation_lod, wave, time_d, rule, stellar_diam_lod):
    """Evaluate prepared curves and sampled sky transmission at one node."""
    if isinstance(coro, AbstractScalarCoronagraph):
        tau = coro.throughput(separation_lod, wave, time_d=time_d, rule=rule)
        sky = coro.occulter_transmission(separation_lod, wave, time_d=time_d)
    else:
        tau = record.evaluate("throughput", separation_lod)
        px, py = coro.grid.unit_to_pixel(separation_lod, 0.0)
        coordinates = jnp.array([[py], [px]])
        sky = map_coordinates(
            coro.background_transmission(wave, grid=coro.grid).data,
            coordinates,
            order=1,
            mode="constant",
            cval=jnp.nan,
        )[0]
    leak = coro.leakage_fraction(
        separation_lod,
        wave,
        time_d=time_d,
        rule=rule,
        stellar_diam_lod=stellar_diam_lod,
    )
    return jnp.stack((tau, leak, sky))


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
    prepared_aperture=None,
    disk_grid=None,
    temperature_K=270.0,
    eps_warm_T_cold=0.0,
    companion_flux_density=None,
) -> PhotonRates:
    """Return photons/s/nm with shape (W, Q), with no detector yield applied.

    Select one planet by flattened scene index, or a background-only aperture
    by separation_arcsec and astronomical position_angle_deg (north to east).
    time_d is scalar days from the scene epoch. Bands are static Band objects.
    The rule and its solid angle are fixed at each band's centre, or at the
    supplied scalar reference_wavelength_nm. Separations use the primary
    Frame with the performance record's diameter convention.

    Exozodi defaults to zodi.jax.AYOExozodi(): analytic radiance at projected physical
    separation system.dist_pc * separation_arcsec, also for background-only
    apertures. It evaluates one scalar per node, ignoring system.disk and
    disk_grid. Supply exozodi=None for zero, or exozodi="scene" for disk light.
    Pass the same model to aperture_quadrature when preparing it separately.

    In scene mode, disk light is centre radiance times aperture solid angle, not a
    footprint average. Without disk_grid the one-pixel grid is built eagerly
    at the planet. For traced moving positions supply a fixed arcsec sky grid
    covering them; bilinear sampling gives NaN outside its support. No disk
    gives zero. Prepare quadrature and prepare_aperture before tracing table
    models; pass prepared_aperture to reuse its records and quadrature.

    Thermal light retains Planck emission times warm emissivity and cold
    transmission, without system throughput. companion_flux_density(wave,
    time_d) supplies photons/s/m^2/nm per reference lod^2, retaining the legacy
    stray-light area law; None means zero. These are node integrands too.
    One call describes one optical path; n_channels and npix_multiplier are
    legacy metadata, not implicit photon or pixel multipliers.
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
        prepared_aperture = prepare_aperture(
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
    if exozodi is None:
        disk = jnp.zeros_like(wave)
    elif isinstance(exozodi, str):
        if exozodi != "scene":
            raise ValueError('exozodi must be a radiance model, "scene", or None')
        disk = _disk_radiance(system, wave, time_d, east, north, disk_grid)
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
    tau, leak, sky, solid_angles_sr, areas = [], [], [], [], []
    diameter = system.star.diameter_arcsec
    if callable(diameter):
        diameter = diameter(dist_pc=system.dist_pc)
    for i, (_, _ref, scale, record) in enumerate(prepared_aperture.references):
        sep = separation / scale
        area = record.evaluate("core_area_lod2", sep)
        areas.append(area)
        solid_angles_sr.append(area * arcsec_to_rad(scale) ** 2)
        response = jnp.stack(
            [
                _node_response(
                    prepared_aperture.leakage_models[i][j],
                    node_record,
                    separation / node_scale,
                    wave[i, j],
                    time_d,
                    node_rule,
                    diameter / prepared_aperture.stellar_scales[i][j],
                )
                for j, (_, node_rule, node_record, node_scale) in enumerate(
                    prepared_aperture.nodes[i]
                )
            ],
            axis=1,
        )
        tau.append(response[0])
        leak.append(response[1])
        sky.append(response[2])
    aperture_solid_angle_sr = jnp.stack(solid_angles_sr)[:, None]
    area_m2 = optical_path.primary.collecting_area_m2
    throughput = jax.vmap(optical_path.system_throughput)(wave.reshape(-1)).reshape(
        wave.shape
    )
    common = area_m2 * throughput
    background = common * jnp.stack(sky)
    thermal_phot_per_s_m2_nm_sr = planck_photon_radiance(wave, temperature_K)
    companions = (
        0.0
        if companion_flux_density is None
        else (
            background
            * jnp.stack(areas)[:, None]
            * companion_flux_density(wave, time_d)
        )
    )
    return PhotonRates(
        planet=common * jnp.stack(tau) * planet,
        star_leakage=common * jnp.stack(leak) * star,
        local_zodi=background * local * aperture_solid_angle_sr,
        exozodi=background * disk * aperture_solid_angle_sr,
        thermal=area_m2
        * eps_warm_T_cold
        * thermal_phot_per_s_m2_nm_sr
        * aperture_solid_angle_sr,
        stellar_companions=companions,
    )
