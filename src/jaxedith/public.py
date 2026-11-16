"""Read photon-native scenes into one aperture's spectral photon budget."""

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import arcsec2rad, c, h, k_B, nm2m, um2nm
from hwoutils.conversions import wavenumber_cm_to_nm
from hwoutils.grids import PixelGrid, SkyOrientation
from hwoutils.map_coordinates import map_coordinates
from hwoutils.spectral import Band, Quadrature
from optixstuff import MultiBandCoronagraph, PhotonRates
from optixstuff.coronagraph import AbstractScalarCoronagraph
from skyscapes.background import AYOZodi, LeinertZodi
from yippy import Circular
from zodi.tables import TABLE19_WAVELENGTH_UM

_DEFAULT_RULE = Circular(0.7)
_DEFAULT_ZODI = AYOZodi()


def _bands(band_or_bands):
    return (band_or_bands,) if isinstance(band_or_bands, Band) else tuple(band_or_bands)


def _spectral_knots(model):
    """Collect declared wavelength tables recursively at the host boundary."""
    if model is None:
        return []
    knots = []
    declared = getattr(model, "quadrature_knots_nm", None)
    if callable(declared):
        declared = declared()
    if declared is not None:
        knots.extend(np.asarray(declared).ravel())
    wavelengths = getattr(model, "wavelengths_nm", None)
    if wavelengths is not None:
        knots.extend(np.asarray(wavelengths).ravel())
    wavenumbers = getattr(model, "nu_grid", None)
    if wavenumbers is not None:
        knots.extend(np.asarray(wavenumber_cm_to_nm(wavenumbers)).ravel())
    if isinstance(model, MultiBandCoronagraph):
        centers = np.sort(model.band_centers_nm)
        knots.extend((centers[1:] + centers[:-1]) / 2)
    if isinstance(model, eqx.Module):
        for field in fields(model):
            child = getattr(model, field.name)
            if isinstance(child, eqx.Module):
                knots.extend(_spectral_knots(child))
            elif isinstance(child, tuple):
                for item in child:
                    if isinstance(item, eqx.Module):
                        knots.extend(_spectral_knots(item))
    return knots


def aperture_quadrature(
    system,
    optical_path,
    band_or_bands,
    *,
    quadrature_order=4,
    zodi=_DEFAULT_ZODI,
    companion_flux_density=None,
) -> Quadrature:
    """Prepare shared host quadrature, split at every declared spectral table.

    Includes star, planet, disk, optics, local zodi, QE and nearest-band
    dispatch breakpoints. Custom tabulated models must expose
    quadrature_knots_nm (property or method) or wavelengths_nm. Build before
    tracing model arrays and pass the result to aperture_rates and detector.
    """
    knots = _spectral_knots(system) + _spectral_knots(optical_path)
    knots += _spectral_knots(zodi) + _spectral_knots(companion_flux_density)
    if isinstance(zodi, (AYOZodi, LeinertZodi)):
        knots.extend(TABLE19_WAVELENGTH_UM * um2nm)
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


def _node_response(coro, separation_lod, wave, time_d, rule):
    """Read node curves and sampled sky transmission, with traced band dispatch."""
    if isinstance(coro, MultiBandCoronagraph):
        values = jnp.stack(
            [
                _node_response(band, separation_lod, wave, time_d, rule)
                for band in coro.bands
            ]
        )
        nearest = jnp.argmin(
            jnp.abs(wave[:, None] - jnp.asarray(coro.band_centers_nm)), axis=1
        )
        return jnp.take_along_axis(values, nearest[None, None, :], axis=0)[0]
    tau = jax.vmap(
        lambda w: coro.throughput(separation_lod, w, time_d=time_d, rule=rule)
    )(wave)
    leak = jax.vmap(
        lambda w: coro.leakage_fraction(separation_lod, w, time_d=time_d, rule=rule)
    )(wave)
    if isinstance(coro, AbstractScalarCoronagraph):
        # Scalar image methods are documented zero placeholders; the
        # calibrated scalar sky transmission is the physical quantity.
        sky = jax.vmap(
            lambda w: coro.occulter_transmission(separation_lod, w, time_d=time_d)
        )(wave)
    else:
        px, py = coro.grid.unit_to_pixel(separation_lod, 0.0)
        coordinates = jnp.array([[py], [px]])
        sky = jax.vmap(
            lambda w: map_coordinates(
                coro.background_transmission(w, grid=coro.grid).data,
                coordinates,
                order=1,
                mode="constant",
                cval=jnp.nan,
            )[0]
        )(wave)
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
    rule=_DEFAULT_RULE,
    quadrature_order=4,
    zodi=_DEFAULT_ZODI,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
    reference_wavelength_nm=None,
    quadrature=None,
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

    Disk light is the centre radiance times aperture solid angle, not a
    footprint average. Without disk_grid the one-pixel grid is built eagerly
    at the planet. For traced moving positions supply a fixed arcsec sky grid
    covering them; bilinear sampling gives NaN outside its support. No disk
    gives zero. Prepare quadrature before tracing wavelength-table leaves.

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
    if quadrature is None:
        quadrature = aperture_quadrature(
            system,
            optical_path,
            bands,
            quadrature_order=quadrature_order,
            zodi=zodi,
            companion_flux_density=companion_flux_density,
        )
    if quadrature.nodes_nm.shape[0] != len(bands):
        raise ValueError("quadrature must have one row per band")
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
    disk = _disk_radiance(system, wave, time_d, east, north, disk_grid)
    local = (
        jnp.zeros_like(wave)
        if zodi is None
        else zodi.photon_radiance(
            wave,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
    )
    coro = optical_path.coronagraph
    tau, leak, sky, solid_angle, areas = [], [], [], [], []
    for i, band in enumerate(bands):
        ref = (
            band.center_nm
            if reference_wavelength_nm is None
            else reference_wavelength_nm
        )
        reference_coro = (
            coro.band(ref) if isinstance(coro, MultiBandCoronagraph) else coro
        )
        frame = optical_path.primary.frame(
            ref, reference_coro.curves(rule).frame.diameter_convention
        )
        sep = separation / frame.lod_to_arcsec(ref)
        area = reference_coro.core_area(sep, ref, time_d=time_d, rule=rule)
        areas.append(area)
        solid_angle.append(area * (frame.lod_to_arcsec(ref) * arcsec2rad) ** 2)
        response = _node_response(coro, sep, wave[i], time_d, rule)
        tau.append(response[0])
        leak.append(response[1])
        sky.append(response[2])
    omega = jnp.stack(solid_angle)[:, None]
    area_m2 = optical_path.primary.collecting_area_m2
    throughput = jax.vmap(optical_path.system_throughput)(wave.reshape(-1)).reshape(
        wave.shape
    )
    common = area_m2 * throughput
    background = common * jnp.stack(sky)
    lam_m = wave * nm2m
    exponent = h * c / (lam_m * k_B * temperature_K)
    # exp(-x)/(1-exp(-x)) avoids overflow in cold visible-band emission.
    planck = 2 * c / lam_m**4 * jnp.exp(-exponent) / (-jnp.expm1(-exponent)) * nm2m
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
        local_zodi=background * local * omega,
        exozodi=background * disk * omega,
        thermal=area_m2 * eps_warm_T_cold * planck * omega,
        stellar_companions=companions,
    )
