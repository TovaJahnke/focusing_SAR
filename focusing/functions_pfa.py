#!/usr/bin/env python3
"""

Script with main functions to focus an Spotlight SAR CPHD (FX-domain spotlight) into a TIFF image using a flexible
polar-format style regridding pipeline.

Imported into the focus_geotiff.py script.

Key assumptions:
- CPHD 1.1, monostatic, spotlight
- DomainType=FX (range frequency samples already available)
- Signal stored as I/Q float pairs (CF8 represented as [..., 2] float32)

"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from cphd_safe_reader import open_cphd_reader

C_MPS = 299_792_458.0 #speed of light in meters per second
WGS84_A = 6378137.0 #WGS84 semi-major axis of the Earth in meters
WGS84_E2 = 6.69437999014e-3 #WGS84 eccentricity squared

"""Converts ECEF coordinates to geodetic coordinates meaning latitude, longitude, and height above the ellipsoid WGS84."""
def ecef_to_llh(x: float, y: float, z: float) -> Tuple[float, float, float]: 
    lon = math.atan2(y, x) #longitude is the angle between the x-axis and the projection of the point onto the xy-plane
    p = math.hypot(x, y) #distance from the z-axis to the point
    lat = math.atan2(z, p * (1.0 - WGS84_E2)) #latitude is the angle between the z-axis and the point
    for _ in range(8): #We iterate because converting from ECEF to geodetic latitude on an ellipsoidal Earth has no closed-form solution, so we use a few fast refinement steps to reach machine-precision accuracy.
        sin_lat = math.sin(lat) #sine of the latitude
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n #height above the ellipsoid
        lat_new = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h))) #new latitude
        if abs(lat - lat_new) < 1e-14:
            lat = lat_new #if the new latitude is close to the old latitude, we break the loop
            break
        lat = lat_new #otherwise, we update the latitude
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat) #normal radius of curvature
    h = p / max(math.cos(lat), 1e-12) - n #height above the ellipsoid
    return lat, lon, h #returns the latitude, longitude, and height above the ellipsoid

"""want a local scene plane basis vectors in ECEF. ECEF is a geocentric coordinate system with the origin at the center of the Earth. ENU is a local coordinate system with the origin at the reference point."""
#x is east, y is north, z is up, PFA woeks in a local cartesian coordinate system.
def enu_basis_from_ecef_ref(ref_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat, lon, _ = ecef_to_llh(float(ref_ecef[0]), float(ref_ecef[1]), float(ref_ecef[2])) #converts the reference point to geodetic coordinates
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64) #east is the unit vector in the east direction
    north = np.array( #north is the unit vector in the north direction
        [
            -math.sin(lat) * math.cos(lon),
            -math.sin(lat) * math.sin(lon),
            math.cos(lat),
        ],
        dtype=np.float64,
    )
    up = np.array( #up is the unit vector in the up direction
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ],
        dtype=np.float64,
    )
    return east, north, up #returns the east, north, and up basis vectors


def to_complex_iq(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[-1] == 2:
        return raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)
    if np.iscomplexobj(raw):
        return raw.astype(np.complex64)
    raise ValueError(f"Unsupported signal shape/dtype for CPHD raw block: {raw.shape} {raw.dtype}")


""" Tukey window is a window function that uces sidelobes and can help reduce artifacts in the focused image. Adjust the alpha parameter to control the amount of tapering."""
def tukey(n: int, alpha: float = 0.2) -> np.ndarray: # n is the number of samples in the window, alpha is the transition width
    if n <= 1: #edge case: if the number of samples is less than or equal to 1, return a window of all ones
        return np.ones(max(n, 1), dtype=np.float32)
    if alpha <= 0: #edge case: if the transition width is less than or equal to 0, return a window of all ones
        return np.ones(n, dtype=np.float32)
    if alpha >= 1: #edge case: if the transition width is greater than or equal to 1, return a Hann window
        return np.hanning(n).astype(np.float32)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64) #creates a linear space of n samples from 0 to 1
    w = np.ones(n, dtype=np.float64) #initializes a window of all ones
    first = x < alpha / 2.0 #first is the index of the samples before the transition width
    last = x >= 1.0 - alpha / 2.0 #last is the index of the samples after the transition width
    w[first] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[first] - alpha / 2.0))) #applies the Tukey window to the first samples
    w[last] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[last] - 1.0 + alpha / 2.0))) #applies the Tukey window to the last samples
    return w.astype(np.float32) #returns the window as a float32 array
#More taper (higher alpha) → lower sidelobes, better PSLR/ISLR, but wider mainlobe (slightly worse resolution).
#Less taper (lower alpha) → sharper resolution, but more ringing around bright targets.


"""scale_to_uint16 chooses a dB display window, normalizes and gamma-adjusts the image,
then quantizes it to 16-bit grayscale for saving as a TIFF."""


def scale_to_uint16(
    img_db: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.7,
    db_min: float | None = None,
    db_max: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    if db_min is None or db_max is None:
        lo = float(np.percentile(img_db, lower_pct))
        hi = float(np.percentile(img_db, upper_pct))
    else:
        lo, hi = float(db_min), float(db_max)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((img_db - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, 1.0 / float(gamma))
    return (x * 65535.0 + 0.5).astype(np.uint16)


"""scale_to_uint8 chooses a dB display window, normalizes and gamma-adjusts the image,
then quantizes it to 8-bit grayscale for saving as a TIFF."""

def scale_to_uint8(
    img_db: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.7,
    db_min: float | None = None,
    db_max: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    if db_min is None or db_max is None:
        lo = float(np.percentile(img_db, lower_pct))
        hi = float(np.percentile(img_db, upper_pct))
    else:
        lo, hi = float(db_min), float(db_max)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((img_db - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, 1.0 / float(gamma))
    return (x * 255.0 + 0.5).astype(np.uint8) 


#KB is theoretically best practice kernel with high interpolation accuracy but nn or bilinear are good alternatives
def kaiser_bessel_kernel(t: np.ndarray, width: float, beta: float, denom: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be > 0")
    x = (2.0 * t) / width # t is the offset between the sample and the center of the kernel and width is the width of the kernel
    out = np.zeros_like(x, dtype=np.float32) #initializes an array of zeros with the same shape as x
    m = np.abs(x) < 1.0 #m is the index of the samples where the kernel is not zero
    if np.any(m):
        out[m] = (np.i0(beta * np.sqrt(1.0 - x[m] * x[m])) / denom).astype(np.float32) #applies the Kaiser-Bessel kernel to the samples
    return out #outputs weights between 0 and 1 for each sample.
#np.i0 is the modified Bessel function of the first kind, order 0.
#beta is the shape parameter of the kernel, larger beta → narrower mainlobe, more sidelobes.
#denom is the normalization factor for the kernel.

