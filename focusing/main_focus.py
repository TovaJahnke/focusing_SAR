#!/usr/bin/env python3
#focus_pfa_geotiff.py is the main script for focusing the CPHD data and geocoding it to a GeoTIFF file.
"""
PFA focus + geocoding: Raw CPHD → Focus → Geocode → GeoTIFF (EPSG:4326)

Imports functions from functions_pfa.py, which contains the core PFA implementation and related utilities.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np


_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from cphd_safe_reader import open_cphd_reader
from functions_pfa.py import (
    C_MPS,
    WGS84_A,
    WGS84_E2,
    ecef_to_llh,
    enu_basis_from_ecef_ref,
    to_complex_iq,
    tukey,
    scale_to_uint16,
    scale_to_uint8,
    kaiser_bessel_kernel,
)



def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scene_extent_candidates_m(meta) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    scene = getattr(meta, "SceneCoordinates", None)
    if scene is None:
        return out

    area = getattr(scene, "ImageArea", None)
    if area is not None:
        x1y1 = getattr(area, "X1Y1", None)
        x2y2 = getattr(area, "X2Y2", None)
        if x1y1 is not None and x2y2 is not None:
            x1 = _safe_float(getattr(x1y1, "X", None))
            y1 = _safe_float(getattr(x1y1, "Y", None))
            x2 = _safe_float(getattr(x2y2, "X", None))
            y2 = _safe_float(getattr(x2y2, "Y", None))
            if None not in (x1, y1, x2, y2):
                out["image_area"] = (abs(x2 - x1), abs(y2 - y1))

    grid = getattr(scene, "ImageGrid", None)
    if grid is not None:
        iax = getattr(grid, "IAXExtent", None)
        iay = getattr(grid, "IAYExtent", None)
        if iax is not None and iay is not None:
            line_spacing = _safe_float(getattr(iax, "LineSpacing", None))
            num_lines = _safe_float(getattr(iax, "NumLines", None))
            sample_spacing = _safe_float(getattr(iay, "SampleSpacing", None))
            num_samples = _safe_float(getattr(iay, "NumSamples", None))
            if None not in (line_spacing, num_lines, sample_spacing, num_samples):
                out["image_grid"] = (abs(num_lines * line_spacing), abs(num_samples * sample_spacing))

    return out

#Get the recommended sizes for the nnft-x and nfft_y parameters based on the metadata in the CPHD file. 
#This is important because if the nfft sizes are too small, you will get a focused image that is smaller than the actual scene extent. 
#The recommendation is based on the metadata.
def _print_extent_recommendation_xy(
    meta,
    dx_native: float,
    dy_native: float,
    nfft_x: int,
    nfft_y: int,
    osf: int,
    keep_osf_extent: bool,
) -> None:
    candidates = _scene_extent_candidates_m(meta)
    if not candidates:
        return

    current_nx = nfft_x * osf if keep_osf_extent else nfft_x
    current_ny = nfft_y * osf if keep_osf_extent else nfft_y
    current_lx = current_nx * dx_native
    current_ly = current_ny * dy_native
    print(
        f"[scene-est] current focused extent: ~{current_lx:.1f} m x {current_ly:.1f} m "
        f"({'keep-osf-extent' if keep_osf_extent else 'center-cropped'})"
    )

    for label, (target_lx, target_ly) in candidates.items():
        rec_nfft_x = int(math.ceil(target_lx / max(dx_native * max(osf if keep_osf_extent else 1, 1), 1e-12)))
        rec_nfft_y = int(math.ceil(target_ly / max(dy_native * max(osf if keep_osf_extent else 1, 1), 1e-12)))
        print(f"[scene-est] metadata {label} extent: ~{target_lx:.1f} m x {target_ly:.1f} m")
        print(
            f"[scene-est] recommended --nfft-x/--nfft-y for {label}: "
            f">= {max(128, rec_nfft_x)} / {max(128, rec_nfft_y)} "
            f"(with --osf {osf}{' and --keep-osf-extent' if keep_osf_extent else ''})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PFA focus + geocoding → GeoTIFF (EPSG:4326)."
    )
    p.add_argument("cphd_path", type=Path, help="Input CPHD file")
    p.add_argument("--out", type=Path, default=Path("umbra_pfa_geotiff.tif"), help="Output GeoTIFF path")
    p.add_argument("--index", default=0, help="Channel index (int) or ID (e.g. Primary)") #Generally 0 or "Primary" for single-channel CPHDs, but can be set to other values for multi-channel files.

    p.add_argument("--v-start", type=int, default=0) #Start vector index for processing (default: 0)
    p.add_argument("--v-stop", type=int, default=-1) #Stop vector index for processing (default: -1, meaning process all vectors)
    p.add_argument("--s-start", type=int, default=0) #Start sample index for processing (default: 0)
    p.add_argument("--s-stop", type=int, default=-1) #Stop sample index for processing (default: -1, meaning process all samples)
    p.add_argument("--v-stride", type=int, default=1) #Stride for vector index (default: 1, meaning process every vector)
    p.add_argument("--s-stride", type=int, default=4) #Stride for sample index (default: 4, meaning process every 4th sample to reduce memory usage)
    p.add_argument(
        "--estimate-nfft-only",
        action="store_true",
        help="Print scene-extent-based nfft-x/nfft-y recommendation and exit before expensive focusing",
    ) #Get recommendation of grid sizes but exit before focusing start

    p.add_argument("--nfft", type=int, default=4096, help="Base FFT size for both x and y when --nfft-x/--nfft-y are not set")
    p.add_argument("--nfft-x", type=int, default=None, help="FFT size in x (range) direction")
    p.add_argument("--nfft-y", type=int, default=None, help="FFT size in y (azimuth) direction")
    p.add_argument("--grid", choices=["nearest", "bilinear", "kb"], default="bilinear")
    p.add_argument("--osf", type=int, default=2)
    p.add_argument(
        "--keep-osf-extent",
        action="store_true",
        help="Keep the full oversampled image extent instead of center-cropping back to nfft_x/nfft_y",
    ) #Oversampling can improve interpolation accuracy at the cost of a larger output image. By default, we center-crop back to nfft_x/nfft_y size, but with this flag we can keep the full osf-extended extent.
    p.add_argument("--kb-width", type=float, default=3.0) #parameter for kb kernel
    p.add_argument("--kb-beta", type=float, default=8.6) #parameter for kb kernel

    p.add_argument("--tukey-alpha", type=float, default=0.2) #Tukey parameter
    p.add_argument("--ml-az", type=int, default=1) #Multilook in azimuth (y)
    p.add_argument("--ml-rg", type=int, default=1) #Multilook in range (x)

    p.add_argument("--k-lower-pct", type=float, default=0.0) 
    p.add_argument("--k-upper-pct", type=float, default=100.0)
    p.add_argument("--k-pad-frac", type=float, default=0.0)

    p.add_argument("--db-min", type=float, default=-45) 
    p.add_argument("--db-max", type=float, default=-15)
    p.add_argument("--gamma", type=float, default=1.2)
    p.add_argument("--strict-db-window", action="store_true")
    p.add_argument("--uint8", action="store_true", help="Output 8-bit TIFF instead of 16-bit")

    p.add_argument(
        "--geocode-res",
        type=float,
        default=None,
        help="Geocoded output resolution in degrees (default: ~match input GSD)",
    )
    p.add_argument(
        "--geocode-oversample",
        type=float,
        default=1.0,
        help="Oversample factor for geocoded grid (default 1.0)",
    )
    p.add_argument(
        "--geocode-chunk-rows",
        type=int,
        default=512,
        help="Geocode output rows per chunk (lower = less RAM, potentially slower)",
    )
    p.add_argument(
        "--final-height",
        type=int,
        default=None,
        help="Final GeoTIFF height in pixels (overrides resolution-derived height)",
    )
    p.add_argument(
        "--final-width",
        type=int,
        default=None,
        help="Final GeoTIFF width in pixels (overrides resolution-derived width)",
    )
    p.add_argument(
        "--tiff-compression",
        choices=["none", "deflate", "lzw", "zstd"],
        default="deflate",
        help="GeoTIFF compression method (default: deflate)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0_total = time.perf_counter()

    cphd_path = args.cphd_path.expanduser().resolve()
    if not cphd_path.exists():
        raise FileNotFoundError(cphd_path)

    reader = open_cphd_reader(cphd_path)
    meta = reader.cphd_meta
    chan = meta.Data.Channels[0]
    nv_total = int(chan.NumVectors)
    ns_total = int(chan.NumSamples)

    v_stop = nv_total if args.v_stop < 0 else min(args.v_stop, nv_total)
    s_stop = ns_total if args.s_stop < 0 else min(args.s_stop, ns_total)
    v_start = max(0, args.v_start)
    s_start = max(0, args.s_start)
    v_slice = slice(v_start, v_stop, max(1, args.v_stride))
    s_slice = slice(s_start, s_stop, max(1, args.s_stride))
    idx = int(args.index) if str(args.index).isdigit() else str(args.index)

    print(f"[info] Reading CPHD: {cphd_path}")
    print(f"[info] Total vectors/samples: {nv_total} x {ns_total}")

    tx_pos = np.asarray(reader.read_pvp_variable("TxPos", idx, v_slice), dtype=np.float64)
    rcv_pos = np.asarray(reader.read_pvp_variable("RcvPos", idx, v_slice), dtype=np.float64)
    srp_pos = np.asarray(reader.read_pvp_variable("SRPPos", idx, v_slice), dtype=np.float64)
    fx1 = np.asarray(reader.read_pvp_variable("FX1", idx, v_slice), dtype=np.float64)
    fx2 = np.asarray(reader.read_pvp_variable("FX2", idx, v_slice), dtype=np.float64)
    nv = tx_pos.shape[0]
    ns = max(0, len(range(s_start, s_stop, max(1, args.s_stride))))

    pc = 0.5 * (tx_pos + rcv_pos)
    #los = pc - srp_pos
    los = srp_pos - pc

    los_norm = np.linalg.norm(los, axis=1, keepdims=True) + 1e-12
    u = los / los_norm
    srp_ref = np.mean(srp_pos, axis=0)
    east, north, up_vec = enu_basis_from_ecef_ref(srp_ref)
    ux = u @ east
    uy = u @ north

    k1 = (4.0 * np.pi / C_MPS) * fx1
    k2 = (4.0 * np.pi / C_MPS) * fx2
    kc_line = 0.5 * (k1 + k2)
    kc_x = float(np.mean(kc_line * ux))
    kc_y = float(np.mean(kc_line * uy))
    kx_end_a = k1 * ux - kc_x
    ky_end_a = k1 * uy - kc_y
    kx_end_b = k2 * ux - kc_x
    ky_end_b = k2 * uy - kc_y
    kx_all = np.concatenate([kx_end_a, kx_end_b])
    ky_all = np.concatenate([ky_end_a, ky_end_b])

    k_lo, k_hi = float(args.k_lower_pct), float(args.k_upper_pct)
    if not (0.0 <= k_lo < k_hi <= 100.0):
        raise ValueError("Require 0 <= --k-lower-pct < --k-upper-pct <= 100")
    kx_lo, kx_hi = np.percentile(kx_all, [k_lo, k_hi]).astype(np.float64)
    ky_lo, ky_hi = np.percentile(ky_all, [k_lo, k_hi]).astype(np.float64)
    pad_frac = max(0.0, float(args.k_pad_frac))
    padx = pad_frac * (kx_hi - kx_lo + 1e-12)
    pady = pad_frac * (ky_hi - ky_lo + 1e-12)
    kx_min, kx_max = kx_lo - padx, kx_hi + padx
    ky_min, ky_max = ky_lo - pady, ky_hi + pady

    nfft_x = int(args.nfft_x) if args.nfft_x is not None else int(args.nfft)
    nfft_y = int(args.nfft_y) if args.nfft_y is not None else int(args.nfft)
    if nfft_x < 128 or nfft_y < 128:
        raise ValueError("nfft_x and nfft_y must be >= 128")
    osf = int(args.osf)
    if osf not in (1, 2):
        raise ValueError("--osf must be 1 or 2")
    grid_nx = nfft_x * osf
    grid_ny = nfft_y * osf
    dkx = (kx_max - kx_min) / (grid_nx - 1)
    dky = (ky_max - ky_min) / (grid_ny - 1)
    if dkx <= 0 or dky <= 0:
        raise RuntimeError("Degenerate k-space extent.")
    dx_native = 2.0 * np.pi / max(kx_max - kx_min, 1e-12)
    dy_native = 2.0 * np.pi / max(ky_max - ky_min, 1e-12)
    _print_extent_recommendation_xy(
        meta=meta,
        dx_native=dx_native,
        dy_native=dy_native,
        nfft_x=nfft_x,
        nfft_y=nfft_y,
        osf=osf,
        keep_osf_extent=bool(args.keep_osf_extent),
    )
    if args.estimate_nfft_only:
        print("[done] Estimate-only mode: exiting before k-space gridding and focusing.")
        return

    t0_read = time.perf_counter()
    raw = reader.read_raw(v_slice, s_slice, index=idx, squeeze=True)
    z = to_complex_iq(raw)
    nv, ns = z.shape
    wv = tukey(nv, alpha=args.tukey_alpha).reshape(-1, 1)
    ws = tukey(ns, alpha=args.tukey_alpha).reshape(1, -1)
    z = z * (wv * ws)
    t_read = time.perf_counter() - t0_read
    print(f"[info] Loaded chip shape: {z.shape}")

    grid = np.zeros((grid_ny, grid_nx), dtype=np.complex64)
    wgt = np.zeros((grid_ny, grid_nx), dtype=np.float32)
    kb_denom = float(np.i0(args.kb_beta)) if args.grid == "kb" else 1.0
    kb_radius = int(math.ceil(float(args.kb_width) / 2.0)) if args.grid == "kb" else 0
    sample_idx = np.arange(ns, dtype=np.float64)
    inv_dkx = 1.0 / dkx
    inv_dky = 1.0 / dky
    kb_width = float(args.kb_width)
    kb_beta = float(args.kb_beta)

    t0_grid = time.perf_counter()
    for i in range(nv):
        if ns > 1:
            k_step = (k2[i] - k1[i]) / float(ns - 1)
        else:
            k_step = 0.0
        kx0 = k1[i] * ux[i] - kc_x
        ky0 = k1[i] * uy[i] - kc_y
        dkx_line = k_step * ux[i]
        dky_line = k_step * uy[i]
        xf = (kx0 - kx_min) * inv_dkx + sample_idx * (dkx_line * inv_dkx)
        yf = (ky0 - ky_min) * inv_dky + sample_idx * (dky_line * inv_dky)
        zi = z[i].astype(np.complex64, copy=False)
        if args.grid == "nearest":
            ix = np.rint(xf).astype(np.int32)
            iy = np.rint(yf).astype(np.int32)
            good = (ix >= 0) & (ix < grid_nx) & (iy >= 0) & (iy < grid_ny)
            if np.any(good):
                ixg, iyg = ix[good], iy[good]
                np.add.at(grid, (iyg, ixg), zi[good])
                np.add.at(wgt, (iyg, ixg), 1.0)
        elif args.grid == "bilinear":
            ix0 = np.floor(xf).astype(np.int32)
            iy0 = np.floor(yf).astype(np.int32)
            tx = (xf - ix0).astype(np.float32)
            ty = (yf - iy0).astype(np.float32)
            for di, dj, w in [
                (0, 0, (1 - tx) * (1 - ty)),
                (0, 1, tx * (1 - ty)),
                (1, 0, (1 - tx) * ty),
                (1, 1, tx * ty),
            ]:
                ix, iy = ix0 + dj, iy0 + di
                m = (ix >= 0) & (ix < grid_nx) & (iy >= 0) & (iy < grid_ny)
                if np.any(m):
                    np.add.at(grid, (iy[m], ix[m]), (zi[m] * w[m]))
                    np.add.at(wgt, (iy[m], ix[m]), w[m])
        elif args.grid == "kb":
            ix_c = np.rint(xf).astype(np.int32)
            iy_c = np.rint(yf).astype(np.int32)
            offsets = list(range(-kb_radius, kb_radius + 1))
            wx = []
            wy = []
            for ox in offsets:
                wx.append(kaiser_bessel_kernel(xf - (ix_c + ox), kb_width, kb_beta, kb_denom))
            for oy in offsets:
                wy.append(kaiser_bessel_kernel(yf - (iy_c + oy), kb_width, kb_beta, kb_denom))
            for oy in range(-kb_radius, kb_radius + 1):
                for ox in range(-kb_radius, kb_radius + 1):
                    ix, iy = ix_c + ox, iy_c + oy
                    w = (wx[ox + kb_radius] * wy[oy + kb_radius]).astype(np.float32)
                    m = (ix >= 0) & (ix < grid_nx) & (iy >= 0) & (iy < grid_ny) & (w > 0)
                    if np.any(m):
                        np.add.at(grid, (iy[m], ix[m]), zi[m] * w[m])
                        np.add.at(wgt, (iy[m], ix[m]), w[m])

    nz = wgt > 0
    grid[nz] /= wgt[nz]
    t_grid = time.perf_counter() - t0_grid

    img_os = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid)))
    if osf == 1 or args.keep_osf_extent:
        img_c = img_os
        if args.verbose and osf > 1 and args.keep_osf_extent:
            print(f"[info] Keeping full oversampled extent: {img_c.shape}")
    else:
        sx = (grid_nx - nfft_x) // 2
        sy = (grid_ny - nfft_y) // 2
        img_c = img_os[sy : sy + nfft_y, sx : sx + nfft_x]
        if args.verbose and osf > 1:
            print(f"[info] Center-cropped oversampled image to: {img_c.shape}")

    ml_az, ml_rg = max(1, int(args.ml_az)), max(1, int(args.ml_rg))
    if ml_az > 1 or ml_rg > 1:
        m, n = img_c.shape
        m2, n2 = m - (m % ml_az), n - (n % ml_rg)
        if m2 > 0 and n2 > 0:
            inten = np.abs(img_c[:m2, :n2]) ** 2
            inten_ml = inten.reshape(m2 // ml_az, ml_az, n2 // ml_rg, ml_rg).mean(axis=(1, 3))
            img_mag = np.sqrt(inten_ml).astype(np.float32)
        else:
            img_mag = np.abs(img_c).astype(np.float32)
    else:
        img_mag = np.abs(img_c).astype(np.float32)


    # ------------------------------------------------------------------
    H_img, W_img = img_mag.shape
    inten = img_mag.astype(np.float64) ** 2

    py, px = np.unravel_index(np.argmax(img_mag), img_mag.shape)
    peak_val = float(img_mag[py, px])

    win_radius = 10
    y0 = max(0, py - win_radius)
    y1 = min(H_img, py + win_radius + 1)
    x0 = max(0, px - win_radius)
    x1 = min(W_img, px + win_radius + 1)
    patch = inten[y0:y1, x0:x1]

    ml_r = 1
    my0 = max(0, py - ml_r - y0)
    my1 = min(patch.shape[0], py - y0 + ml_r + 1)
    mx0 = max(0, px - ml_r - x0)
    mx1 = min(patch.shape[1], px - x0 + ml_r + 1)

    mainlobe = patch[my0:my1, mx0:mx1]
    sidelobes = patch.copy()
    sidelobes[my0:my1, mx0:mx1] = 0.0

    main_peak = float(np.max(mainlobe)) if mainlobe.size else peak_val
    side_peak = float(np.max(sidelobes)) if np.any(sidelobes > 0) else 0.0
    pslr_db = 20.0 * np.log10(side_peak / (main_peak + 1e-12)) if side_peak > 0 else float("-inf")

    ml_energy = float(np.sum(mainlobe)) if mainlobe.size else 0.0
    sl_energy = float(np.sum(sidelobes))
    islr_db = (
        10.0 * np.log10(sl_energy / (ml_energy + 1e-12)) if sl_energy > 0 and ml_energy > 0 else float("-inf")
    )

    row = img_mag[py, :]
    col = img_mag[:, px]
    half = peak_val * 0.5

    def _fwhm_1d(profile: np.ndarray, peak_index: int, half_val: float) -> float:
        left = peak_index
        while left > 0 and profile[left] > half_val:
            left -= 1
        right = peak_index
        n_prof = profile.size
        while right < n_prof - 1 and profile[right] > half_val:
            right += 1
        return float(max(1, right - left))

    fwhm_row = _fwhm_1d(row, px, half)
    fwhm_col = _fwhm_1d(col, py, half)

    total_energy = np.sum(inten)
    if total_energy > 0:
        p = inten.ravel() / total_energy
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = float("nan")

    mean_mag = float(np.mean(img_mag))
    std_mag = float(np.std(img_mag))
    contrast = std_mag / (mean_mag + 1e-12) if mean_mag > 0 else float("nan")

    gx = np.diff(img_mag, axis=1)[:-1, :]
    gy = np.diff(img_mag, axis=0)[:, :-1]
    grad_energy = float(np.mean(gx**2 + gy**2))

    print(
        "[metrics] PSLR={:.1f} dB, ISLR={:.1f} dB, FWHM_row={:.2f} px, "
        "FWHM_col={:.2f} px, Entropy={:.3f}, Contrast={:.3f}, GradEnergy={:.3e}".format(
            pslr_db, islr_db, fwhm_row, fwhm_col, entropy, contrast, grad_energy
        )
    )

    img_db = 20.0 * np.log10(img_mag + 1e-8)
    db_min, db_max = args.db_min, args.db_max
    if db_min is not None and db_max is not None and not args.strict_db_window:
        if ((img_db >= float(db_min)) & (img_db <= float(db_max))).mean() < 0.005:
            db_min, db_max = None, None
    img_scaled = (
        scale_to_uint8(img_db, db_min=db_min, db_max=db_max, gamma=args.gamma)
        if args.uint8
        else scale_to_uint16(img_db, db_min=db_min, db_max=db_max, gamma=args.gamma)
    )

    dx = dx_native
    dy = dy_native
    print(f"[geo-debug] dx={dx:.6f} m, dy={dy:.6f} m, ratio dy/dx={dy/dx:.4f}")
    print(f"[geo-debug] expected ~0.24 m from product metadata (range/az pixel spacing)")
    
    #dx = 0.24 #specifically for capella
    #dy = 0.24 #specifically for capella


    srp_lat, srp_lon, srp_h = ecef_to_llh(float(srp_ref[0]), float(srp_ref[1]), float(srp_ref[2]))


    H, W = img_scaled.shape
    x_ul = (0 - W / 2.0 + 0.5) * dx 
    y_ul = (H / 2.0 - 0.5) * dy 

    

    corners_enu = [
        (x_ul, y_ul),
        (x_ul + (W - 1) * dx, y_ul),
        (x_ul + (W - 1) * dx, y_ul - (H - 1) * dy),
        (x_ul, y_ul - (H - 1) * dy),
    ]
    corners_ecef = [
        srp_ref + x * east + y * north for x, y in corners_enu
    ]
    corners_llh = [ecef_to_llh(px, py, pz) for px, py, pz in corners_ecef]
    lats = [math.degrees(lat) for lat, _, _ in corners_llh]
    lons = [math.degrees(lon) for _, lon, _ in corners_llh]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    margin = 1.0e-6
    lat_min = max(-90.0, lat_min - margin)
    lat_max = min(90.0, lat_max + margin)
    lon_min -= margin
    lon_max += margin

    gsd_lat_approx = abs(dy) / 111000.0
    gsd_lon_approx = abs(dx) / (111000.0 * max(math.cos(srp_lat), 1e-6))
    res_deg = args.geocode_res
    if res_deg is None:
        res_deg = min(gsd_lat_approx, gsd_lon_approx)
    res_deg = res_deg / float(args.geocode_oversample)
    if res_deg <= 0:
        res_deg = 1e-6

    nlat_auto = max(2, int((lat_max - lat_min) / res_deg) + 1)
    nlon_auto = max(2, int((lon_max - lon_min) / res_deg) + 1)
    nlat = nlat_auto if args.final_height is None else int(args.final_height)
    nlon = nlon_auto if args.final_width is None else int(args.final_width)
    if nlat < 2 or nlon < 2:
        raise ValueError("--final-height and --final-width must be >= 2 when provided")
    chunk_rows = max(1, int(args.geocode_chunk_rows))
    lat_outs = np.linspace(lat_max, lat_min, nlat)
    lon_outs = np.linspace(lon_min, lon_max, nlon)

    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        raise RuntimeError("Geocoding requires scipy. Install with: pip install scipy")

    dtype_out = np.uint8 if args.uint8 else np.uint16
    max_val = 255.0 if args.uint8 else 65535.0
    geocoded_img = np.zeros((nlat, nlon), dtype=dtype_out)
    img_src = img_scaled.astype(np.float32, copy=False)
    lon_rad = np.deg2rad(lon_outs)
    cos_lon = np.cos(lon_rad)[None, :]
    sin_lon = np.sin(lon_rad)[None, :]
    inv_dx = 1.0 / max(dx, 1e-12)
    inv_dy = 1.0 / max(dy, 1e-12)

    for row0 in range(0, nlat, chunk_rows):
        row1 = min(row0 + chunk_rows, nlat)
        lat_chunk = lat_outs[row0:row1]
        lat_rad = np.deg2rad(lat_chunk)[:, None]
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        n_ell = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

        x = (n_ell + srp_h) * cos_lat * cos_lon
        y = (n_ell + srp_h) * cos_lat * sin_lon
        z = (n_ell * (1.0 - WGS84_E2) + srp_h) * sin_lat

        vec_x = x - srp_ref[0]
        vec_y = y - srp_ref[1]
        vec_z = z - srp_ref[2]

        x_enu = vec_x * east[0] + vec_y * east[1] + vec_z * east[2]
        y_enu = vec_x * north[0] + vec_y * north[1] + vec_z * north[2]

        j_src = (x_enu - x_ul) * inv_dx
        i_src = (y_ul - y_enu) * inv_dy
        i_src = (H - 1) - i_src   # <-- ADD THIS LINE
        coords = np.vstack((i_src.ravel(), j_src.ravel()))
        chunk_vals = map_coordinates(
            img_src,
            coords,
            order=3, #1
            mode="constant",
            cval=0.0,
        ).reshape(row1 - row0, nlon)
        geocoded_img[row0:row1, :] = np.clip(chunk_vals, 0.0, max_val).astype(dtype_out)
        if args.verbose and (row0 == 0 or row1 == nlat or (row0 // chunk_rows) % 10 == 0):
            print(f"[info] Geocode chunk rows: {row0}:{row1} / {nlat}")

    dlon_out = (lon_max - lon_min) / max(nlon - 1, 1)
    dlat_out = (lat_min - lat_max) / max(nlat - 1, 1)

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tifffile
    except ImportError:
        raise RuntimeError("GeoTIFF output requires tifffile. Install with: pip install tifffile")

    geokeys = np.array([
        1, 1, 0, 3,
        1024, 0, 1, 2,
        1025, 0, 1, 1,
        2048, 0, 1, 4326,
    ], dtype=np.uint16)
    model_pixel_scale = (float(abs(dlon_out)), float(abs(dlat_out)), 0.0)
    tiepoint = (0.0, 0.0, 0.0, float(lon_min), float(lat_max), 0.0)
    extratags = [
        (33550, "d", 3, model_pixel_scale, False),
        (33922, "d", 6, tiepoint, False),
        (34735, "H", int(geokeys.size), geokeys, False),
    ]
    compression = None if args.tiff_compression == "none" else str(args.tiff_compression)
    tifffile.imwrite(
        str(out_path),
        geocoded_img,
        photometric="minisblack",
        compression=compression,
        predictor=2 if compression is not None else False,
        tile=(512, 512),
        bigtiff=geocoded_img.nbytes >= (4 * 1024**3),
        extratags=extratags,
    )

    t_total = time.perf_counter() - t0_total
    print(f"[done] Wrote GeoTIFF (EPSG:4326): {out_path}")
    print(f"[done] Geocoded size: {nlat} x {nlon}, ~{abs(dlat_out)*111e3:.2f}m x {abs(dlon_out)*111e3*math.cos(srp_lat):.2f}m GSD")
    print(f"[done] Timing (s): read={t_read:.2f}, grid={t_grid:.2f}, total={t_total:.2f}")


if __name__ == "__main__":
    main()
