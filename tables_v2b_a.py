"""
tables_v2b_a.py  —  Angle-Outer Interceptor Table  (concentric / 3-D mode)
===========================================================================

Coordinate systems used here (names kept consistent with the thesis spec):

  balmis native  [time, rv, alfa]
    rv   = geocentric radius from Earth centre [m]      (column 1 of bmf_trj)
    alfa = geocentric central angle from launch point [rad]  (column 2 of bmf_trj)
    alfa starts at 0.0 at the interceptor launch point (ILP) and grows along
    the trajectory.

  Angle-Outer grid  [theta, (rv, time)]
    outer coordinate: theta  = geocentric central angle from ILP  [rad]
                               = alfa directly from balmis
    inner coordinate: rv     = geocentric radius from Earth centre [m]
    Each ray theta_j holds a monotone-culled list of (rv, time) pairs
    representing the interceptor envelope at that angle.

  Missile in CSM  (Spherical Law of Cosines)
    psi  = geocentric arc ILP → MLP  [rad]
    xi   = missile's geocentric angle from MLP along its own trajectory plane [rad]
    phi  = dihedral between missile plane and ILP-EC-MLP plane  [rad]
    theta_csm = acos(cos(psi)*cos(xi) + sin(psi)*sin(xi)*cos(phi))

Column layout of int_table_v1 (raw interceptor trajectories):
    [0] psi_launch  launch-fan angle  [deg]  (= 0 for vertical, grows with range)
    [1] time        flight time       [s]
    [2] rv          geocentric radius [m]
    [3] alfa/theta  geocentric central angle from ILP [rad]

Column layout of int_table_v2 (angle-outer grid, one entry per theta ray):
    [0] psi_launch  [deg]
    [1] time        [s]
    [2] rv          [m]   ← what the search in short_search compares against m_rad
    [3] theta_j     [rad] ← the ray's angle (same for every row in the segment)
"""

import numpy as np
from math import radians, degrees, cos, sin, acos, atan2, pi, sqrt, ceil
from multiprocessing import Pool, cpu_count

import balmis as bm
import fcc_constants
from fcc_constants import beta_step, alfa_step, angle_eps, R_e, int_vl_height

# ---------------------------------------------------------------------------
# Module-level Earth radius (metres)
# ---------------------------------------------------------------------------
_R_E = float(R_e) if R_e > 1_000_000 else float(R_e) * 1000.0

# ===========================================================================
# 1.  Raw interceptor trajectory fan  (Table 1)
# ===========================================================================

def _int_trj_worker(args):
    """
    Compute one interceptor trajectory for psi = i_i * psi_step degrees.

    balmisflight returns bmf_trj columns:
        0: time [s]
        1: rv   geocentric radius [m]
        2: alfa geocentric central angle from ILP [rad]

    Output columns: [psi_deg, time, rv, alfa]
    """
    int_data, psi_step, i_i = args
    psi_deg = i_i * psi_step

    local = int_data.copy()
    local["vert_launch_height"] = int_vl_height if local["traj_type"] == "int_exo" else 0
    local["flight_path_angle"]  = 90.0 - psi_deg   # 0° psi → vertical launch

    raw = bm.balmisflight(local, True, False)   # [time, rv, alfa, v, a]
    n   = len(raw)

    out    = np.zeros((n, 4))
    max_rv = 0.0

    for j in range(n):
        t_j    = raw[j, 0]
        rv_j   = raw[j, 1]   # already geocentric radius [m]
        alfa_j = raw[j, 2]   # geocentric angle from ILP [rad], starts at 0

        out[j, 0] = psi_deg
        out[j, 1] = t_j
        out[j, 2] = rv_j
        out[j, 3] = alfa_j

        if rv_j > max_rv:
            max_rv = rv_j

    return out, max_rv


def interceptor_table(interc_data, ind_flight_dataprint=False, psi_step=0.25):
    """
    Generate the raw interceptor trajectory fan.

    Returns
    -------
    int_table_0 : np.ndarray of object
        Array of length psi_n, each element is an (n_j × 4) array with
        columns [psi_deg, time, rv_m, alfa_rad].
    """
    # max_psi = 120 covers both positive and negative flight path angles
    # (flight_path_angle = 90° - psi).  For psi > 90°, the interceptor
    # is launched downwards – necessary to reach points behind the ILP.
    max_psi = 180   # increase to 180 if even flatter trajectories are needed
    psi_n   = int(max_psi / psi_step)

    worker_args = [(interc_data.copy(), psi_step, j) for j in range(psi_n)]
    int_table_0 = np.empty(psi_n, dtype=object)

    print("\n[Angle-Outer] Building raw interceptor fan …")
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for i, (traj, max_rv) in enumerate(pool.imap(_int_trj_worker, worker_args)):
            int_table_0[i] = traj
            if i % 20 == 0 or i == psi_n - 1:
                print(f"\r  {i / (psi_n - 1) * 100:5.1f}%  psi={i * psi_step:5.1f}°  "
                      f"apogee={(max_rv - R_e) / 1000:7.1f} km", end="")

    print("\n[Angle-Outer] Raw fan complete.")
    return int_table_0


# ===========================================================================
# 2.  Angle-Outer grid  (Table 2)
# ===========================================================================

def interception_table(interceptor_data, int_table_v1,
                       psi_step=0.25, theta_step_deg=fcc_constants.alfa_step, concentric=True):
    """
    Build the angle-outer interception table.

    Outer coordinate: theta  = geocentric angle from ILP  [rad]  (analogous to elevation)
    Inner coordinate: rv     = geocentric radius [m]              (analogous to slant range)

    For each ray theta_j: collect ALL trajectory points from ALL fan trajectories
    whose alfa (geocentric angle) equals theta_j (via interpolation).
    Then apply Pareto-front (time↑, rv↑) and backward-sweep monotonicity enforcement.

    Output columns per ray:  [psi_deg, time, rv_m, theta_j_rad]
    """
    # =========================================================================
    # MULTI-PACK UNWRAPPING GATEWAY (Sichert Form (122,) statt (3,))
    # =========================================================================
    if int_table_v1 is not None and hasattr(int_table_v1, "__len__") and len(int_table_v1) == 3:
        if isinstance(int_table_v1, (list, np.ndarray)) and not isinstance(int_table_v1, dict):
            int_table_v1 = int_table_v1[2]

    valid = [t for t in int_table_v1 if t is not None and len(t) > 1]
    if not valid:
        print("[Angle-Outer] ERROR: int_table_v1 is empty.")
        return np.array([], dtype=object)

    psi_n     = len(valid)
    all_alfa  = np.concatenate([t[:, 3] for t in valid])
    theta_min = float(np.min(all_alfa))
    theta_max = float(np.max(all_alfa))
    print("minallalfa ", degrees(theta_min), " maxallalfa ", degrees(theta_max))

    theta_step_rad = radians(theta_step_deg)
    theta_grid     = np.arange(theta_min, theta_max + 1e-9, theta_step_rad)
    """
    theta_step_deg = 0.01  # Use a smaller step (e.g., 0.1°)
    theta_min = 0.0       # Start at 0°
    theta_max = radians(90)  # Go up to 90°
    theta_grid = np.arange(theta_min, theta_max + 1e-9, radians(theta_step_deg))
    """
    n_rays         = len(theta_grid)

    print(f"\n[Angle-Outer] Grid: theta {degrees(theta_min):.3f}° → "
          f"{degrees(theta_max):.3f}°  step={theta_step_deg}°  rays={n_rays}")

    # Pre-build per-ray hit lists in one pass over ALL trajectory points.
    ray_hits = [[] for _ in range(n_rays)]

    for i_psi, traj in enumerate(valid):
        alfa  = traj[:, 3]   # monotone increasing geocentric angle [rad]
        n_pts = len(alfa)
        psi_deg_val = traj[0, 0]

        for i_t in range(n_pts - 1):
            a1, a2 = alfa[i_t], alfa[i_t + 1]
            if a2 < a1:
                continue   # safety: skip non-monotone step

            k_lo = int(np.searchsorted(theta_grid, a1, side='left'))
            k_hi = int(np.searchsorted(theta_grid, a2, side='right')) - 1
            k_lo = max(0,        k_lo)
            k_hi = min(n_rays-1, k_hi)

            rv1, rv2 = traj[i_t, 2], traj[i_t + 1, 2]
            tm1, tm2 = traj[i_t, 1], traj[i_t + 1, 1]
            da = a2 - a1

            for k in range(k_lo, k_hi + 1):
                theta_j = theta_grid[k]
                if theta_j < a1 - 1e-14 or theta_j > a2 + 1e-14:
                    continue
                factor = 0.0 if da < 1e-14 else (theta_j - a1) / da
                factor = max(0.0, min(1.0, factor))
                i_rv   = rv1 + factor * (rv2 - rv1)
                i_time = tm1 + factor * (tm2 - tm1)
                ray_hits[k].append([psi_deg_val, i_time, i_rv, theta_j])

    print(f"\n[Angle-Outer] Hits collected. Building monotone envelope per ray...")

    out = []

    for k, hits in enumerate(ray_hits):
        theta_j = theta_grid[k]

        if k % 20 == 0 or k == n_rays - 1:
            pct = 100.0 * k / max(n_rays - 1, 1)
            print(f"\r  {pct:5.1f}%  ray {k+1}/{n_rays}  theta={degrees(theta_j):7.3f}°", end="")

        if not hits:
            out.append(np.array([[0.0, 0.0, _R_E, theta_j]], dtype=np.float64))
            continue

        hits_arr = np.array(hits, dtype=np.float64)

        # Keep earliest time at each radius, full span, ascending radius.
        gaps = np.diff(np.unique(hits_arr[:, 2]))
        rv_tol = 0.25 * np.median(gaps) if len(gaps) else 0.0

        order = np.argsort(hits_arr[:, 2], kind='mergesort')
        hits_arr = hits_arr[order]

        tab_rows = []
        i = 0
        n = len(hits_arr)
        while i < n:
            j = i
            r0 = hits_arr[i, 2]
            best = hits_arr[i]
            while j < n and hits_arr[j, 2] - r0 <= rv_tol:
                if hits_arr[j, 1] < best[1]:
                    best = hits_arr[j]
                j += 1
            tab_rows.append(best)
            i = j

        tab = np.array(tab_rows, dtype=np.float64)

        out.append(tab if len(tab) > 0 else np.array([[0.0, 0.0, _R_E, theta_j]], dtype=np.float64))

    print("\n[Angle-Outer] Grid complete.")
    return np.array(out, dtype=object)


# ===========================================================================
# 3.  Missile trajectory resampling onto the beta grid  (for short_search)
# ===========================================================================

def mis_traj_sample(tar_trj, beta_step=None, psi_deg=0.0, phi_deg=0.0, concentric=True):
    """
    Resample the missile trajectory to a regular angular grid (beta_step).
    For concentric mode, psi_deg and phi_deg are ignored here; the spherical law of cosines
    mapping is done later in short_search.

    """
    psi_deg=0.0
    phi_deg=0.0

    if tar_trj.ndim == 1:
        tar_trj = tar_trj.reshape(1, -1)

    #if beta_step is None:
    #    beta_step = set_beta_step   # default from fcc_constants

    psi_rad = radians(psi_deg)
    phi_rad = radians(phi_deg)
    if concentric:
        bs_rad = radians(float(alfa_step))
    else:
        bs_rad = radians(float(beta_step))       

    alfa = tar_trj[:, 2]           # missile geocentric angles from MLP
    rv   = tar_trj[:, 1]           # geocentric radii [m]
    t    = tar_trj[:, 0]           # time [s]
    n    = len(tar_trj)

    a_min = float(np.min(alfa))
    a_max = float(np.max(alfa))

    max_pts = 3600

    out     = np.zeros((max_pts, 5))
    n_out   = 0

    for i in range(n - 1):
        a1, a2 = alfa[i], alfa[i + 1]
        a_lo, a_hi = (a1, a2) if a1 <= a2 else (a2, a1)

        if a_hi - a_lo < 1e-14:
            continue

        k_lo = int(ceil(a_lo / bs_rad))
        k_hi = int(a_hi / bs_rad)

        for k in range(k_lo, k_hi + 1):
            xi = k * bs_rad

            denom  = a2 - a1
            if abs(denom) < 1e-14:
                continue
            f = (xi - a1) / denom
            f = max(0.0, min(1.0, f))

            t_k  = t[i]  + f * (t[i + 1]  - t[i])
            rv_k = rv[i] + f * (rv[i + 1] - rv[i])

            if concentric:
                cos_theta_csm = cos(psi_rad) * cos(xi) + sin(psi_rad) * sin(xi) * cos(phi_rad)
                
                cos_theta_csm = max(-1.0, min(1.0, cos_theta_csm))
                theta_csm = acos(cos_theta_csm)
                
                if theta_csm > 1e-12:
                    sin_alpha = (sin(xi) * sin(phi_rad)) / sin(theta_csm)
                    sin_alpha = max(-1.0, min(1.0, sin_alpha))
                    
                    denom_csm = sin(psi_rad) * sin(theta_csm)
                    if abs(denom_csm) > 1e-12:
                        cos_alpha = (cos(xi) - cos(psi_rad) * cos_theta_csm) / denom_csm
                        cos_alpha = max(-1.0, min(1.0, cos_alpha))
                    else:
                        cos_alpha = 1.0
                    
                    x_m = _R_E * theta_csm * sin_alpha
                    z_m = _R_E * theta_csm * cos_alpha
                else:
                    x_m = 0.0
                    z_m = 0.0
            else:
                # 2-D classic: pass through existing x/z if available
                if tar_trj.shape[1] > 5:
                    x_m = tar_trj[i, 5] + f * (tar_trj[i + 1, 5] - tar_trj[i, 5])
                    z_m = tar_trj[i, 6] + f * (tar_trj[i + 1, 6] - tar_trj[i, 6]) \
                          if tar_trj.shape[1] > 6 else 0.0
                else:
                    x_m = z_m = 0.0

            if n_out < max_pts:
                out[n_out, 0] = t_k
                out[n_out, 1] = rv_k
                out[n_out, 2] = xi
                out[n_out, 3] = x_m
                out[n_out, 4] = z_m
                n_out += 1 

    return out[:n_out]
