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
    [3] theta_j    [rad] ← the ray's angle (same for every row in the segment)
"""

import time
import numpy as np
from math import radians, degrees, cos, sin, acos, atan2, pi, sqrt, ceil
from multiprocessing import Pool, cpu_count

import balmis as bm
import fcc_constants
from fcc_constants import beta_step, alfa_step, angle_eps, R_e, int_vl_height
from pathlib import Path

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
    start = time.time()
    max_psi = 90 
    psi_n   = int(max_psi / psi_step)

    worker_args = [(interc_data.copy(), psi_step, j) for j in range(psi_n)]
    int_table_0 = np.empty(psi_n, dtype=object)

    print("\n[Angle-Outer] Building interceptor table…")
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for i, (traj, max_rv) in enumerate(pool.imap(_int_trj_worker, worker_args)):
            int_table_0[i] = traj
            if i % 20 == 0 or i == psi_n - 1:
                print(f"\r  {i / (psi_n - 1) * 100:5.1f}%  psi={i * psi_step:5.1f}°  "
                      f"apogee={(max_rv - R_e) / 1000:7.1f} km", end="")

    end = time.time()
    print("\n[Angle-Outer] Interceptor table complete; calculation time: ", round(end - start , 2), " seconds")
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

    Output columns per ray:  [psi_deg, time, rv_m, theta_j_rad]
    """
    start = time.time()
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
    print("minalltheta ", degrees(theta_min), " maxalltheta ", degrees(theta_max))


    script_dir = Path(__file__).parent
    output_dir = script_dir / "images" / "concentric"
    output_dir.mkdir(parents=True, exist_ok=True)
    for current_theta in np.arange(theta_min, theta_max + 0.001, 0.005):
        crossings = raw_ray_crossings(int_table_v1, current_theta)
        plot_raw_ray_folding(
            crossings, 
            current_theta, 
            save_path = output_dir / f"con_{current_theta}.png")

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

    print(f"\n[Angle-Outer] Hits collected. Building interception table...")

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

    end = time.time()
    print("\n[Angle-Outer] Interception table complete; calculation time: ", round(end - start , 2), " seconds")
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


# ===========================================================================
# 4.  Ray monotonicity check ("phantom" interception point detector)
# ===========================================================================

def check_ray_monotonicity(int_table,
    type="concentric",
    plot=True,
    plot_metric="dpsi",
    save_path=None,
    top_n=3):
    """
    Diagnostic for the both interceptor tables.

    Each ray theta_j/beta_j is built by interception_table() from whichever raw
    trajectory (of the whole psi_launch fan) reaches (theta_j/beta_j, rv) earliest,
    independently for every radius bucket along the ray. Consecutive rows in
    a ray -- "further along the beam", ascending rv -- are therefore not
    necessarily samples of one continuous flight: the row that wins at one
    radius bucket can come from a very different psi_launch than the row
    that wins at the next. Where that happens, short_search's linear
    interpolation between the two rows blends fragments of two unrelated
    trajectories (e.g. one depressed, one lofted) into a "phantom" hit that
    doesn't correspond to any real interceptor flight.

    This function does not try to fix that -- it just walks each ray from
    its lowest row ("underground", rv closest to R_e) to its highest
    ("maximum altitude") and flags every place where the
    requirement -- dpsi is bigger than the psi_step -- is
    violated, i.e. interpolation not only happening between two nearest 
    rays.

    Parameters
    ----------
    int_table : object ndarray
        Output of interception_table(...): one entry per
        theta/beta ray, each an (n, 4) array of [psi_deg, time, rv, theta_j/beta_j],
        sorted ascending by rv.

    Returns
    -------
    violations : list of dict
        One entry per violating (row i, row i+1) pair, in ray order. Keys:
        ray_index, theta_j/beta_j_deg, row_index, rv_a_km, rv_b_km, t_a, t_b, dt
        (t_b - t_a, negative by construction), psi_a_deg, psi_b_deg,
        dpsi_deg (|psi_b - psi_a|, the tell for "two different trajectories
        stitched together").
    stats : dict
        n_rays, n_rays_checked (>=2 rows), n_rays_with_violation,
        n_pairs_checked, n_violations, worst_dt (most negative), and, among
        violating pairs only, min/max/mean of dpsi_deg -- compare against
        dpsi_all_mean (the same, over every pair, violating or not) to see
        whether violations really do concentrate at large psi_launch jumps.
    top_by_dt : list of dict
        Top 3 violations with the worst (most negative) dt values.
    top_by_dpsi : list of dict
        Top 3 violations with the largest dpsi_deg jumps.
    """

    BIN_SIZE_DEG = 1.0
    TOP_N = 3

    violations = []
    n_rays = len(int_table)
    n_rays_checked = 0
    n_pairs_checked = 0
    dpsi_all = []
    angle_key = "beta_j_deg" if type == "ilp" else "theta_j_deg"

    for k in range(n_rays):
        ray = int_table[k]
        if ray is None or len(ray) < 2:
            continue
        n_rays_checked += 1
        angle_rad = ray[0, 3]

        for i in range(len(ray) - 1):
            psi_a, t_a, rv_a = ray[i, 0], ray[i, 1], ray[i, 2]
            psi_b, t_b, rv_b = ray[i + 1, 0], ray[i + 1, 1], ray[i + 1, 2]

            if rv_b <= rv_a:
                continue  # not actually further along the beam; shouldn't
                          # happen given the table is rv-sorted, but guard anyway

            n_pairs_checked += 1
            dpsi = abs(psi_b - psi_a)

            dpsi_all.append(dpsi)

            if t_b < t_a:
                violations.append({
                    'ray_index': k,
                    angle_key: degrees(angle_rad),
                    'row_index': i,
                    'rv_a_km': rv_a / 1000.0, 'rv_b_km': rv_b / 1000.0,
                    't_a': float(t_a), 't_b': float(t_b), 'dt': float(t_b - t_a),
                    'psi_a_deg': float(psi_a), 'psi_b_deg': float(psi_b),
                    'dpsi_deg': float(dpsi),
                })

    viol_dpsi = [v['dpsi_deg'] for v in violations]
    stats = {
        'n_rays': n_rays,
        'n_rays_checked': n_rays_checked,
        'n_rays_with_violation': len(set(v['ray_index'] for v in violations)),
        'n_pairs_checked': n_pairs_checked,
        'n_violations': len(violations),
        'worst_dt': min((v['dt'] for v in violations), default=0.0),
        'dpsi_violations_min': min(viol_dpsi, default=None),
        'dpsi_violations_max': max(viol_dpsi, default=None),
        'dpsi_violations_mean': (sum(viol_dpsi) / len(viol_dpsi)) if viol_dpsi else None,
        'dpsi_all_mean': (sum(dpsi_all) / len(dpsi_all)) if dpsi_all else None,
    }
    angle_histogram = {}
    if violations:
        for v in violations:
            th = v[angle_key]
            bin_start = float(int(th // BIN_SIZE_DEG) * BIN_SIZE_DEG)
            bin_key = f"{bin_start:.1f}_to_{bin_start + BIN_SIZE_DEG:.1f}_deg"
            angle_histogram[bin_key] = angle_histogram.get(bin_key, 0) + 1

    if plot:
        plot_violation_histogram(
            violations, metric=plot_metric, type=type, save_path=save_path
        )

    # 3. Top 3 worst time inversions (most negative dt)
    top_by_dt = sorted(violations, key=lambda v: v['dt'])[:TOP_N]

    # 4. Top 3 largest launch angle discontinuities
    top_by_dpsi = sorted(violations, key=lambda v: v['dpsi_deg'], reverse=True)[:TOP_N]

    return violations, stats, angle_histogram, top_by_dt, top_by_dpsi

""" End of check_ray_monotonicity """

import matplotlib.pyplot as plt

def plot_violation_histogram(
    violations, metric="dpsi", type="concentric", save_path=None
):
    """Plots a 2D histogram overlaying ray angle against a violation severity metric.

    Parameters
    ----------
    violations : list of dict
        Output from check_ray_monotonicity().
    metric : str, optional ('dpsi' or 'dt')
        Metric to display on the y-axis.
    type : str, optional ('ilp' or 'concentric')
        Determines ray angle label semantics and plot title.
    save_path : str, optional
        Path to save the plot figure. If None, displays via plt.show().
    """
    if not violations:
        print("No violations found to plot!")
        return

    angle_key = "beta_j_deg" if type == "ilp" else "theta_j_deg"
    angle_symbol = r"\beta" if type == "ilp" else r"\theta"

    thetas = [v[angle_key] for v in violations]

    if metric == "dpsi":
        y_data = [v["dpsi_deg"] for v in violations]
        y_label = r"Launch Angle Gap $\Delta\psi$ (degrees)"
        cmap = "YlOrRd"
    elif metric == "dt":
        y_data = [v["dt"] for v in violations]
        y_label = r"Time Step $\Delta t$ (seconds)"
        cmap = "magma"
    else:
        raise ValueError("Metric must be either 'dpsi' or 'dt'")

    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(10, 6))
    counts, xedges, yedges, im = plt.hist2d(
        thetas, y_data, bins=[40, 40], cmap=cmap, cmin=1
    )
    

    plt.xlabel(
        rf"Ray Angle ${angle_symbol}$ (degrees)"
    )
    plt.ylabel(y_label)
    # plt.title(save_path)

    cb = plt.colorbar(im)
    cb.set_label("Number of Violations", rotation=270, labelpad=30)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def raw_ray_crossings(int_table_v1, ray_angle_rad, angle_column=3):
    """
    For ONE fixed ray angle (theta_j for concentric, beta_j for classic),
    walk every raw trajectory in the psi-launch fan (int_table_v1, the
    pre-culling Table 1 from interceptor_table()/int_trj_calc -- NOT the
    already-built ray table) and find where it crosses exactly this angle,
    with NO earliest-time-per-bucket merging and NO monotonicity filtering
    applied. This is the raw (psi -> r, t) relationship the table-building
    step has to summarize into a handful of rows -- if it isn't monotonic in
    psi, any table row that blends two of these points (because they passed
    a purely *local* r/t check against whatever was already kept) is mixing
    two physically disconnected trajectories, regardless of whether the
    final table looks locally monotonic.
 
    Parameters
    ----------
    int_table_v1 : object ndarray
        Table 1 -- the RAW trajectory fan, one entry per psi_launch, each a
        (n, 4) array of [psi_deg, time, rv, angle] (angle = alfa for
        concentric's [3], or the equivalent classic column -- pass whichever
        this fan actually has via angle_column).
    ray_angle_rad : float
        The fixed ray/beam angle to intersect every trajectory against.
    angle_column : int
        Which column of each trajectory holds the angle to intersect on
        (alfa for tables_v2b_a's fan is column 3; check tables_v2b_0's raw
        fan layout for the classic case and adjust if it differs).
 
    Returns
    -------
    crossings : list of (psi_deg, t, r), sorted ascending by psi_deg.
        Only trajectories whose angle range actually spans ray_angle_rad
        contribute a point.
    """
    crossings = []
    for traj in int_table_v1:
        if traj is None or len(traj) < 2:
            continue
        angle = traj[:, angle_column]
        psi_deg = traj[0, 0]
        for i in range(len(traj) - 1):
            a1, a2 = angle[i], angle[i + 1]
            lo, hi = (a1, a2) if a1 <= a2 else (a2, a1)
            if hi - lo < 1e-14:
                continue
            if lo <= ray_angle_rad <= hi:
                f = (ray_angle_rad - a1) / (a2 - a1)
                t = traj[i, 1] + f * (traj[i + 1, 1] - traj[i, 1])
                r = traj[i, 2] + f * (traj[i + 1, 2] - traj[i, 2])
                crossings.append((float(psi_deg), float(t), float(r)))
                break  # a single trajectory crosses a given angle once
                       # (angle is monotonic along one flight); stop after
                       # the first crossing found
 
    crossings.sort(key=lambda c: c[0])
    return crossings
 
 
def _find_direction_segments(crossings):
    """
    Split the psi-sorted crossings into maximal runs where r moves the same
    direction (non-decreasing, or non-increasing) as psi increases -- the
    "increasing"/"decreasing" sequences described in the discussion this
    accompanies. Returns a list of (start_idx, end_idx, direction) with
    direction in {'increasing', 'decreasing'} (inclusive indices into
    crossings), plus, for each segment, whether t moved consistently with r
    (t_consistent=True) or not -- a segment where r increases but t doesn't
    is exactly the "opposite direction" case called out above.
    """
    if len(crossings) < 2:
        return []
 
    segments = []
    r = [c[2] for c in crossings]
    t = [c[1] for c in crossings]
 
    seg_start = 0
    direction = None  # 'increasing' or 'decreasing'
 
    for i in range(1, len(crossings)):
        dr = r[i] - r[i - 1]
        step_dir = 'increasing' if dr >= 0 else 'decreasing'
        if direction is None:
            direction = step_dir
        elif step_dir != direction:
            segments.append(_make_segment(seg_start, i - 1, direction, r, t))
            seg_start = i - 1
            direction = step_dir
 
    segments.append(_make_segment(seg_start, len(crossings) - 1, direction, r, t))
    return segments
 
 
def _make_segment(i0, i1, direction, r, t):
    dr = r[i1] - r[i0]
    dt = t[i1] - t[i0]
    t_consistent = (dt >= 0) if direction == 'increasing' else (dt <= 0)
    return {'start_idx': i0, 'end_idx': i1, 'direction': direction, 't_consistent': t_consistent}
 
 
def plot_raw_ray_folding(crossings, ray_angle_deg, psi_range=None, save_path=None):
    """
    Four panels, stacked, all showing the SAME raw crossing points along one
    fixed ray, just against different independent variables -- so a fold
    that's ambiguous in one view is unmistakable in another:
 
      1. altitude vs psi   -- r(psi): where the ray crossing moves up/down
                              as the launch angle sweeps
      2. time vs psi       -- t(psi): same sweep, arrival time
      3. psi vs time       -- the inverse view: if this is not single-valued,
                              one arrival time corresponds to several launch
                              angles, i.e. interpolating psi from t is not
                              well-posed
      4. altitude vs time  -- r(t) along the ray: this is exactly the
                              (t, r) pair the table-building step sees and
                              filters on; loops/backtracking here are what
                              lets a point from a different psi-branch pass
                              a purely local monotonicity check
 
    Segments are coloured by direction of r vs psi (green = r increasing,
    red = r decreasing) and dashed where t moves opposite to r within that
    segment, consistently across all four panels, so the same segment is
    identifiable in every view.
 
    psi_range : optional (psi_min_deg, psi_max_deg) to zoom in, e.g. (0, 35).
    """
    if psi_range is not None:
        crossings = [c for c in crossings if psi_range[0] <= c[0] <= psi_range[1]]
 
    if len(crossings) < 2:
        print("Not enough crossings in this psi range to show anything.")
        return
 
    segments = _find_direction_segments(crossings)
    psi = [c[0] for c in crossings]
    t = [c[1] for c in crossings]
    alt_km = [(c[2]) / 1000.0 for c in crossings]
 
    fig, axes = plt.subplots(2, 2, figsize=(17, 17))
    ax1, ax2, ax3, ax4 = axes.flatten()
 
    colors = {'increasing': '#2e8b57', 'decreasing': '#c0392b'}
 
    # (x_data, y_data, xlabel, ylabel) per panel
    panels = [
        (psi, alt_km, 'launch angle psi, deg',   'altitude at ray crossing, km'),
        (psi, t,      'launch angle psi, deg',   'time at ray crossing, s'),
        (t,   psi,    'time at ray crossing, s', 'launch angle psi, deg'),
        (t,   alt_km, 'time at ray crossing, s', 'altitude at ray crossing, km'),
    ]
 
    for ax, (xd, yd, xlab, ylab) in zip(axes.flatten(), panels):
        for seg in segments:
            i0, i1 = seg['start_idx'], seg['end_idx']
            c = colors[seg['direction']]
            style = '-' if seg['t_consistent'] else '--'
            ax.plot(xd[i0:i1 + 1], yd[i0:i1 + 1], style, color=c, linewidth=2)
        ax.scatter(xd, yd, s=5, color='black', zorder=0.1)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle=':', alpha=0.6)
 
    ax1.set_title('Ray angle = {:.3f} deg  --  {} direction-reversal segment(s)\n'
                  '(green = r increasing with psi, red = r decreasing; '
                  'dashed = t moves opposite to r in that segment)'.format(
                      ray_angle_deg, len(segments)), fontsize=10)
 
    plt.tight_layout()
 
    for i, seg in enumerate(segments):
        print("  segment {}: psi {:.2f} to {:.2f} deg  ({}, t {}consistent)".format(
            i, psi[seg['start_idx']], psi[seg['end_idx']], seg['direction'],
            '' if seg['t_consistent'] else 'NOT '))
 
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print("Saved to", save_path)
    else:
        plt.show()
 
""" End of raw_ray_folding """