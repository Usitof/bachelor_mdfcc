import numpy as np
import numba as nb
import balmis as bm
from math import radians, degrees, ceil, cos, sin, atan, sqrt, asin, acos, pi, floor
from multiprocessing import Pool, cpu_count

# --- KONSTANTEN (Beispielwerte aus fcc_constants) ---
R_e = 6370000.0
angle_eps = 1e-12
set_beta_step = 0.1
int_vl_height = 0.0

# =================================================================
# COORDINATE CONVERTERS (EXACT CSM LOGIC)
# =================================================================

@nb.njit(cache=True, fastmath=True)
def missile_to_csm(xi, h, psi, phi):
    """
    Spherical Law of Cosines: Reduziert 3D Trajektorie auf 2D (theta, h).
    theta = Zentralwinkel zum Interceptor Launch Point (ILP).
    """
    cos_theta = cos(psi) * cos(xi) + sin(psi) * sin(xi) * cos(phi)
    theta = acos(max(-1.0, min(1.0, cos_theta)))
    return theta, h

@nb.njit(cache=True, fastmath=True)
def csm_to_mdfcc(theta, h):
    """
    Konvertiert CSM (theta, h) in Sensor-Koordinaten (Elevation, Slant Range).
    """
    r_p = R_e + h
    rs_sq = R_e**2 + r_p**2 - 2 * R_e * r_p * cos(theta)
    rs = sqrt(max(0.0, rs_sq))
    # Elevation epsilon
    arg = (r_p * sin(theta)) / rs
    epsilon = acos(max(-1.0, min(1.0, arg)))
    return epsilon, rs

# =================================================================
# INTERCEPTOR TABLE GENERATION
# =================================================================

def _int_trj_worker(args):
    int_data, psi_step, i_i = args
    i_psi = i_i * psi_step
    
    local_data = int_data.copy()
    local_data["flight_path_angle"] = 90.0 - i_psi
    
    # Simuliert den Flug ab ILP
    int_traj = bm.balmisflight(local_data, True, False)
    # Rückgabe: [psi, time, rad, theta] 
    # theta ist hier direkt der Bogenabstand zum ILP
    n = len(int_traj)
    out = np.zeros((n, 4))
    for j in range(n):
        out[j] = [i_psi, int_traj[j,0], int_traj[j,1], int_traj[j,2]]
    return out

def interception_table(interceptor_data, int_table_v1, beta_step=0.1):
    """ Erstellt die CSM-Abfangmatrix. Winkel von Zenit abwärts. """
    beta_min = min([min(t[:, 3]) for t in int_table_v1 if t is not None])
    beta_n = int(floor((90 - degrees(beta_min)) / beta_step) + 2)
    int_table_v2 = np.empty(beta_n, dtype=object)
    
    psi_n = len(int_table_v1)
    i_beg = np.full(psi_n, 1)

    for iter_k in range(beta_n):
        k_beta = radians(90 - iter_k * beta_step)
        current_points = []
        
        for i_psi in range(psi_n):
            traj = int_table_v1[i_psi]
            # Suche k_beta in Trajektorie i_psi
            for i_t in range(i_beg[i_psi], len(traj)-1):
                b1, b2 = traj[i_t, 3], traj[i_t+1, 3]
                if (b1 >= k_beta >= b2) or (b1 <= k_beta <= b2):
                    # Lineare Interpolation für Zeit und Radius
                    db = b2 - b1
                    frac = (k_beta - b1) / db if abs(db) > 1e-15 else 0.0
                    i_time = traj[i_t, 1] + frac * (traj[i_t+1, 1] - traj[i_t, 1])
                    i_rad = traj[i_t, 2] + frac * (traj[i_t+1, 2] - traj[i_t, 2])
                    
                    current_points.append([i_psi, i_time, i_rad, k_beta])
                    i_beg[i_psi] = i_t
                    break
        
        int_table_v2[iter_k] = np.array(current_points) if current_points else None

    # FLIP: So dass Index 0 = beta_min ist
    return np.flip(int_table_v2)

# =================================================================
# MISSILE SAMPLING (INTO CSM)
# =================================================================

@nb.njit(cache=True, fastmath=True)
def sample_missile_to_csm_numba(tar_trj, psi, phi, beta_step_val):
    """ Samples missile directly into the (theta, h) CSM system. """
    mis_traj_csm = np.empty((7200, 3)) # [time, r, theta_csm]
    
    # 1. Transformiere die gesamte Solver-Bahn in CSM-Winkel
    # tar_trj[:, 4] ist xi (Abstand ab MLP), tar_trj[:, 3] ist r
    xi_vals = tar_trj[:, 4]
    r_vals = tar_trj[:, 3]
    
    # 2. Resampling auf das CSM Grid (beta_step)
    # Hier wird der sphärische Kosinussatz für jeden Punkt angewendet
    i_m = 0
    # Wir iterieren über das gewünschte theta-Raster
    # (vereinfacht: wir nehmen die transformierten Punkte direkt)
    for i in range(len(tar_trj)):
        xi = xi_vals[i]
        r = r_vals[i]
        
        # Sphärische Transformation
        cos_theta = cos(psi) * cos(xi) + sin(psi) * sin(xi) * cos(phi)
        theta_csm = acos(max(-1.0, min(1.0, cos_theta)))
        
        mis_traj_csm[i, 0] = tar_trj[i, 0] # time
        mis_traj_csm[i, 1] = r
        mis_traj_csm[i, 2] = theta_csm
        i_m += 1
        
    return mis_traj_csm[:i_m]

# =================================================================
# TEST-VALIDIERUNG
# =================================================================

if __name__ == "__main__":
    # Test Degenerate Case: phi = 0
    psi_test = 0.1
    xi_test = 0.05
    phi_test = 0.0
    
    theta, _ = missile_to_csm(xi_test, 0, psi_test, phi_test)
    print(f"Test phi=0: theta={theta:.4f} (Erwartet: {abs(psi_test - xi_test):.4f})")
    
    # Test Non-Degenerate: Rechtwinkliges sphärisches Dreieck
    theta_nd, _ = missile_to_csm(pi/2, 0, pi/2, pi/2)
    print(f"Test phi=90: theta={theta_nd/pi*180:.1f}° (Erwartet: 90.0°)")
