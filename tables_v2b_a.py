import numpy as np
import numba as nb
import balmis as bm
from math import radians, degrees, ceil, cos, sin, atan, sqrt, asin, acos, pi, floor, atan2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import fcc_constants
from fcc_constants import beta_step as set_beta_step, angle_eps, R_e, int_vl_height

# =================================================================
# transformation (shell metric)
# =================================================================

@nb.njit(cache=True, fastmath=True)
def _transform_point_full_logic(z_rad, z_ang, i_psi, is_exo):
    h = z_rad - R_e  
    shell_id = np.floor(h / 10000.0) 
    
    if abs(z_ang) < angle_eps:
        i_rad = h
        i_ang = pi/2 if is_exo else radians(90.0 - i_psi)
    else:
        i_rad2 = R_e**2 + z_rad**2 - 2.0 * R_e * z_rad * cos(z_ang)
        i_rad = sqrt(max(0.0, i_rad2))
        arg = (z_rad * cos(z_ang) - R_e) / i_rad
        i_ang = asin(max(-1.0, min(1.0, arg)))
        
        if z_ang > pi:
            if z_ang < 2.0 * pi: i_ang = - pi - i_ang
            elif z_ang < 3.0 * pi: i_ang = - 2.0 * pi + i_ang
            
    return i_rad, i_ang, h, shell_id

# =================================================================
# worker (calculation per psi)
# =================================================================

def _int_trj_worker(args):
    int_data, psi_step, i_i = args
    i_psi = i_i * psi_step
    
    local_data = int_data.copy()
    local_data["vert_launch_height"] = int_vl_height if local_data["traj_type"] == "int_exo" else 0
    local_data["flight_path_angle"] = 90.0 - i_psi
    
    int_traj = bm.balmisflight(local_data, True, False)
    n = len(int_traj)
    
    # psi, time, r, beta, x, y, shell_id
    out = np.zeros((n, 7)) 
    is_exo = (local_data["traj_type"] == "int_exo")
    max_h = 0.0

    for j in range(n):
        z_t, z_rad, z_ang = int_traj[j, 0], int_traj[j, 1], int_traj[j, 2]
        i_rad, i_ang, h, s_id = _transform_point_full_logic(z_rad, z_ang, i_psi, is_exo)
        
        if h > max_h: max_h = h
        
        if i_i == 0:
            out[j] = [0.0, z_t, z_rad - R_e, pi/2, 0.0, z_rad - R_e, s_id]
        else:
            out[j] = [i_psi, z_t, i_rad, i_ang, i_rad * cos(i_ang), i_rad * sin(i_ang), s_id]
            
    return out, max_h

# =================================================================
# main function
# =================================================================

def interceptor_table(interc_data, ind_flight_dataprint=False, psi_step=0.25):
    max_psi = 120.0
    psi_n = int(max_psi / psi_step)
    
    worker_args = [(interc_data.copy(), psi_step, j) for j in range(psi_n)]
    int_table_0 = np.empty(psi_n, dtype=object)
    
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for i, (res_array, max_h_m) in enumerate(pool.imap(_int_trj_worker, worker_args)):
            int_table_0[i] = res_array
            if i % 20 == 0 or i == psi_n - 1:
                h_km = max_h_m / 1000.0
                print(f"\r Progress: {i/(psi_n-1)*100:3.0f}% | Psi: {i*psi_step:5.1f}° | Max-H: {h_km:6.1f} km", end='')
    
    print("\n[MDFCC] Interceptor table generated.")
    return int_table_0

def interception_table(interceptor_data, int_table_v1, psi_step=0.25, beta_step=0.1):
    from tqdm import tqdm
    beta_step_rad = radians(beta_step)
    
    valid = [t for t in int_table_v1 if t is not None and len(t) > 0]
    if not valid: return np.array([], dtype=object)

    all_th = np.concatenate([t[:, 3] for t in valid])
    beta_grid = np.arange(np.min(all_th), np.max(all_th) + 1e-7, beta_step_rad)
    
    out = []
    for b in tqdm(beta_grid, desc=" Interception Points"):
        segment_data = [] 
        for t in valid:
            idx_apogee = np.argmax(t[:, 2]) 
            t_ascent = t[:idx_apogee+1]
            th_vals = t_ascent[:, 3]

            if th_vals.min() - 1e-7 <= b <= th_vals.max() + 1e-7:
                found_interval = False
                for i in range(len(th_vals) - 1):
                    b1, b2 = th_vals[i], th_vals[i+1]
                    if (b1 <= b <= b2) or (b2 <= b <= b1):
                        i_t1, i_t2 = i, i+1
                        found_interval = True
                        break
                
                if found_interval:
                    rad_1, rad_2 = t_ascent[i_t1, 2], t_ascent[i_t2, 2]
                    beta1, beta2 = t_ascent[i_t1, 3], t_ascent[i_t2, 3]
                    time1, time2 = t_ascent[i_t1, 1], t_ascent[i_t2, 1]
                    s_id1, s_id2 = t_ascent[i_t1, 6], t_ascent[i_t2, 6]

                    # --- Geometrische Interpolation ---
                    db = beta1 - beta2
                    if abs(db) > 1e-12:
                        y = rad_2 * cos(db) - rad_1
                        x = rad_2 * sin(db)
                        gamma = atan2(y, x)
                        i_rad = rad_1 * cos(gamma) / cos(gamma + beta1 - b)

                        dr1_q = rad_1**2 + i_rad**2 - 2 * rad_1 * i_rad * cos(beta1 - b)
                        dr_q  = rad_1**2 + rad_2**2 - 2 * rad_1 * rad_2 * cos(db)
                        i_time = time1 + (time2 - time1) * sqrt(max(0.0, dr1_q/dr_q))
                    else:
                        i_rad = rad_1
                        i_time = time1

                    s_id = s_id1 if abs(b - beta1) < abs(b - beta2) else s_id2
                    # Struktur: [psi, time, rad, beta, x, y, shell_id]
                    segment_data.append([t[0, 0], i_time, i_rad, b, i_rad * cos(b), i_rad * sin(b), s_id])
        
        if len(segment_data) > 0:
            seg_array = np.array(segment_data)
            seg_array = seg_array[seg_array[:, 1].argsort()]
            out.append(seg_array)
        else:
            out.append(np.array([[0.0, 0.0, 0.0, b, 0.0, 0.0, -1.0]]))
            
    return np.array(out, dtype=object)



# =================================================================
# MISSILE SAMPLING
# =================================================================
@nb.njit(cache=True, fastmath=True)
def _spherical_3d_to_2d_angle(psi, xi, phi):
    cos_theta = cos(psi) * cos(xi) + sin(psi) * sin(xi) * cos(phi)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return acos(cos_theta)

@nb.njit(cache=True, fastmath=True)
def _numba_mis_traj_loop(tar_trj, beta_step, qbeta_beg, angle_eps, psi, phi, concentric):
    n_points = len(tar_trj)
    working_trj = np.empty((n_points, 5))
    
    # 1. Trajektorie koordinatenrein befüllen
    for i in range(n_points):
        working_trj[i, 0] = tar_trj[i, 0] # time
        
        if concentric:
            # KORREKTUR: Direkt die echten topozentrischen 3D-Werte des Startplatzes nutzen
            working_trj[i, 3] = tar_trj[i, 3] # Echte Sichtweite (r) statt geozentrischer Radius
            working_trj[i, 4] = tar_trj[i, 4] # Echter Höhenwinkel (beta_r) statt Zentralwinkel
        else:
            # Der flache Pfad bleibt für das alte Modell zu 100% identisch
            working_trj[i, 3] = tar_trj[i, 1] 
            working_trj[i, 4] = tar_trj[i, 2]

    mis_traj = np.empty((3600, 5)) 
    i_mbeta = 0 
    
    _step_rad = float(beta_step)
    current_qbeta_deg = qbeta_beg
    current_qbeta_rad = radians(current_qbeta_deg)
    
    # EXAKTE INDIZIERUNG: Kein unvollständiges Slicing
    diff = working_trj[0, 4] - current_qbeta_rad
    if abs(diff) < angle_eps:
        i_time = working_trj[0, 0]
        i_rad = working_trj[0, 3]
        mis_traj[i_mbeta, 0] = i_time
        mis_traj[i_mbeta, 1] = i_rad
        mis_traj[i_mbeta, 2] = current_qbeta_rad
        mis_traj[i_mbeta, 3] = i_rad * cos(current_qbeta_rad)
        mis_traj[i_mbeta, 4] = i_rad * sin(current_qbeta_rad)
        
        current_qbeta_deg += float(beta_step)
        current_qbeta_rad = radians(current_qbeta_deg)
        i_mbeta += 1

    # Hauptschleife über alle Trajektorien-Intervalle
    for i_tr in range(n_points - 1):
        beta_curr = working_trj[i_tr, 4]
        beta_next = working_trj[i_tr + 1, 4]
        
        while True:
            # Richtungsunabhängige Intervallprüfung
            if (beta_curr <= current_qbeta_rad <= beta_next) or (beta_next <= current_qbeta_rad <= beta_curr) or (abs(beta_curr - current_qbeta_rad) < angle_eps):
                
                rad_1 = working_trj[i_tr, 3]
                rad_2 = working_trj[i_tr + 1, 3]
                beta1 = working_trj[i_tr, 4]
                beta2 = working_trj[i_tr + 1, 4]
                
                db = beta1 - beta2
                if abs(db) > 1e-12:
                    gamma = atan2((rad_2 * cos(db) - rad_1), (rad_2 * sin(db)))
                    if abs(beta1 - current_qbeta_rad) < angle_eps:
                        i_rad = rad_1
                        dr1_q = 0.0
                    else:
                        i_rad = rad_1 * cos(gamma) / cos(gamma + beta1 - current_qbeta_rad)        
                        dr1_q = rad_1**2 + i_rad**2 - 2.0 * rad_1 * i_rad * cos(beta1 - current_qbeta_rad)
                    
                    dr_q = rad_1**2 + rad_2**2 - 2.0 * rad_1 * rad_2 * cos(db)
                else:
                    i_rad = rad_1
                    dr1_q = 0.0
                    dr_q = 0.0

                time1 = working_trj[i_tr, 0]
                time2 = working_trj[i_tr + 1, 0]
                
                if dr_q > 0.0 and dr1_q > 0.0:
                    i_time = time1 + (time2 - time1) * sqrt(dr1_q / dr_q)
                else:
                    i_time = time1
                
                # KORREKTUR: Zuweisung komplett explizit pro Element ausführen!
                if i_mbeta < 3600:
                    mis_traj[i_mbeta, 0] = i_time
                    mis_traj[i_mbeta, 1] = i_rad
                    mis_traj[i_mbeta, 2] = current_qbeta_rad
                    mis_traj[i_mbeta, 3] = i_rad * cos(current_qbeta_rad)
                    mis_traj[i_mbeta, 4] = i_rad * sin(current_qbeta_rad)
                    i_mbeta += 1
                
                current_qbeta_deg += float(beta_step)
                current_qbeta_rad = radians(current_qbeta_deg)
            else:
                break
                
    res = mis_traj[:i_mbeta]
    if len(res) > 0:
        argsw = np.argsort(res[:, 0])
        return res[argsw]
    return res


def mis_traj_sample(tar_trj, beta_step=None, psi_deg=0.0, phi_deg=0.0, concentric=True):
    if tar_trj.ndim == 1:
        tar_trj = tar_trj.reshape(1, -1)

    if beta_step is None:
        beta_step = 0.025
    local_eps = angle_eps if 'angle_eps' in globals() else 1e-7

    angles = []
    for i in range(len(tar_trj)):
        if concentric:
            theta = degrees(tar_trj[i, 4])
        else:
            theta = degrees(tar_trj[i, 2])
        angles.append(theta)
    
    min_theta = min(angles)
    qbeta_beg = int(ceil(min_theta / beta_step)) * beta_step
    
    res = _numba_mis_traj_loop(tar_trj, float(beta_step), float(qbeta_beg), float(local_eps), 0.0, 0.0, concentric)
    return res
