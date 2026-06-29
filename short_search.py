import balmis as bm
import time
import tables_v2b_0 as tb
import tables_v2b_a as tb_con
import footprintv2 as fp
import numpy as np
from math import pi, radians, degrees, sin, cos, acos, asin, tan, ceil, sqrt
from fcc_constants import R_e, eps, angle_eps, out_time_freq, dist_limit, \
    beta_step, ang_quant, mrl_factor1, mrl_factor2
import fcc_constants

global dist_to_trj_min

shift_min = 1000 # min distance between ILP and MLP

#@profile
def short_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, 
                 t_delay=0, h_discr=0, mode2=False, debug_hits=False, sls_second=False, 
                 concentric=False, mrange=None):
    """
    Finds the earliest interception point.
    
    Parameters -- possibly also needed: detection altitude, and over horizon time
    ----------
    trj : array of [time, R, alfa]
        missile trajectory in polar coords, center in Earth center
    int_table : array of trajectories -- psi, time, r, beta
        sampled interception table (array)   
    h_int_min : 
        min acceptable height of interception
    t_int_lnc :
        earliest possible interceptor launch time counted from missile launch (missile burn time + interceptor launch delay)
    fi_opt : 
        by default: direction to impact point from ILP to MLP line;  >>>radians<<<
        in mode2: angle between trajectory plane and ILP-EarthCenter-MLP plane  >>>radians<<<
    dist_opt : 
        by default: distance to impact point from ILP;
        in mode2: shift (distance form ILP to MLP)
    t_delay : 
        launch delay after over horizon detection ('0' means criterium not used)
    h_discr :
        warhead discrimination height ('0' means criterium not used)

    Returns
    -------
    Coords and time of hit if found, false if not

    """
    oth_intercept = -999
    """
    if False : #mode2 : + line 1276 below -- ???
        if fi_opt > pi/2 :
            print("Omega > 90")
            return False
    """
    debug_print = False
    
    """
    ss_dist_max = trj[len(trj) - 1, 2] * R_e # this is missile range, dist_opt must be less than missile range
        
    if dist_opt >= ss_dist_max :
        print("\n>>> short_search exception_2: dist = {:f} km >= missile range = {:f} km<<<".format(dist_opt/1000, ss_dist_max/1000))
        return False
    """
    st = time.time()


    if not mode2:
        tar_traj = bm.trj_from_center(trj, fcc_constants.R_e, fi_opt, dist_opt)
    else:
        tar_traj = bm.trj_shift_turn(trj, fcc_constants.R_e, dist_opt, fi_opt)

    if not np.any(tar_traj) :
        return False
    
    global dist_to_trj_min
    if concentric:
        dist_to_trj_min = min(tar_traj[:, 2] * fcc_constants.R_e) / 1000.0
    else:
        dist_to_trj_min = min(tar_traj[:, 5]) / 1000.0


    """ Limitations on time of interceptor launch calculated before sampling """
    
    t_bout = t_int_lnc - t_delay
    #if t_delay == 0 :
    #    t_int_lnc = 0

    i_dr = 0
    max_idx = len(tar_traj) - 1
    
    if not sls_second:
        if det_range > 0:
            if concentric:
                psi_rad = dist_opt / R_e 
                phi_rad = fi_opt       
                
                valid_detection_found = False
                
                for idx in range(max_idx):
                    m_rad  = tar_traj[idx, 1]
                    m_beta = tar_traj[idx, 2] 
                    
                    cos_theta = cos(psi_rad) * cos(m_beta) + sin(psi_rad) * sin(m_beta) * cos(phi_rad)
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    theta_3d  = acos(cos_theta)

                    slant_range = sqrt(R_e**2 + m_rad**2 - 2 * R_e * m_rad * cos(theta_3d))
                    
                    if slant_range <= det_range:
                        i_dr = idx
                        valid_detection_found = True
                        break
                
                if not valid_detection_found:
                    i_dr = max_idx - 1
            else:

                while i_dr < max_idx and tar_traj[i_dr, 3] > det_range:
                    i_dr += 1
                    if i_dr == max_idx:
                        if debug_print:
                            print("short search exception: missile always beyond detection range")
                        return False
                        
            if i_dr < max_idx or not concentric:
                t_int_lnc = max(t_bout, tar_traj[i_dr, 0]) + t_delay
                
        # --- Radar-Horizont-Kriterium (Over the Horizon Detection OHD) ---
        if t_delay != 0:
            i_ohd = 0
            if concentric:
                # 3D-CONCENTRIC HORIZONT: Spalte 2 hält den geozentrischen Zentralwinkel (alfa).
                # Die Rakete steigt ab alfa = 0.0 sofort über den Horizont des ILP!
                while i_ohd < max_idx and tar_traj[i_ohd, 2] <= 1e-6:
                    i_ohd += 1
                
                if i_ohd < len(tar_traj):
                    i_ohd_x = i_ohd
                    while i_ohd_x < max_idx:
                        if (tar_traj[i_ohd_x, 0] - tar_traj[i_ohd, 0]) >= t_delay:
                            break
                        i_ohd_x += 1
                    t_int_lnc = max(t_int_lnc, tar_traj[i_ohd, 0] + t_delay)
            else:
                # 2D-LEGACY HORIZONT (Nutzt Spalte 4 für beta)
                while i_ohd < len(tar_traj) and tar_traj[i_ohd, 4] <= 0:
                    i_ohd += 1
                    if i_ohd == len(tar_traj):
                        if debug_print:
                            print("short search exception: ohd criteria used, but the whole trajectory is below horizon")
                        return False
                        
                if i_ohd < len(tar_traj):
                    i_ohd_x = i_ohd
                    while i_ohd_x < len(tar_traj) and tar_traj[i_ohd_x, 4] > 0:
                        if (tar_traj[i_ohd_x, 0] - tar_traj[i_ohd, 0]) >= t_delay:
                            break
                        i_ohd_x += 1
                        if i_ohd_x == len(tar_traj):
                            if debug_print:
                                print("short search exception: ohd criteria used, but the trajectory drops below horizon before t_delay")
                            return False                  
                    t_int_lnc = max(t_int_lnc, tar_traj[i_ohd, 0] + t_delay)
        
        else:
            t_int_lnc = fcc_constants.sat_delay
            #t_delay   = 0
    
        if h_discr :    # warhead discrimination height criterium
            if h_discr < 1000 : 
                h_discr *= 1000
            i_hid = len(tar_traj) - 1
            while (tar_traj[i_hid, 1] - R_e) < h_discr :
                if tar_traj[i_hid, 0] < t_int_lnc : 
                    break
                i_hid -= 1
            if debug_print : print("t_int_lnc = {}, t(h_discr) = {}, i_hid = {} len(tar_traj)-1 = {}".format(t_int_lnc, tar_traj[i_hid, 0], i_hid, len(tar_traj)-1))
            t_int_lnc = max(t_int_lnc, tar_traj[i_hid, 0])
    # end if not sls_flag
    
    i_end = len(tar_traj) - 1
    while tar_traj[i_end, 1] - R_e < h_int_min : # from impact backwards find the first point above MIA (min intercept altitude)
        i_end -= 1
        if i_end == 0 :
            if debug_print : 
                print("short search exception: i_mis == 0 (missile trajectory below min acceptable intercept altitude)")
            return False
            
    i_end = min(i_end, len(tar_traj) - 1)
    t_end = tar_traj[i_end, 0]
    #print("i_end=", i_end, "t_end=", t_end)

    if t_end <= t_int_lnc :
        if debug_print : 
            print("short search exception: interception not possible due to too short missile flight time")
        return False

#    beta_n = 360
    
    """ Sampling rate for missile trajectory resampling, must be the same as int_table sampling rate $$$"""
    #beta_step = 0.025 #0.25

    # dist_opt is the metric distance between ILP and MLP (in meters)
    # fi_opt is the flight path plane aspect angle (phi)
    # print("dist_opt: ", dist_opt)
    # print("fi_opt: ", fi_opt)
    # dist_opt is ALWAYS in metres; fi_opt is ALWAYS in radians.
    # Convert once here; never divide/multiply again before passing to s_func.
    psi_rad_val = dist_opt / fcc_constants.R_e   # arc angle ILP→MLP [rad]
    psi_d       = degrees(psi_rad_val)            # degrees – for mis_traj_sample and debug only
    phi_d       = degrees(fi_opt)                 # degrees – for mis_traj_sample only

    if concentric:
        mis_traj = tb_con.mis_traj_sample(trj, beta_step=fcc_constants.beta_step, psi_deg=psi_d, phi_deg=phi_d, concentric=True) 
                
    else:
        mis_traj = tb.mis_traj_sample(tar_traj)

    # =================================================================
    # SICHERHEITS-CHECK: Leere Trajektorien abfangen (Verhindert ValueError)
    # =================================================================
    if mis_traj is None or len(mis_traj) == 0:
        if debug_print:
            print("[DEBUG CONCENTRIC] Early Exit: mis_traj ist leer für diese Geometrie.")
        return False # Kein Abfangpunkt in dieser Footprint-Schrittweite möglich

    # Erst wenn Punkte existieren, darf mathematisch ausgewertet werden:
    m_beta_max = max(mis_traj[:, 2])
    i_m_beta_max = np.argmax(mis_traj[:, 2])


 
    """ until 221116
    i1_m_beg = 0
    while mis_traj[i1_m_beg, 0] < t_int_lnc : # find number of the lowest (earliest) possible missile beta for interceptor launch
        i1_m_beg += 1
    i1_m_beg -= 1 # one before to compensate for sampling interval
#    print("t_int_lnc={}, i1_m_beg = {}, mis_traj[i1_m_beg, 0]={}".format(t_int_lnc, i1_m_beg, mis_traj[i1_m_beg]))
    """
    
    """ from 221117 """ 
    i1m_a = 0                    # find number of the lowest (earliest) possible missile beta for interceptor launch
    i1m_b = len(mis_traj) - 1
    if mis_traj[i1m_b, 0] <= t_int_lnc : # trajectory ends before t_int_lnc
        if debug_print : print("i1m_b = {}, mis_traj[i1m_b, 0] = {} <= t_int_lnc = {}".format(i1m_b, mis_traj[i1m_b, 0], t_int_lnc))
        return False
    else :
        while (i1m_b - i1m_a) > 1 :
            i1m_x = int(ceil(i1m_a + i1m_b) / 2)
            f1m_x = mis_traj[i1m_x, 0] - t_int_lnc
            if f1m_x > 0 :
                i1m_b = i1m_x
            elif f1m_x < 0 :
                i1m_a = i1m_x
            else :
                i1m_a = i1m_x
                break
        i1_m_beg = i1m_a
    
    othi_r = -9
    if oth_intercept > -999 : # for endo-atm if intercept has to be above start horizon by oth_intercept degrees
        othi_r = radians(oth_intercept)
        
    i2_m_beg = 0 # find the number of the earliest missile beta equal to the lowest interceptor beta
    #intemp = int_table[0]    
    beta_zero = max(othi_r, int_table[0][0, 3])
    
    if concentric and beta_zero < 0.5:
        beta_zero_rad = np.radians(1.0)  # Reicht aus, um den 0.03° Randfehler zu überspringen!
    else:
        beta_zero_rad = np.radians(beta_zero) if beta_zero > 1.5 else beta_zero

    if not concentric:
        while mis_traj[i2_m_beg, 2] < beta_zero_rad:
            if i2_m_beg == len(mis_traj) - 1:
                if debug_print: 
                    print("False: all missile trajectory below lowest interceptor beta")
                return False
            i2_m_beg += 1
    else:
        # PHYSIKALISCHE SUCHE FÜR CONCENTRIC:
        # Wir suchen chronologisch den ersten echten Punkt, an dem die Rakete über den Horizont kommt.
        i2_m_beg = 0
        found_valid_beta = False
        
        for idx in range(len(mis_traj)):
            if mis_traj[idx, 2] >= beta_zero_rad:
                # Absicherung: Wir überspringen den fehlerhaften allerersten Index am Boden
                if idx > 1:
                    i2_m_beg = idx
                    found_valid_beta = True
                    break
                
        if not found_valid_beta:
            # Fallback auf den mathematischen Scheitelpunkt, falls die Rakete extrem flach bleibt
            max_idx = 0
            max_val = mis_traj[0, 2]
            for idx in range(len(mis_traj)):
                if mis_traj[idx, 2] > max_val:
                    max_val = mis_traj[idx, 2]
                    max_idx = idx
            i2_m_beg = max_idx

    m_beg = max(i1_m_beg, i2_m_beg)
    beta_il = mis_traj[m_beg, 2]
    # Max-Valid-Index bestimmen
    max_valid_index = len(int_table) - 1
    grid_step_rad = np.radians(fcc_constants.beta_step)

    if concentric:
        table_beta_min = int_table[0][0, 3]
        table_beta_max = int_table[-1][0, 3]
        
        beta_array = np.array([int_table[k][0, 3] for k in range(len(int_table))])
        int_beta_il = int(np.argmin(np.abs(beta_array - beta_il)))
        int_beta_il = max(0, min(int_beta_il, max_valid_index))
        
        current_eps = grid_step_rad * 0.51
            
        if beta_il < table_beta_min or beta_il > table_beta_max:
            if debug_print:
                print(f"[DEBUG CONCENTRIC] Kein Footprint: Raketen-Beta ({degrees(beta_il):.2f}°) "
                      f"liegt außerhalb der Abfangtabelle ({degrees(table_beta_min):.2f}° - {degrees(table_beta_max):.2f}°)")
            return False
        
        # print(f"\n[DEBUG CONCENTRIC] Rakete beta_il (deg): {degrees(beta_il):.2f} | Best Match Tabelle (deg): {degrees(int_table[int_beta_il][0, 3]):.2f}")
    else:

        # Originaler Suchalgorithmus für das flache 2D-Modell
        int_beta_il = 0
        current_eps = angle_eps
        
        if (int_table[int_beta_il][0, 3] > m_beta_max):
            if debug_print: 
                print("All of the possibly interceptable missile trajectory is below lowest interceptor beta")
            return False
        
                # Der originale Suchalgorithmus wird NUR im klassischen Modell fertig ausgeführt
        while int_beta_il < max_valid_index and int_table[int_beta_il][0, 3] < (beta_il - current_eps):
            int_beta_il += 1
            if int_beta_il > max_valid_index:
                int_beta_il = max_valid_index
                break

            if not np.any(int_table[int_beta_il]):
                if debug_print:
                    print("!!!", end='')
                if abs(pi/2 - beta_il) < current_eps:
                    int_beta_il = max_valid_index
                    break
                else:
                    m_beg += 1
                    if m_beg >= len(mis_traj): 
                        break
                    beta_il = mis_traj[m_beg, 2]           
                    int_beta_il -= 1

    # Das sichere Zeit- und Höhen-Debug-Print dynamisch ausführen
    try:
        ray = int_table[int_beta_il]
        if ray is not None and len(ray) > 0 and ray[0, 3] != -1.0:
            if concentric:
                t_mis = mis_traj[m_beg, 0]
                idx_closest = int(np.argmin(np.abs(ray[:, 1] - t_mis)))
                int_flight_time = f"{ray[idx_closest, 1]:.1f}"
            else:
                r_mis = mis_traj[m_beg, 1]
                idx_closest = int(np.argmin(np.abs(ray[:, 2] - r_mis)))
                int_flight_time = f"{ray[idx_closest, 1]:.1f}"
        else:
            int_flight_time = "N/A (Leer)"
    except Exception as e:
        int_flight_time = f"Err ({str(e)})"

    # WICHTIG: Das Epsilon für concentric absichern und den Reset-Winkel-Bug verhindern
    if concentric:
        current_eps = grid_step_rad * 0.51
    else:
        current_eps = angle_eps

    # Plausibilitätsprüfung vor der Hauptschleife
    if abs(int_table[int_beta_il][0, 3] - beta_il) > current_eps: 
        if debug_print: 
            print("short search exception: betas not equal. Delta too large.")
        if not concentric:
            return False


    # =================================================================
    # PRE-BUILD beta_array ONCE (outside loop) for O(n log n) instead of O(n²)
    # =================================================================
    i_i_beta = int_beta_il
    hit = False

# Vorbereitung für den Concentric-Modus außerhalb der Zeitschleife
    if concentric:
        # Einmalige Initialisierung der Tabellengrenzen – NUR HIER
        psi_rad = psi_rad_val          # already computed as dist_opt / R_e [rad]
        phi_rad = fi_opt               # fi_opt is always in radians
        table_beta_min = int_table[0][0, 3]
        table_beta_max = int_table[-1][0, 3]
        grid_step_rad  = radians(fcc_constants.beta_step)
        max_seg_idx    = len(int_table) - 1

    hit = False

    for i_m_beta in range(m_beg, len(mis_traj)):
        m_beta = mis_traj[i_m_beta, 2]
        m_rad  = mis_traj[i_m_beta, 1]   # geozentrischer Radius [m]
        m_time = mis_traj[i_m_beta, 0]
        x_rad  = mis_traj[i_m_beta, 3]
        y_rad  = mis_traj[i_m_beta, 4]

        if concentric:
            # 1. Sphärischer Kosinussatz
            cos_theta = cos(psi_rad) * cos(m_beta) + sin(psi_rad) * sin(m_beta) * cos(phi_rad)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta_3d  = acos(cos_theta)

            # 2. Nur innerhalb der Tabellengrenzen weitermachen
            if theta_3d < table_beta_min - 1e-6 or theta_3d > table_beta_max + 1e-6:
                if False and i_m_beta % 10 == 0:
                    print(f"[CONC-DBG] i={i_m_beta}, xi={degrees(m_beta):.2f}°, "
                          f"theta={degrees(theta_3d):.2f}° -> out of table bounds")
                continue

            # 3. Passenden Strahl finden
            theta_grid = np.array([ray[0, 3] for ray in int_table if len(ray) > 0])
            seg_idx = np.searchsorted(theta_grid, theta_3d, side='right') - 1
            seg_idx = max(0, min(seg_idx, len(int_table) - 1))
            ray = int_table[seg_idx]

            if ray is None or len(ray) == 0 or ray[0, 3] == -1.0:
                if False:
                    print(f"[CONC-DBG] i={i_m_beta} -> ray invalid/empty at seg_idx={seg_idx}")
                continue

            # --- Debug: Zeige den gefundenen Strahl ---
            if False and i_m_beta % 10 == 0:
                ray_alt_min = (ray[:,2].min() - fcc_constants.R_e)/1000.0
                ray_alt_max = (ray[:,2].max() - fcc_constants.R_e)/1000.0
                print(f"[CONC-DBG] i={i_m_beta}, xi={degrees(m_beta):.2f}°, "
                      f"theta={degrees(theta_3d):.2f}° -> ray alt=[{ray_alt_min:.0f}-{ray_alt_max:.0f}] km")

            if False:
                print(f"    -> using seg_idx={seg_idx}, ray theta={degrees(ray[0,3]):.2f}°")

            # 4. Höhen‑ und Reichweiten‑Gates
            int_altitude = m_rad - fcc_constants.R_e
            if int_altitude < h_int_min:
                if debug_hits:
                    print(f"    -> altitude {int_altitude/1000:.1f}km < MIA {h_int_min/1000:.1f}km")
                continue
            if maxia and m_rad > (fcc_constants.R_e + maxia):
                if debug_hits:
                    print(f"    -> above maxia ({maxia/1000:.1f}km)")
                continue
            if op_range and m_rad > op_range:
                if debug_hits:
                    print(f"    -> beyond op_range")
                continue

            # 5. Binäre Suche nach Radius im Strahl
            ray_r = ray[:, 2]
            if m_rad < ray_r[0] or m_rad > ray_r[-1]:
                if False:
                    print(f"    -> m_rad {m_rad/1000:.1f}km outside ray range "
                        f"[{ray_r[0]/1000:.1f}-{ray_r[-1]/1000:.1f}]km")
                continue

            i_a, i_b = 0, len(ray) - 1
            while (i_b - i_a) > 1:
                i_x = (i_a + i_b) // 2
                if ray_r[i_x] > m_rad:
                    i_b = i_x
                else:
                    i_a = i_x

            r1_int, r2_int = ray_r[i_a], ray_r[i_b]
            denom_r = r2_int - r1_int
            factor_r = 0.0 if abs(denom_r) < 1e-12 else (m_rad - r1_int) / denom_r

            # 6. Interzeptor‑Flugzeit interpolieren
            int_flight_time = ray[i_a, 1] + factor_r * (ray[i_b, 1] - ray[i_a, 1])

            # 7. Zeitfenster prüfen
            t_available = m_time - t_int_lnc
            if debug_hits and i_m_beta % 10 == 0:
                print(f" int_flight={int_flight_time:.1f}s, t_avail={t_available:.1f}s", end="")
                if int_flight_time > t_available:
                    print(" -> FAIL: too late")
                else:
                    print(" -> HIT possible?")

            if int_flight_time > t_available:
                continue
            
            if concentric and mrange is None:
                    f_alfa = trj[-1, 2] 
                    mrange = f_alfa * R_e  

            # 8. Treffer – kartesische Koordinaten berechnen
            hit_x_km, hit_z_km = omegashift_2_xy(phi_rad, dist_opt, mrange)

            if False:
                print(f"🎯 HIT at i={i_m_beta}, theta={degrees(theta_3d):.2f}°, "
                    f"alt={int_altitude/1000:.1f}km, t_miss={m_time:.1f}s, "
                    f"t_int={int_flight_time:.1f}s, X={hit_x_m:.2f}km, Z={hit_z_m:.2f}km")

            return np.array([m_time, m_rad, m_beta, hit_x_km, hit_z_km])

        # ----------------------------------------------------------------
        # CLASSIC 2-D BRANCH (100% kompatibel isoliert)
        # ----------------------------------------------------------------
        else:
            if not np.any(int_table[i_i_beta]):
                continue

            if abs(m_beta - int_table[i_i_beta][0, 3]) > current_eps:
                print("betas not equal exception: m={}, i={}".format(mis_traj[i_m_beta, 2], int_table[i_i_beta][0, 3]))
                print("i_m_beta={}, i_i_beta={}".format(i_m_beta, i_i_beta))
                hit = False
                break

            y_rad_flat   = y_rad + fcc_constants.R_e  # Verwende Konstante aus Namespace
            int_altitude_classic = sqrt(x_rad*x_rad + y_rad_flat*y_rad_flat) - fcc_constants.R_e
            is_valid_point = (
                ((not op_range) or (m_rad <= op_range)) and
                (m_beta > othi_r) and
                ((not maxia) or (int_altitude_classic <= maxia))
            )

            if is_valid_point:
                i_a = 0
                i_b = len(int_table[i_i_beta]) - 1
                if i_b > 0:
                    if int_table[i_i_beta][i_b, 2] > m_rad:
                        while (i_b - i_a) > 1:
                            i_x = int(ceil(i_a + i_b) / 2)
                            f_x = int_table[i_i_beta][i_x, 2] - m_rad
                            if f_x > 0:   i_b = i_x
                            elif f_x < 0: i_a = i_x
                            else:
                                hit = int_table[i_i_beta][i_b], mis_traj[i_m_beta], int_table[i_i_beta][i_b], int_table[i_i_beta][i_b]
                                break
                        i_hit = i_b
                        if not hit and (int_table[i_i_beta][i_hit - 1, 2] < m_rad):
                            r1 = int_table[i_i_beta][i_hit - 1, 2]
                            r2 = int_table[i_i_beta][i_hit,     2]
                            t1 = int_table[i_i_beta][i_hit - 1, 1]
                            t2 = int_table[i_i_beta][i_hit,     1]
                            t_hit   = t1 + (t2 - t1) * (m_rad - r1) / (r2 - r1)
                            psi1    = int_table[i_i_beta][i_hit - 1, 0]
                            psi2    = int_table[i_i_beta][i_hit,     0]
                            psi_hit = psi1 + (psi2 - psi1) * (m_rad - r1) / (r2 - r1)
                            hit = [psi_hit, t_hit, m_rad, int_table[i_i_beta][1, 3]], mis_traj[i_m_beta], int_table[i_i_beta][i_hit - 1], int_table[i_i_beta][i_hit]
                        elif not hit and int_table[i_i_beta][0, 2] == m_rad:
                            hit = int_table[i_i_beta][0], mis_traj[i_m_beta], int_table[i_i_beta][0], int_table[i_i_beta][0]
                    elif int_table[i_i_beta][i_b, 2] == m_rad:
                        hit = int_table[i_i_beta][i_b], mis_traj[i_m_beta], int_table[i_i_beta][i_b], int_table[i_i_beta][i_b]
                elif int_table[i_i_beta][0, 2] == m_rad:
                    hit = int_table[i_i_beta][0], mis_traj[i_m_beta], int_table[i_i_beta][0], int_table[i_i_beta][0]

            if hit:
                hit_i, hit_m, _, _ = hit
                if hit_i[3] >= othi_r:
                    if (hit_i[1] <= (hit_m[0] - t_int_lnc)) and (hit_m[0] < t_end):
                        if debug_hits:
                            print("hit i_angle={:f}, m_angle={:f}".format(degrees(hit_i[3]), degrees(hit_m[2])))
                    else:
                        hit = False
                else:
                    hit = False

            if hit:
                # Echter Treffer im Classic-Zweig -> Abbruch der Bahn-Schleife
                break

            # Inkrementierung/Dekrementierung des Classic-Höhenwinkel-Index
            if i_m_beta < (len(mis_traj) - 1):
                m_beta_val = mis_traj[i_m_beta, 2]
                if m_beta_val < m_beta_max:
                    if   i_m_beta < i_m_beta_max: i_i_beta += 1
                    elif i_m_beta > i_m_beta_max: i_i_beta -= 1
                else:
                    if mis_traj[i_m_beta + 1, 2] < m_beta_val: i_i_beta -= 1

            if i_i_beta == -1:
                hit = False
                break

    # Schleifenende (for i_m_beta...)
    et = time.time()
    elapsed_time = et - st
    
    if hit and debug_print:
        print("Full trajectory generation and search time = {:.3f}s, hit=True".format(elapsed_time))
    
    # =================================================================
    # FINALE PIPELINE-KONFORME RÜCKGABE
    # =================================================================
    if hit:
        return hit
    else:
        return False

""" End of short_search """

def short_search2(trj, int_table, h_int_min, maxia, t_int_lnc, omega, shift, op_range, det_range, 
                  t_delay=0, h_discr=0, debug_hits=False, concentric=False, mrange = None) :
    if concentric:
        f_alfa = trj[-1, 2] 
        mrange = f_alfa * R_e
    return short_search(trj, int_table, h_int_min, maxia, t_int_lnc, omega, shift, op_range, det_range, 
                        t_delay, h_discr, True, debug_hits, False, concentric, mrange)

def sls_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, 
               t_delay=0, h_discr=0, mode2=False, debug_hits=False, 
               concentric=False, mrange=None):
    """
    Finds interception points for shoot-look-shoot mode. First, runs short_search, then takes missile time at the first interception point
    and runs short_search again passing missile time instead of t_int_lnc. Returns both interception data points to check if this works.
    Both will not be needed when function is confirmed to be good.
    This is a valid procedure because short_search finds the earliest interception point, so if there is no second hit for the first
    found first hit, there won't be any for later ones.
    
    ----------
    trj : array of [time, R, alfa]
        missile trajectory in polar coords, center in Earth center
    int_table : array of trajectories
        sampled interception table (array)
    h_int_min : 
        min acceptable height of interception
    t_int_lnc :
        earliest possible interceptor launch time counted from missile launch (missile burn time + interceptor launch delay)
    fi_opt : 
        direction to impact point from ILP to MLP line, >>>radians<<<
    dist_opt : 
        distance to impact point from ILP
    t_delay : 
        launch delay after over horizon detection ('0' means criterium not used)
    h_discr :
        warhead discrimination height ('0' means criterium not used)
    mode2 :
        mode2 search (via shift and omega instead of fi and distance)

    Returns
    -------
    Coords and time of hit if found, false if not

    """
    if debug_hits:
        print(f"\n[SLS-DEBUG] === Starting SLS Search (Concentric={concentric}) ===")
        print(f"[SLS-DEBUG] Parameters: dist_opt={dist_opt:.2f}, t_int_lnc={t_int_lnc:.2f}, t_delay={t_delay}")

    # --- FIRST SHOT ---
    hit_1 = short_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, t_delay, h_discr, mode2, debug_hits, False, concentric, mrange)
    
    hit_1_valid = False
    if concentric:
        if isinstance(hit_1, np.ndarray) and hit_1.size > 0:
            hit_1_valid = True
    else:
        if hit_1:  
            hit_1_valid = True

    if debug_hits:
        print(f"[SLS-DEBUG] HIT 1 Raw Output: {type(hit_1)} (is_valid={hit_1_valid})")
        if hit_1_valid:
            if concentric:
                print(f"[SLS-DEBUG] HIT 1 Data (Array): t_miss={hit_1[0]:.2f}s, R={hit_1[1]:.2f}m, beta={hit_1[2]:.4f}rad, X={hit_1[3]:.2f}m, Z={hit_1[4]:.2f}m")
            else:
                # Annahme: hit_1 ist ein Tupel, wobei das innere Element 1 die Trajektorienwerte enthält
                print(f"[SLS-DEBUG] HIT 1 Data (Tuple): {hit_1}")
        else:
            print(f"[SLS-DEBUG] HIT 1 failed. Aborting SLS sequence.")

    # --- SECOND SHOT ---
    if hit_1_valid:
        # KORREKTUR: Der zweite Schuss verzögert sich um t_delay relativ zur STARTZEIT des ersten Schusses (t_int_lnc)
        # Dadurch schießt der zweite Interceptor rechtzeitig, während die Rakete noch im Flug ist.
        t_int_lnc_second = hit_1[1][0] if not concentric else hit_1[0]
            
        if debug_hits:
            print(f"[SLS-DEBUG] -> HIT 1 valid. Launching Second Shot at t={t_int_lnc_second:.2f}s (t_delay={t_delay}s)...")

        # Führe short_search für den zweiten Schuss mit der neuen Startzeit aus
        hit_2 = short_search(trj, int_table, h_int_min, maxia, t_int_lnc_second, fi_opt, dist_opt, op_range, det_range, t_delay, h_discr, mode2, debug_hits, True, concentric, mrange)
    
        hit_2_valid = False
        if concentric:
            if isinstance(hit_2, np.ndarray) and hit_2.size > 0:
                hit_2_valid = True
        else:
            if hit_2:
                hit_2_valid = True

        if debug_hits:
            print(f"[SLS-DEBUG] -> HIT 2 Raw Output: {type(hit_2)} (is_valid={hit_2_valid})")
            if hit_2_valid:
                if concentric:
                    print(f"[SLS-DEBUG] 🎉 SUCCESS: HIT 2 Data (Array): t_miss={hit_2[0]:.2f}s, R={hit_2[1]:.2f}m, beta={hit_2[2]:.4f}rad, X={hit_2[3]:.2f}m, Z={hit_2[4]:.2f}m")
                else:
                    print(f"[SLS-DEBUG] 🎉 SUCCESS: HIT 2 Data (Tuple): {hit_2}")
            else:
                print(f"[SLS-DEBUG] ❌ HIT 2 failed. The second interceptor could not catch the target (Too late / Out of bounds).")

        if hit_2_valid:
            return hit_2

    return False





def sls_search2(trj, int_table, h_int_min, maxia, t_int_lnc, omega, shift, op_range, det_range, 
                t_delay=0, h_discr=0, debug_hits=False, concentric=False) :
    
    if concentric:
        f_alfa = trj[-1, 2]
        mrange = f_alfa * R_e
    else:
        mrange = None

    return sls_search(
        trj, int_table, h_int_min, maxia, t_int_lnc, 
        fi_opt=omega, dist_opt=shift, op_range=op_range, det_range=det_range, 
        t_delay=t_delay, h_discr=h_discr, mode2=True, 
        debug_hits=debug_hits, concentric=concentric, mrange=mrange
    )


                    
def angle_dist_tab2(s_func,
                    t_trj,
                    t_itable,
                    h_min, 
                    maxia,
                    t_lnc,
                    op_range,
                    det_range,
                    t_delay,
                    h_discr,
                    t_ang_beg,
                    t_ang_end,
                    t_angle_step,
                    dist_beg,
                    num_dist,
                    plot_hit_charts = False,
                    hit_chart_angle = 180,
                    m_type = 0,
                    i_type = 0,
                    t_itable_con = None
                    ) :
    """
    !!! Mode1 routine, for mode2 see probing2 below !!!
    Build hit/no hit table for a range of footprint angles (t_ang_beg to f_ang_enf) and distances (dist_beg to dist_max)

    Returns
    -------
    fp_sector : array or list of arrays
        Coordinates of all found footprint spots (x, y) for standard (and concentric if active)
    """
    is_compare_mode = t_itable_con is not None

    dist_beg_m = dist_beg * 1000
    dist_max = t_trj[len(t_trj) - 1, 2] * R_e - 1000 # missile range minus 1 km

    def evaluate_probing_grid(active_table, is_concentric, label_str):
        beta_min = active_table[0][0, 3] - radians(beta_step)
        int_max_range = -2 * beta_min * R_e
        print("\n[{}] beta_min={:.2f}, int_max_range={:.1f} km".format(label_str, degrees(beta_min), int_max_range/1000))

        search_range = min(dist_limit, dist_max + int_max_range)
        dist_step = (search_range - dist_beg_m) / num_dist
        fp_sector_local = np.empty([0, 2])
        
        f_dist = dist_beg_m
        stadt = time.process_time()
        
        while f_dist <= search_range + .1 :
            if f_dist <= dist_max + .1 :
                f_ang = t_ang_end
            else :
                f_ang = min(90, t_ang_end) # to reduce search when int_max_range > dist_max

            enadt = time.process_time()
            if enadt - stadt >= out_time_freq :
                print("\r[{}] Probing distance {:5.0f} km...   ".format(label_str, f_dist/1000), end='')
                stadt = enadt

            v_angle_step = t_angle_step
            if f_dist > dist_beg_m :
                v_angle_step *= (search_range - dist_beg_m) / 2 / (f_dist - dist_beg_m)
            
            while f_ang >= t_ang_beg :
                f_ang_r = radians(f_ang)
                
                hit_ss = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_ang_r, f_dist, 
                                op_range, det_range, t_delay, h_discr, concentric=is_concentric)

                if hit_ss :
                    if abs(f_ang) <= angle_eps :
                        x1 = 0
                        y1 = f_dist / 1000
                    elif abs(f_ang - 180) <= angle_eps :
                        x1 = 0
                        y1 = -f_dist / 1000
                    else :
                        f_delta = f_dist / R_e
                        f_delta_x = asin(sin(f_ang_r) * sin(f_delta))
                        f_delta_y = asin(tan(f_delta_x) / tan(f_ang_r))
                        if f_delta > pi/2 :
                            if f_ang_r >= pi/2 :
                                f_delta_y = - pi - f_delta_y
                            else : 
                                f_delta_y = pi - f_delta_y
                            
                    x1 = - f_delta_x * R_e / 1000
                    y1 =   f_delta_y * R_e / 1000

                    fp_sector_local = np.append(fp_sector_local, [[x1, y1]], axis=0) # defended spots only

                    if plot_hit_charts and i_type : 
                        if f_ang == hit_chart_angle :
                            fp.plot_hit(hit_ss, t_trj, m_type, i_type, f_ang, f_dist, int_max_range/1000, op_range, det_range, t_lnc-t_delay, h_min/1000, h_discr/1000)

                if f_dist > 0 :
                    f_ang -= v_angle_step
                else :
                    break
                    
            f_dist += dist_step
            
        print("\n[{}] Probing sector processing complete.".format(label_str))
        return fp_sector_local

    if is_compare_mode:
        fp_sector_std = evaluate_probing_grid(t_itable, False, "Standard")
        fp_sector_con = evaluate_probing_grid(t_itable_con, True, "Concentric")
        return [fp_sector_std, fp_sector_con]
    else:
        fp_sector_std = evaluate_probing_grid(t_itable, False, "Standard")
        return fp_sector_std

def probing2(s_func,
             t_trj,
             t_itable,
             h_min, 
             maxia,
             t_lnc,
             op_range,
             det_range,
             t_delay,
             h_discr,
             t_acc,
             t_ang_beg, # irrelevant
             t_ang_end, # irrelevant
             t_angle_step,
             dist_beg,  # not used here
             num_dist,
             plot_hit_charts = False,
             hit_chart_angle = 180,
             m_type = 0,
             i_type = 0,
             t_itable_con = None
             ) :

    """
    Build hit/no hit table for mode2 using a range of footprint angles and distances (shifts).
    
    Returns
    -------
    fp_sector : array or list of arrays
        Coordinates of all found footprint spots (x, y) for standard (and concentric if active)
    """
    is_compare_mode = t_itable_con is not None

    f_alfa = t_trj[len(t_trj) - 1, 2]
    mrange = f_alfa * R_e
    
    t_ang_beg_local = 0
    t_ang_end_local = 90 
    shift_end = shift_min 

    def evaluate_probing_grid_m2(active_table, is_concentric, label_str):

        def is_hit(res):
            """Return True if res is a valid hit for the current mode."""
            if is_concentric:
                return isinstance(res, np.ndarray) and res.size > 0
            return bool(res)

        beta_min = active_table - radians(beta_step)
        int_max_range = -2 * beta_min * R_e
        print("\n[{}] beta_min={:.2f}, int_max_range={:.1f} km".format(label_str, degrees(beta_min), int_max_range/1000))

        ilp_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, mrange, op_range, det_range, t_delay, h_discr)
        if ilp_ok :
            f_dist = min(int_max_range / 8 + mrange, dist_limit)
            f_dist0 = mrange
            f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_dist, op_range, det_range, t_delay, h_discr)
            
            while is_hit(f_ok) :             
                f_y_keep = f_ok
                f_dist0 = f_dist
                if f_dist >= dist_limit : break
                f_dist = f_dist * 2 - mrange
                f_dist = min(f_dist, dist_limit)
                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_dist, op_range, det_range, t_delay, h_discr)
            
            do_search = not f_ok
            f_a = f_dist0
            f_b = f_dist
            while do_search :
                f_x = f_a + (f_b - f_a) / 2
                f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr)
                if is_hit(f_xy) :
                    f_a = f_x
                    f_y_keep = f_xy
                else :
                    f_b = f_x
                if (abs(f_b - f_a) < f_b * t_acc) or (abs(f_b - f_a) < 10) : break

            if plot_hit_charts and i_type :
                hit_target = f_ok if abs(f_a - dist_limit) < eps else f_y_keep
                if hit_target:
                    fp.plot_hit(hit_target, t_trj, m_type, i_type, 0, f_a - mrange, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
        else : 
            f_a = mrange
            
        shift_beg = f_a
        dist_step = (shift_end - shift_beg) / num_dist
        fp_sector_local = np.empty()
        
        f_dist = shift_beg
        stm2 = time.process_time()
        
        while f_dist >= shift_end - .1 :
            enm2 = time.process_time()
            if enm2 - stm2 >= out_time_freq :
                print("\r[{}] Probing distance {:6.0f} km...   ".format(label_str, (f_dist-mrange)/1000), end='')
                stm2 = enm2
                
            f_ang = t_ang_beg_local
            while f_ang <= t_ang_end_local :
                f_ang_r = radians(f_ang)
                
                hit_ss = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_ang_r, f_dist, 
                                op_range, det_range, t_delay, h_discr, concentric=is_concentric)
                    
                if hit_ss :
                    if abs(f_ang) <= angle_eps :
                        x1 = 0
                        y1 = (f_dist - mrange) / 1000
                    else :
                        f_delta_x = asin(sin(f_ang_r) * sin(f_alfa))
                        if f_alfa > pi/2 :
                            f_gamma_y = pi - asin(tan(f_delta_x) / tan(f_ang_r))
                        else :
                            f_gamma_y = asin(tan(f_delta_x) / tan(f_ang_r))            
                        x1 = - f_delta_x * R_e / 1000
                        y1 = (f_dist - f_gamma_y * R_e) / 1000
                    
                    fp_sector_local = np.append(fp_sector_local, [[x1, y1]], axis=0)
                    
                    if plot_hit_charts and i_type : 
                        if f_ang == hit_chart_angle :
                            fp.plot_hit2(hit_ss, t_trj, m_type, i_type, f_ang, f_dist, int_max_range/1000, op_range, det_range, t_lnc-t_delay, h_min/1000, h_discr/1000)
                
                f_ang += t_angle_step
                
            f_dist += dist_step
            
        print("\n[{}] Probing sector processing complete.".format(label_str))
        return fp_sector_local

    if is_compare_mode:
        fp_sector_std = evaluate_probing_grid_m2(t_itable, False, "Standard")
        fp_sector_con = evaluate_probing_grid_m2(t_itable_con, True, "Concentric")
        return [fp_sector_std, fp_sector_con]
    else:
        fp_sector_std = evaluate_probing_grid_m2(t_itable, False, "Standard")
        return fp_sector_std


""" End of probing2 """


def omegashift_2_xy_v0(omega, shift, mrange) :
    """
    Converts shift, omega and mrange to x and y on the surface of the Earth sphere.
    Calculation uses the Spherical Law of Cosines and Napier's rules.
    English comments version.

    Parameters
    ----------
    omega : float (radians)
        Dihedral angle between the trajectory plane and the ILP-MLP plane.        
    shift : float (meters)
        ILP-MLP distance on the spherical Earth surface.
    mrange : float (meters)
        Missile range on the spherical Earth surface.

    Returns
    -------
    x, y : float (km)
        Coordinates centered at the ILP, projected on the spherical Earth surface.
    """
    f_alfa = mrange / R_e
    
    if abs(omega) <= angle_eps :
        x = 0
        y = (shift - mrange) / 1000
    else :
        # Compute lateral displacement angle delta_x using spherical right triangles
        f_delta_x = asin(sin(omega) * sin(f_alfa))
        
        # Compute longitudinal angle gamma_y along the great circle arc
        cos_gamma_y = cos(f_alfa) / cos(f_delta_x)
        
        # CRITICAL PROTECTION: Clip floating-point noise to prevent acos domain errors
        cos_gamma_y = max(-1.0, min(1.0, cos_gamma_y))
        
        f_gamma_y = acos(cos_gamma_y)
        
        # Map angular arc lengths directly to metric surface distances in kilometers
        x = - f_delta_x * R_e / 1000
        y = (shift - f_gamma_y * R_e) / 1000
    
    return x, y

""" End of omegashift_2_xy_v0 """


def omegashift_2_xy(omega, shift, mrange):
    """
    Converts shift, omega and mrange to fi and dist on the surface of the Earth sphere,
    then unrolls them onto a local Cartesian tangent plane.
    Calculation uses the Spherical Law of Cosines.

    Parameters
    ----------
    omega : float (radians)
        Dihedral angle between the trajectory plane and the ILP-MLP plane.        
    shift : float (meters)
        ILP-MLP distance on the spherical Earth surface.
    mrange : float (meters)
        Missile range on the spherical Earth surface.

    Returns
    -------
    x, y : float (km)
        Coordinates centered at the ILP, unfolded onto a localized tangent plane.
    """
    f_alfa = mrange / R_e
    f_gamma = shift / R_e
    
    if abs(omega) <= angle_eps :
        fi = 0
        dist = (shift - mrange) / 1000
    else :
        # Calculate the true spherical distance angle (delta) from ILP to target
        cos_delta = cos(f_gamma) * cos(f_alfa) + sin(f_gamma) * sin(f_alfa) * cos(omega)
        cos_delta = max(-1.0, min(1.0, cos_delta)) # Numerical safety check
        f_delta = acos(cos_delta)
        
        # Convert angular arc length to actual surface distance in kilometers
        dist = f_delta * R_e / 1000
        
        # Calculate true azimuth angle (fi) out of the ILP station using the side-cosines rule
        num_cos_fi = cos(f_alfa) - cos(f_gamma) * cos(f_delta)
        den_sin_fi = sin(f_gamma) * sin(f_delta)
        
        if abs(den_sin_fi) > 1e-12:
            cos_fi = num_cos_fi / den_sin_fi
            cos_fi = max(-1.0, min(1.0, cos_fi)) # Numerical safety check
            fi = acos(cos_fi)
        else:
            fi = 0.0

    # Unfold polar spherical components into the plane tracking coordinate matrix
    x = -dist * sin(fi)
    y =  dist * cos(fi)
            
    return x, y

""" End of omegashift_2_xy """

def footprint_mode2(s_func, # s_func muss entweder short_search2 oder sls_search2 sein
                    t_trj,
                    t_itable,
                    h_min, 
                    maxia,
                    t_lnc,
                    angle_step,
                    op_range,
                    det_range,
                    t_delay,
                    h_discr,
                    t_acc,
                    num_dist,
                    plot_hit_charts = False,
                    hit_chart_angle = 180,
                    m_type = 0,
                    i_type = 0,
                    t_itable_con = None
                    ) :

    # Dynamische Gitter- und Genauigkeitsanpassung für den Concentric-Modus
    if t_itable_con is not None:
        num_dist_internal = max(num_dist * 4, 60)
        t_acc_internal = max(t_acc, 0.005)
    else:
        num_dist_internal = num_dist
        t_acc_internal = t_acc

    fp_calc_param = 3
    stop_par = 10
    is_compare_mode = t_itable_con is not None

    def fp_part_calc_v3(fp_front, fp_back, num_steps=num_dist_internal, concentric=False) :
        """
        v3 -- Berechnet die Footprint-Grenzpunkte (omega) für jeden Shift-Schritt
        sowie alle Zwischenpunkte mit einer gleichmäßigen Bogen- und Winkelschrittweite.
        Garantierte raumstabile Uniform Accuracy im Concentric-Modus.
        """
        active_table = t_itable_con if concentric else t_itable

        def is_hit(res):
            """Return True if res is a valid hit for the current mode."""
            if concentric:
                return isinstance(res, np.ndarray) and res.size > 0
            return bool(res)

        fp_part = np.array([[0, fp_front, 0, 0, (fp_front - mrange)/1000]])
    
        t_ang_beg = 0
        t_ang_end = pi/2
        stop_ang_par = stop_par / mrange 
        
        shift_beg = fp_front
        shift_end = fp_back
        
        shift_step = (shift_end - shift_beg) / num_steps

        f_shift = shift_beg

        stm2 = time.process_time()
        f_x  = t_ang_beg
        f_x_keep = t_ang_beg
        ang_quant_r = radians(min(angle_step, ang_quant))
        
        max_dist_line = False
        is_first_step = True
        
        while f_shift - shift_end >= 0 : 
            enm2 = time.process_time()
            dum2 = enm2 - stm2
            if dum2 >= out_time_freq :
                print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                stm2 = enm2
                
            if concentric and is_first_step:
                f_x = max(t_ang_beg, f_x)

            p_dist_input = f_shift

            f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, p_dist_input, op_range, det_range, t_delay, h_discr, concentric=concentric)
            
            """if concentric:
                print(f"[FP-DEBUG] shift={(f_shift/1000 if concentric else f_shift/1000):.1f} km, "
                    f"omega={degrees(f_x):.1f}° -> f_ok is {type(f_ok)}")
                if isinstance(f_ok, np.ndarray):
                    print(f"hit coords: X={f_ok[3]:.2f} km, Z={f_ok[4]:.2f} km")
                else:
                    print("no hit")"""

            if concentric and is_first_step and not is_hit(f_ok):
                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, t_ang_beg, p_dist_input, op_range, det_range, t_delay, h_discr, concentric=concentric)
            
            is_first_step = False 

            if is_hit(f_ok):
                f_a = f_x
                f_b = t_ang_end

                while True :
                    f_x = f_a + (f_b - f_a) / 2
                    
                    f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, p_dist_input, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    
                    if is_hit(f_xy) :
                        f_a = f_x
                    else :
                        f_b = f_x
                    if (abs(f_b - f_a) < f_b  * t_acc_internal) or (abs(f_b - f_a) < stop_ang_par) :
                        break
                        
                fill_line = True
                if f_b == t_ang_end :
                    if max_dist_line : 
                        fill_line = False
                        f_shift_z = f_shift - shift_step/2
                        
                        if concentric:
                            if concentric and isinstance(f_ok_local, np.ndarray) and f_ok_local.size >= 5:
                                x = f_ok_local[3] 
                                y = f_ok_local[4]
                            else:
                                continue
                        else:
                            x, y = omegashift_2_xy(f_a, f_shift_z, mrange * mrl_factor2)                
                        
                        fp_part = np.append(fp_part, [[f_a, f_shift_z, 2, x, y]], axis=0)
                    else :
                        max_dist_line = True 
                else :
                    max_dist_line = False

                if fill_line :    
                    f_x_num = round((f_a - f_x_keep) / ang_quant_r)
                    if not f_x_num :
                        f_x_num = 1
                    f_x_step = (f_a - f_x_keep) / f_x_num
                    for i in range(f_x_num) :
                        f_x_i = f_x_keep + (i + 1) * f_x_step
                        if f_shift == shift_beg :
                            i_shift = f_shift
                        else :
                            i_shift = f_shift - shift_step + shift_step * (i + 1) / f_x_num
                        
                        target_shift = f_shift
                        p_dist_local = i_shift
                        
                        if concentric:
                            f_ok_local = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x_i, p_dist_local, op_range, det_range, t_delay, h_discr, concentric=True)
                            
                            if (f_ok_local is not False 
                            and f_ok_local is not None 
                            and hasattr(f_ok_local, "__len__") 
                            and len(f_ok_local) > 4):
                                x = f_ok_local[3]
                                y = f_ok_local[4]
                            else:
                                continue
                        else:
                            x, y = omegashift_2_xy(f_x_i, i_shift, mrange)          
                        
                        if i == f_x_num - 1 :
                            fp_part = np.append(fp_part, [[f_x_i, target_shift, f_b - f_a, x, y]], axis=0) 
                        else :
                            fp_part = np.append(fp_part, [[f_x_i, target_shift if concentric else i_shift, 1, x, y]], axis=0)        

                f_x_keep = f_a
                f_x = f_a
            else : 
                if concentric:
                    f_b = f_x
                    f_a = max(t_ang_beg, f_x_keep - radians(2.0))
                else:
                    f_b = f_x
                    f_a = t_ang_beg

                while True : 
                    f_x = f_a + (f_b - f_a) / 2
                    
                    p_dist_fallback = f_shift
                    f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, p_dist_fallback, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    
                    if is_hit(f_xy) :
                        f_a = f_x
                    else :
                        f_b = f_x
                    if (abs(f_b - f_a) < f_b  * t_acc_internal) or (abs(f_b - f_a) < stop_ang_par) : 
                        break

                f_x_num = round(abs(f_a - f_x_keep) / ang_quant_r)
                if not f_x_num :
                    f_x_num = 1
                f_x_step = (f_a - f_x_keep) / f_x_num
                for i in range(f_x_num) :
                    f_x_i = f_x_keep + (i + 1) * f_x_step
                    i_shift = f_shift - shift_step + shift_step * (i + 1) / f_x_num

                    if concentric:
                        p_dist_local_fb = i_shift
                        f_ok_local = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x_i, p_dist_local_fb, op_range, det_range, t_delay, h_discr, concentric=True)
                        
                        if (f_ok_local is not False 
                            and f_ok_local is not None 
                            and hasattr(f_ok_local, "__len__") 
                            and len(f_ok_local) > 4):
                            x = f_ok_local[3]
                            y = f_ok_local[4]
                        else:
                            continue
                    else:
                        x, y = omegashift_2_xy(f_x_i, i_shift, mrange)
                        
                    if i == f_x_num - 1 :
                        fp_part = np.append(fp_part, [[f_x_i, f_shift, f_b - f_a, x, y]], axis=0) 
                    else :
                        fp_part = np.append(fp_part, [[f_x_i, i_shift, 1, x, y]], axis=0)         

                f_x_keep = f_a
                f_x = f_a

            if concentric:
                f_shift -= abs(shift_step)
            else:
                if f_shift > shift_end :
                    f_shift += shift_step
                    if f_shift < shift_end :
                        f_shift = shift_end
                else :
                    f_shift += shift_step

        pass 
        
        max_dist_line = False
        f_a = t_ang_beg 
        f_shift = shift_end
        if abs(f_shift - shift_min) < shift_min :
            ang_quant_r *= 2
            max_dist_line = True
            
        f_x_num = round(abs(f_a - f_x_keep) / ang_quant_r)
        if not f_x_num :
            f_x_num = 1
        f_x_step = (f_a - f_x_keep) / f_x_num
        
        p_dist_back = shift_end
        target_shift_back = shift_end

        for i in range(0, f_x_num) :
            f_x_i = f_x_keep + (i + 1) * f_x_step
            if max_dist_line :
                f_x_i_half = f_x_keep + (i + 1/2) * f_x_step

                if concentric:
                    f_ok_local = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x_i_half, p_dist_back, op_range, det_range, t_delay, h_discr, concentric=True)
                    if (f_ok_local is not False 
                            and f_ok_local is not None 
                            and hasattr(f_ok_local, "__len__") 
                            and len(f_ok_local) > 4):
                        x = f_ok_local[3]
                        y = f_ok_local[4]
                    else:
                        continue
                else:
                    x, y = omegashift_2_xy(f_x_i_half, shift_end, mrange * mrl_factor2)                
                fp_part = np.append(fp_part, [[f_x_i_half, target_shift_back, 2, x, y]], axis=0)
                
                if concentric:
                    f_ok_local = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x_i, p_dist_back, op_range, det_range, t_delay, h_discr, concentric=True)
                    if (f_ok_local is not False 
                            and f_ok_local is not None 
                            and hasattr(f_ok_local, "__len__") 
                            and len(f_ok_local) > 4):
                        x = f_ok_local[3]
                        y = f_ok_local[4]
                    else:
                        continue
                else:
                    x, y = omegashift_2_xy(f_x_i, shift_end, mrange * mrl_factor1)                
                fp_part = np.append(fp_part, [[f_x_i, target_shift_back, 3, x, y]], axis=0)
            else :
                if concentric:
                    f_ok_local = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x_i, p_dist_back, op_range, det_range, t_delay, h_discr, concentric=True)
                    if (f_ok_local is not False 
                            and f_ok_local is not None 
                            and hasattr(f_ok_local, "__len__") 
                            and len(f_ok_local) > 4):
                        x = f_ok_local[3]
                        y = f_ok_local[4]
                    else:
                        continue
                else:
                    x, y = omegashift_2_xy(f_x_i, shift_end, mrange)                
                fp_part = np.append(fp_part, [[f_x_i, target_shift_back, 1, x, y]], axis=0)

        print("\rProcessing distance complete", end='')

        if len(fp_part) > 0:
            print(f"[FP-DEBUG] fp_part shape: {fp_part.shape}")
            print(f"  first 3 rows:\n{fp_part[:3]}")
            print(f"  last 3 rows:\n{fp_part[-3:]}")

        # Symmetrische Spiegelung und polare Glättung (Rippel-Bereinigung)
        # Symmetrische Spiegelung der berechneten Koordinatensegmente
        len_fp_part = len(fp_part)
        if len_fp_part > 0 :   
            for i_f in range (len_fp_part - 1) :
                fp_part_if = fp_part[len_fp_part - 2 - i_f]
                fp_part = np.append(fp_part, [fp_part_if], axis=0)
                fp_part[len_fp_part + i_f, 0] = - fp_part_if[0]
                fp_part[len_fp_part + i_f, 3] = - fp_part_if[3]
        
            if concentric:
                cx = np.mean(fp_part[:, 3])
                cy = np.mean(fp_part[:, 4])
                
                angles = np.arctan2(fp_part[:, 4] - cy, fp_part[:, 3] - cx)
                angles_deg = np.round(np.degrees(angles), 1) 
                unique_angles = np.unique(angles_deg)
                
                cleaned_fp = []
                for ua in unique_angles:
                    mask = (angles_deg == ua)
                    candidates = fp_part[mask]
                    if len(candidates) > 0:
                        radii = np.sqrt((candidates[:, 3] - cx)**2 + (candidates[:, 4] - cy)**2)
                        cleaned_fp.append(candidates[np.argmax(radii)])
                
                fp_part = np.array(cleaned_fp)
                
                final_angles = np.arctan2(fp_part[:, 4] - cy, fp_part[:, 3] - cx)
                fp_part = fp_part[final_angles.argsort()]
                
                if len(fp_part) > 1 and not np.array_equal(fp_part[0], fp_part[-1]):
                    fp_part = np.vstack((fp_part, fp_part[0:1]))
        else :
            print("\nFootprint not calculated, check input data")
            fp_part = np.zeros((0, 5))

        if len(fp_part) > 0:
            xs = fp_part[:, 3]
            ys = fp_part[:, 4]
            print(f"[FP-DEBUG] X range: {xs.min():.2f} to {xs.max():.2f} km, "
                f"Y range: {ys.min():.2f} to {ys.max():.2f} km")
            
        return fp_part
    """ end of fp_part_calc_v3 """


    def run_footprint_pipeline(current_table, concentric_flag):
        if concentric_flag:
            # Angle-outer table: column 3 holds theta [rad] (central angle from ILP),
            # column 2 holds rv [m] (geocentric radius).
            # int_max_range = max arc length an interceptor can reach from the ILP.
            max_rv = max(
                row[2] for seg in current_table
                if seg is not None and len(seg) > 0 and seg[0, 3] != -1.0
                for row in seg
            )
            int_max_range = (max_rv - R_e)   # height is not range, use theta_max * R_e
            # Better: use the largest theta in the table * R_e
            max_theta = max(
                float(seg[-1, 3]) for seg in current_table
                if seg is not None and len(seg) > 0 and seg[0, 3] != -1.0
            )
            int_max_range = max_theta * R_e
        else:
            beta_min = current_table[0][0, 3] - radians(beta_step)
            int_max_range = -2 * beta_min * R_e
        print("beta_min={:.2f}, int_max_range={:.1f} km (Concentric={})".format(
            degrees(current_table[0][0, 3]), int_max_range / 1000, concentric_flag))

        f_alfa = t_trj[len(t_trj) - 1, 2]
        global mrange
        mrange = f_alfa * R_e
        fp_single = False   
        dist_step = (shift_min - mrange) / num_dist 
        
        r_e_ref = float(R_e) if R_e > 1000000.0 else float(R_e) * 1000.0

        def is_hit(res):
            """Return True if res is a valid hit for the current mode."""
            if concentric_flag:
                return isinstance(res, np.ndarray) and res.size > 0
            return bool(res)

        p_dist_init = mrange
        ilp_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, p_dist_init, op_range, det_range, t_delay, h_discr, concentric=concentric_flag) 

        if not is_hit(ilp_ok):
            print("Interceptor position undefendable. Attempting to find a footprint behind(?) it...")
            shift_beg = min((mrange + int_max_range), dist_limit)
            shift_end = shift_min 
            
            if concentric_flag:
                dist_step = (shift_end - shift_beg) / num_dist_internal
            else:
                dist_step = (shift_end - shift_beg) / num_dist 
            
            stm2 = time.process_time()
            f_shift = shift_beg 
            f_b = f_shift
            
            while f_shift >= shift_end - .1 :
                if f_shift < mrange * 0.1: 
                    break
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= 1 :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                p_dist_loop = f_shift
                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, p_dist_loop, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                
                if False:
                    mode_str = "Concentric" if concentric_flag else "Classic"
                    # Convert the argument to km for display
                    displayed_dist = p_dist_loop / 1000.0
                    hit_str = "HIT" if (isinstance(f_ok, np.ndarray) and f_ok.size > 0) or (not isinstance(f_ok, np.ndarray) and f_ok) else "no hit"
                    print(f"[DEBUG-BEHIND] {mode_str}: shift={f_shift/1000:.1f} km, "
                          f"s_func arg={displayed_dist:.1f} km, result={hit_str}")


                if is_hit(f_ok):
                    f_b = f_shift
                    f_a = f_shift - dist_step
                    break
                
                if concentric_flag:
                    f_shift -= abs(dist_step)
                else:
                    f_shift += dist_step
                    
            if f_b > mrange :
                print("\nFootprint in front of ILP", end='')

            print("\rProcessing distance complete", end='')

            if not is_hit(f_ok):
                mode_str = "Concentric" if concentric_flag else "Classic"
                print(f"\n[DEBUG-PIPELINE] {mode_str}: No hit behind ILP after searching shifts.")
                print(f"  Last f_shift = {f_shift/1000:.1f} km  (dist_step = {dist_step/1000:.1f} km)")
                last_tested_shift = f_shift + dist_step if not concentric_flag else f_shift - dist_step
                print(f"  Last tested shift = {last_tested_shift/1000:.1f} km")
                print(f"  (s_func call used dist_opt = {p_dist_loop/1000 if not concentric_flag else p_dist_loop*R_e/1000:.1f} km)")
                print(f"  Search range: shift_beg = {shift_beg/1000:.1f} km, shift_end = {shift_end/1000:.1f} km")

            if not is_hit(f_ok) :
                print('\rNo footprint found behind undefendable interceptor position.\nIncreasing "Mode 2 Footprint number of steps" might help to find one.')
                fp_part = np.zeros((0, 5))
                return [fp_part, fp_part]
           
        else : 
            # Regulärer Vorwärtslauf für den Front-Edge-Scan
            f_shift = mrange + int_max_range / 8 
            f_shift0 = mrange
            f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
            
            while is_hit(f_ok) :             
                f_shift0 = f_shift
                if f_shift >= int_max_range + mrange :
                    break
                f_shift = f_shift * 2 - mrange
                f_shift = min(f_shift, int_max_range + mrange)
                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)

            f_b = f_shift0
            f_a = f_shift
        
        f_y_keep = f_ok
        while True :
            f_x = f_a + (f_b - f_a) / 2
            f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
            xy_valid = (isinstance(f_xy, np.ndarray) and f_xy.size > 0) if concentric_flag else bool(f_xy)
            if xy_valid:
                f_b = f_x
                f_y_keep = f_xy
            else :
                f_a = f_x
            if (abs(f_a - f_b) < abs(f_b - mrange) * t_acc) or (abs(f_a - f_b) < stop_par) : 
                break

        if f_a <= int_max_range + mrange : 
            if plot_hit_charts and i_type :
                if f_a == dist_limit : 
                    fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                else :
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
        else :
            print("\nProbing2 exception (b): f_a = {:.2f} > int_max_range = {:.2f}".format(f_a/1000, int_max_range/1000))
            return False
            
        fp_front1 = f_b
        print("\rFront edge of the footprint found, location = {:.0f} km relative to ILP, looking for the back edge...".format((f_b - mrange)/1000))
        # =========================================================================
        # EXECUTION PIPELINE ROUTER - TEIL 2: BACK EDGE DETECTION & GRID FUSION
        # =========================================================================
        shift_beg = fp_front1  # Anschluss an das Ende von Teil 1
        shift_end = shift_min 
        dist_step = (shift_end - shift_beg) / num_dist 
        
        # =========================================================================
        # NEW PIPELINE DEBUG: Initial Loop State Parameters
        # =========================================================================
        print("\n[NEW-DEBUG-BACKSCAN] Initialization Parameters:")
        print(f"  -> shift_beg (Front edge)  = {shift_beg/1000:.2f} km (rel to ILP: {(shift_beg - mrange)/1000:.2f} km)")
        print(f"  -> shift_end (Max search)  = {shift_end/1000:.2f} km (rel to ILP: {(shift_end - mrange)/1000:.2f} km)")
        print(f"  -> dist_step (Signed delta)= {dist_step/1000:.4f} km")
        print(f"  -> num_dist (Grid resolution) = {num_dist}")
        print(f"  -> concentric_flag = {concentric_flag}")
        
        stm2 = time.process_time()
        f_shift = shift_beg 
        loop_counter = 0

        while f_shift >= shift_end - .1 :
            loop_counter += 1
            enm2 = time.process_time()
            dum2 = enm2 - stm2
            if dum2 >= out_time_freq :
                print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                stm2 = enm2

            # Run search function with mode flag passed down
            f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
            
            # =========================================================================
            # NEW PIPELINE DEBUG: Step Evaluation Inside Loop
            # =========================================================================
            f_ok_is_hit = is_hit(f_ok)
            print(f"  [Loop #{loop_counter}] Testing f_shift = {f_shift/1000:.2f} km (rel: {(f_shift - mrange)/1000:.2f} km) -> Hit status: {f_ok_is_hit}")
            
            if not f_ok_is_hit :
                f_b = f_shift
                f_a = f_shift - dist_step
                # =========================================================================
                # NEW PIPELINE DEBUG: Loop Break Conditions Triggers
                # =========================================================================
                print(f"\n  [TRIGGER] 'No-Hit' encountered at f_shift = {f_shift/1000:.2f} km. Breaking out of raw scan loop.")
                print(f"            Setting Bisection brackets:")
                print(f"            -> f_b (No-Hit point, outer) = {f_b/1000:.2f} km")
                print(f"            -> f_a (Hit point, inner)    = {f_a/1000:.2f} km")
                print(f"            -> Raw bracket difference (f_a - f_b) = {(f_a - f_b)/1000:.4f} km")
                break
            
            if f_shift > shift_end :
                f_shift += dist_step
                if f_shift < shift_end :
                    f_shift = shift_end
            else :
                f_shift += dist_step

        print("\rProcessing distance complete", end='')

        # =========================================================================
        # NEW PIPELINE DEBUG: Final State Post-Loop Evaluation
        # =========================================================================
        final_f_ok_is_hit = is_hit(f_ok)
        print("\n[NEW-DEBUG-BACKSCAN] Scan Loop Ended.")
        print(f"  -> Final loop counter = {loop_counter}")
        print(f"  -> Final f_shift status = {f_shift/1000:.2f} km")
        print(f"  -> Final is_hit Evaluation Flag = {final_f_ok_is_hit}")
        print(f"  -> Final assigned f_a = {f_a/1000:.2f} km | f_b = {f_b/1000:.2f} km")

        if is_hit(f_ok):
            print("\rFootprint's back edge is at the maximum missile range. Single part footprint.")
            fp_back1 = shift_end
            fp_single = True
        else :
            
            f_y_keep = f_ok
            while True :
                f_x = f_a + (f_b - f_a) / 2
                f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                if is_hit(f_xy) :
                    f_a = f_x
                    f_y_keep = f_xy
                else :
                    f_b = f_x
                
                if concentric_flag:
                    dynamic_stop = max(stop_par, (abs(f_x) / R_e) * stop_par * 10.0)
                else:
                    dynamic_stop = stop_par

                if ((f_a - f_b) < (f_a - mrange) * t_acc) or (abs(f_a - f_b) < dynamic_stop) : 
                    break
            fp_back1 = f_a
            print("\rBack edge of the footprint found, location = {:.0f} km relative to ILP.".format((f_a - mrange)/1000))

            print("Looking for the footprint's second part...")

            if plot_hit_charts and i_type :
                if f_a == dist_limit : 
                    fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                else :
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                    
        """ Search for the second part of the footprint """
        if not fp_single :
            # Local helper to safely validate hits for both classic and concentric modes
            def check_hit(f_res):
                if concentric_flag:
                    return isinstance(f_res, np.ndarray) and f_res.size > 0
                return is_hit(f_res)

            shift_beg = f_b
            shift_end = shift_min 
            dist_step = (shift_end - shift_beg) / num_dist 
            
            stm2 = time.process_time()
            f_shift = shift_beg 
            while f_shift >= shift_end - .1 :
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= out_time_freq :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                
                # Check hit using the safe helper
                if check_hit(f_ok):
                    f_b = f_shift
                    f_a = f_shift - dist_step
                    # Save the hit for later use (back‑edge extraction)
                    # We need the missile time from the hit - in concentric it's f_ok[0]
                    break
                
                f_shift += dist_step
                    
            print("\rProcessing distance complete", end='')

            # FIX: Use safe helper to avoid NumPy array truth ambiguity errors
            if not check_hit(f_ok) :
                print('\rNo second part of footprint found behind the first part.')
                fp_single = True
            else :
                f_y_keep = f_ok
                while True :
                    f_x = f_a + (f_b - f_a) / 2
                    f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                    
                    # FIX: Use safe helper inside bisection loop
                    if check_hit(f_xy) :
                        f_b = f_x
                        f_y_keep = f_xy
                    else :
                        f_a = f_x
                    if (abs(f_a - f_b) < abs(f_b - mrange) * t_acc) or (abs(f_a - f_b) < stop_par) : 
                        break
                fp_front2 = f_b
                print("\rFront edge of the second part found, location = {:.0f} km relative to ILP.".format((f_b - mrange)/1000))

                if plot_hit_charts and i_type :
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)

                shift_beg = f_b
                shift_end = shift_min 
                            
                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, shift_end, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)        
                
                # FIX: Use safe helper here as well
                if check_hit(f_ok):
                    print("\nBack edge of the footprint's second part is at the maximum missile range")
                    fp_back2 = shift_end
                else :
                    f_a = shift_beg
                    f_b = shift_end
                    f_y_keep = f_ok
                    while True :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                        
                        # FIX: Use safe helper inside rear bisection loop
                        if check_hit(f_xy) :
                            f_a = f_x
                            f_y_keep = f_xy
                        else :
                            f_b = f_x
                        if ((f_a - f_b) < (f_a - mrange) * t_acc) or ((f_a - f_b) < stop_par) : 
                            break
                    fp_back2 = f_a
                    print("\rBack  edge of the second part found, location = {:.0f} km relative to ILP.".format((f_a - mrange)/1000))

        if fp_single :
            fp_part2 = np.zeros((0, 5))
            print("Calculating the footprint", end='')
        else :
            print("Calculating the first part of the footprint", end='')
            
        if fp_calc_param == 3 :
            fp_part1 = fp_part_calc_v3(fp_front1, fp_back1, concentric=concentric_flag)
        
        if not fp_single :
            print("\rCalculating the second (back) part of the footprint", end='')
            if fp_calc_param == 3 :
                fp_part2 = fp_part_calc_v3(fp_front2, fp_back2, concentric=concentric_flag)
                    
        print("\rFootprint calculation complete (Concentric={})".format(concentric_flag))
        return [fp_part1, fp_part2]

    # =================================================================
    # ÄUSSERER PIPELINE-START & RETURN FÜR DIE HAUPTFUNKTION
    # =================================================================
    fp_part_std = run_footprint_pipeline(t_itable, concentric_flag=False)

    if is_compare_mode:
        fp_part_con = run_footprint_pipeline(t_itable_con, concentric_flag=True)
        return [fp_part_std, fp_part_con]
    else:
        return [fp_part_std, [np.zeros((0, 5)), np.zeros((0, 5))]]

""" End of footprint_mode2 """