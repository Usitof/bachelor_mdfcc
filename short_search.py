import balmis as bm
import time
import tables_v2b_0 as tb
import tables_v2b_a as tb_con
import footprintv2 as fp
import numpy as np
from math import pi, radians, degrees, sin, cos, acos, asin, tan, ceil, sqrt
from fcc_constants import R_e, eps, angle_eps, out_time_freq, dist_limit, \
    beta_step, alfa_step, ang_quant, mrl_factor1, mrl_factor2
import fcc_constants

global dist_to_trj_min

shift_min = 1000 # min distance between ILP and MLP

"""
Concentric (3-D, angle-outer) interception helpers. Only caller is
short_search(concentric=True). See tables_v2b_a.py's module docstring for
the psi/xi/phi/theta_csm spherical-law-of-cosines definitions.

xi in that formula is the raw trj[:, 2] column as-is: the missile's own
geocentric angle from ITS OWN LAUNCH POINT, growing from 0 at launch. This
matches omegashift_2_xy()'s existing (unchanged, already-correct-for-classic)
use of the same formula with alfa=mrange/R_e (the full launch-to-impact
range) as one fixed side -- consistent only if MLP is the missile's launch
point, not its impact point: phi is then the dihedral measured at the
launch point, between the ILP direction and the missile's initial heading,
which is exactly the direction xi increases in.
"""

def _theta_ilp(xi, cos_psi, sin_psi, cos_phi):
    """ Geocentric angle from ILP to a missile point `xi` [rad] along its
    own trajectory (measured from its own launch point, as in trj[:, 2]),
    given the ILP-to-MLP arc via cos_psi/sin_psi and the trajectory-plane
    dihedral via cos_phi. """
    c = cos_psi * cos(xi) + sin_psi * sin(xi) * cos_phi
    if c > 1.0 :
        c = 1.0
    elif c < -1.0 :
        c = -1.0
    return acos(c)


def _ray_time_at_radius(ray, m_rad) :
    """
    Interceptor flight time to reach geocentric radius m_rad on one ray, by
    radius bracket + linear interpolation in radius.
    Returns ((t, psi), True) if m_rad is inside the ray's radius span, else (None, False).
    ray columns: [psi_deg, time, rv, theta].
    """
    rr = ray[:, 2]
    n = len(ray)
    if n == 0 or m_rad < rr[0] or m_rad > rr[-1] :
        return None, False
    i_a, i_b = 0, n - 1
    while (i_b - i_a) > 1 :
        i_x = (i_a + i_b) // 2
        if rr[i_x] > m_rad :
            i_b = i_x
        else :
            i_a = i_x
    r1, r2 = rr[i_a], rr[i_b]
    denom = r2 - r1
    f = 0.0 if abs(denom) < 1e-12 else (m_rad - r1) / denom
    t = ray[i_a, 1] + f * (ray[i_b, 1] - ray[i_a, 1])
    psi = ray[i_a, 0] + f * (ray[i_b, 0] - ray[i_a, 0])
    return (t, psi), True


def _flight_time_two_ray(int_table, theta_grid, theta, m_rad) :
    """
    Two-ray-interpolated interceptor flight time at ILP angle `theta` and
    radius m_rad. Brackets theta between adjacent rays k, k+1; radius-brackets
    m_rad in each; interpolates the two flight times by the angle fraction.
    Conservative: both bracketing rays must reach m_rad, otherwise treated as
    unreachable (no one-sided extrapolation).
    theta_grid: ascending array of ray angles = [int_table[k][0,3]].
    """
    if theta < theta_grid[0] or theta > theta_grid[-1] :
        return {'reachable': False, 't': None, 'psi': None}
    k = int(np.searchsorted(theta_grid, theta, side='right') - 1)
    k = max(0, min(k, len(int_table) - 2))
    th_k, th_k1 = theta_grid[k], theta_grid[k + 1]
    ray_k, ray_k1 = int_table[k], int_table[k + 1]

    rk, ok_k = _ray_time_at_radius(ray_k, m_rad)
    rk1, ok_k1 = _ray_time_at_radius(ray_k1, m_rad)

    if ok_k and ok_k1 :
        denom = th_k1 - th_k
        fa = 0.0 if abs(denom) < 1e-15 else (theta - th_k) / denom
        t = rk[0] + fa * (rk1[0] - rk[0])
        psi = rk[1] + fa * (rk1[1] - rk[1])
        return {'reachable': True, 't': t, 'psi': psi}

    return {'reachable': False, 't': None, 'psi': None}


def _concentric_forward_search(trj, int_table, psi_rad, phi_rad,
                               t_int_lnc, t_end, h_int_min, maxia, op_range, R_e) :
    """
    Forward walk over the RAW missile trajectory (trj, as passed into
    short_search -- not the shift/turn-transformed tar_traj) looking for the
    earliest feasible interception.

    trj columns: [time, R, alfa], alfa non-decreasing (missile's own angle
    from its own launch point). psi_rad = dist_opt / R_e is the ILP-to-MLP
    arc, phi_rad = fi_opt is the trajectory-plane dihedral (see module note
    above on the xi convention this requires).

    Returns a 5-tuple-as-array [m_time, m_rad, alfa, int_psi_deg,
    int_flight_time] for the first feasible point (scanning forward in
    time), or False if none is feasible in [t_int_lnc, t_end].
    """
    trj = np.asarray(trj, dtype=np.float64)
    n = len(trj)
    if n < 1 :
        return False

    cos_psi, sin_psi, cos_phi = cos(psi_rad), sin(psi_rad), cos(phi_rad)
    theta_grid = np.array([int_table[k][0, 3] for k in range(len(int_table))])

    i0 = int(np.searchsorted(trj[:, 0], t_int_lnc, side='left'))
    if i0 >= n :
        return False

    for i in range(i0, n) :
        m_time = trj[i, 0]
        if m_time > t_end :
            break
        m_rad = trj[i, 1]

        alt = m_rad - R_e
        if alt < h_int_min :
            continue
        if maxia and m_rad > R_e + maxia :
            continue
        if op_range and m_rad > op_range :
            continue

        theta = _theta_ilp(trj[i, 2], cos_psi, sin_psi, cos_phi)
        look = _flight_time_two_ray(int_table, theta_grid, theta, m_rad)
        if not look['reachable'] :
            continue

        if look['t'] > (m_time - t_int_lnc) :
            continue

        return np.array([m_time, m_rad, trj[i, 2], look['psi'], look['t']],
                        dtype=np.float64)

    return False

def short_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, t_delay=0, h_discr=0, mode2=False, debug_hits=False, sls_second=False, concentric=False, mrange=None) :
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
    concentric :
        if True, search a tables_v2b_a-style angle-outer (theta, radius) int_table
        instead of the classic beta-indexed one, via the raw-trajectory forward walk
        below. mode2 must be True (concentric only has a meaning for shift/omega).
    mrange :
        missile range in metres (trj[-1, 2] * R_e); computed from trj if not given.
        Only used to convert the found hit's (omega, shift) into (x, y) km for the
        returned array -- it does not affect reachability.

    Returns
    -------
    Classic / mode2 (concentric=False): a (hit_i, hit_m, hit_i1, hit_i2) tuple, or
    False if no interception was found.
    Concentric (concentric=True): a 7-element numpy array
        [m_time, m_rad, m_alfa, hit_x_km, hit_z_km, int_psi_deg, int_flight_time],
    or False if no interception was found. Note the different return type --
    callers must not do a bare `if result:` on this (see _is_hit-style helpers).

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

    if not mode2 :
        tar_traj = bm.trj_from_center(trj, R_e, fi_opt, dist_opt) # tar_trj elements contain: time, R, alfa_r, r, beta_r, x, z
    else :
        tar_traj = bm.trj_shift_turn(trj, R_e, dist_opt, fi_opt)  # in mode2: dist_opt is shift, fi_opt is omega

    if not np.any(tar_traj) :
        return False
    
    global dist_to_trj_min
    dist_to_trj_min = min(tar_traj[:, 5]) / 1000

    """ Limitations on time of interceptor launch calculated before sampling """
    
    t_bout = t_int_lnc - t_delay
    #if t_delay == 0 :
    #    t_int_lnc = 0

    i_dr = 0
    if not sls_second :
        if det_range > 0 : # detecion range limitations 
            while tar_traj[i_dr, 3] > det_range : # TODO optimize search $$$
                i_dr += 1
                if i_dr == len(tar_traj) - 1 :
                    if debug_print :
                        print("short search exception: missile always beyond detection range)")
                    return False
            t_int_lnc = max(t_bout, tar_traj[i_dr, 0]) + t_delay  # delay or no delay after detection? $$$
        
        #i_ohd=0
        #if t_delay != 0 : # over the horizon detection criterium
            i_ohd = 0
            while tar_traj[i_ohd, 4] <= 0 : # 4 - beta, 6 - z
                i_ohd += 1
                if i_ohd == len(tar_traj) : # whole trajectory below horizon
                    if debug_print : 
                        print("short search exception: ohd criteria used, but the whole trajecory is below horizon")
                    return False
                    #break
            if i_ohd < len(tar_traj) : # i.e. skip this criterium if whole missile trajectory is below horizon
                i_ohd_x = i_ohd
                while tar_traj[i_ohd_x, 4] > 0: # 4 - beta, 6 - z
                    if (tar_traj[i_ohd_x, 0] - tar_traj[i_ohd, 0]) >= t_delay :
                        break
                    i_ohd_x += 1
                    if i_ohd_x == len(tar_traj) : # whole trajectory below horizon
                        if debug_print : 
                            print("short search exception: ohd criteria used, but the trajecory drops below horizon before t_delay")
                        return False                  
                t_int_lnc = max(t_int_lnc, tar_traj[i_ohd, 0] + t_delay)
        
        else :
            #print("shse fcc_constants.sat_delay = ", fcc_constants.sat_delay)
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

    # =====================================================================
    # CONCENTRIC: forward walk over the raw trajectory using the corrected
    # spherical-law-of-cosines mapping (see _concentric_forward_search).
    # Shares t_int_lnc / t_end / h_int_min / maxia / op_range with the
    # classic path above; diverges here because the interceptor table is
    # theta-indexed (angle-outer) rather than beta-indexed.
    # =====================================================================
    if concentric :
        psi_rad = dist_opt / R_e   # dist_opt ("shift") is metres; psi is the ILP-MLP arc
        phi_rad = fi_opt           # fi_opt ("omega") is already radians

        _hit = _concentric_forward_search(trj, int_table, psi_rad, phi_rad,
                                          t_int_lnc, t_end, h_int_min, maxia, op_range, R_e)
        if _hit is False :
            return False

        m_time, m_rad, m_alfa, int_psi_deg, int_flight_time = _hit
        if mrange is None :
            mrange = trj[-1, 2] * R_e
        hit_x_km, hit_z_km = omegashift_2_xy(fi_opt, dist_opt, mrange)
        return np.array([m_time, m_rad, m_alfa, hit_x_km, hit_z_km,
                         int_psi_deg, int_flight_time], dtype=np.float64)

#    beta_n = 360
    
    """ Sampling rate for missile trajectory resampling, must be the same as int_table sampling rate $$$"""
    #beta_step = 0.025 #0.25

    mis_traj = tb.mis_traj_sample(tar_traj) #, beta_step) -- don't set beta_step unless necessary (time, rad, beta, x, y)
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
    
    while mis_traj[i2_m_beg, 2] < beta_zero : #int_table[0][0, 3] :
        """ this needs to be addressed by sampling int table to negative betas -- done """
        if i2_m_beg == len(mis_traj) - 1 :
            if debug_print : 
                print("False: all missile trajectory below lowest interceptor beta")
            return False
        i2_m_beg += 1
#    print("int_table[0][1, 3]={}, i2_m_beg = {}, mis_traj[i2_m_beg, 0]={}".format(int_table[0][1, 3], i2_m_beg, mis_traj[i2_m_beg]))

    m_beg = max(i1_m_beg, i2_m_beg)
    beta_il = mis_traj[m_beg, 2]

    int_beta_il = 0 # index of the same beta in interception table

    if (int_table[int_beta_il][0, 3] > m_beta_max) : # (beta_il - angle_eps)) : <- ???
        if debug_print : 
            print("All of the possibly interceptable missile trajectory is below lowest interceptor beta")
        return False
    
    while int_table[int_beta_il][0, 3] < (beta_il - angle_eps) : # TODO optimize search $$$
        int_beta_il += 1
        if not np.any(int_table[int_beta_il]) : # to account for empty betas near vertical
            if debug_print :
                print("!!!", end='')
            if abs(pi/2 -  beta_il) < angle_eps :
                int_beta_il = len(int_table) - 1
                break
            else:
                m_beg += 1
                beta_il = mis_traj[m_beg, 2]           
                int_beta_il -= 1

    if abs(int_table[int_beta_il][0, 3] - beta_il) > angle_eps : 
        if debug_print : 
            print("short search exception: betas not equal: int_beta_il={} int_beta={}, mis_beta-eps={}".format(int_beta_il, int_table[int_beta_il][0, 3], beta_il-angle_eps))
            print("i1_m_beg = {} mis_traj[i1_m_beg, 0]={} mis_traj[i1_m_beg, 2]={} t_int_lnc = {}".format(i1_m_beg, mis_traj[i1_m_beg, 0], mis_traj[i1_m_beg, 2], t_int_lnc))
            print("i2_m_beg = {} mis_traj[i2_m_beg, 2]={} int_table[0][0, 3] = {}".format(i2_m_beg, mis_traj[i2_m_beg, 2], int_table[0][0, 3]))
            print("i1_m_beg = {} i2_m_beg = {}".format(i1_m_beg, i2_m_beg))
        #print(int_beta_il, '\n', int_table[int_beta_il], '\n', int_table[int_beta_il+1])
        return False
        
#    i_m_beta = m_beg
    i_i_beta = int_beta_il
    hit = False
#    while (mis_traj[i_m_beta, 0] < t_end) :
    #print(mis_traj[m_beg, 0], mis_traj[len(mis_traj) - 1, 0], t_end, det_range/1000)
    for i_m_beta in range(m_beg, len(mis_traj)) : # TODO make range end calculated from t_end
#        print("\rProcessing beta {:f}".format(degrees(mis_traj[i_m_beta, 2])), end='')
#        itemp = int_table[i_i_beta]
        if np.any(int_table[i_i_beta]) :
            m_beta = mis_traj[i_m_beta, 2]
            if abs(m_beta - int_table[i_i_beta][0, 3]) > angle_eps :
                print("betas not equal exception: m={}, i={}".format(mis_traj[i_m_beta, 2], int_table[i_i_beta][0, 3]))
                print("i_m_beta={}, i_i_beta={}".format(i_m_beta, i_i_beta))
    #            print("int_table[i_i_beta] 1: {} and 10: {}".format(int_table[1], int_table[2]))
                hit = False
                break
                #return False -- 221030 18:35
            m_rad = mis_traj[i_m_beta, 1]
            #if i_m_beta == 610 :                           # debug delete
            #    print("i_m_beta = 610", mis_traj[610])     # debug delete
            
            x_rad = mis_traj[i_m_beta, 3]
            y_rad = mis_traj[i_m_beta, 4] + R_e
            int_altitude = sqrt(x_rad*x_rad + y_rad*y_rad) - R_e
            
            if ((not op_range) or (m_rad <= op_range)) and (m_beta > othi_r) and ((not maxia) or (int_altitude <= maxia)) : # and ((not h_discr) or (int_altitude < bm.atm_limit_100)) : #checking in-atm only functional for endo
                #continue
                i_a = 0
                i_b = len(int_table[i_i_beta]) - 1
                if i_b > 0 : # ==0 when only one point in the beta ray
                    if int_table[i_i_beta][i_b, 2] > m_rad :
                        while (i_b - i_a) > 1 :
                            i_x = int(ceil(i_a + i_b) / 2)
                            f_x = int_table[i_i_beta][i_x, 2] - m_rad
                            if f_x > 0 :
                                i_b = i_x
                            elif f_x < 0 :
                                i_a = i_x
                            else :
                                #hit = int_table[i_i_beta][i_b], mis_traj[i_m_beta]
                                hit = int_table[i_i_beta][i_b], mis_traj[i_m_beta], int_table[i_i_beta][i_b], int_table[i_i_beta][i_b]
                        i_hit = i_b
                        if (int_table[i_i_beta][i_hit - 1, 2] < m_rad) :
                            r1 = int_table[i_i_beta][i_hit - 1, 2]
                            r2 = int_table[i_i_beta][i_hit, 2]
                            t1 = int_table[i_i_beta][i_hit - 1, 1]
                            t2 = int_table[i_i_beta][i_hit, 1]
                            t_hit = t1 + (t2 - t1) * (m_rad - r1) / (r2 - r1)
                
                            psi1 = int_table[i_i_beta][i_hit - 1, 0]
                            psi2 = int_table[i_i_beta][i_hit, 0]
                            psi_hit = psi1 + (psi2 - psi1) * (m_rad - r1) / (r2 - r1)
                                
                            #hit = [psi_hit, t_hit, m_rad, int_table[i_i_beta][1, 3]], mis_traj[i_m_beta]
                            hit = [psi_hit, t_hit, m_rad, int_table[i_i_beta][1, 3]], mis_traj[i_m_beta], int_table[i_i_beta][i_hit - 1], int_table[i_i_beta][i_hit]
                        
                        elif int_table[i_i_beta][0, 2] == m_rad :
                            #hit = int_table[i_i_beta][0], mis_traj[i_m_beta]
                            hit = int_table[i_i_beta][0], mis_traj[i_m_beta], int_table[i_i_beta][0], int_table[i_i_beta][0]
                        """
                        elif False : # i.e. point with i_a == 0 is the first above ground is further than "underground" MIP
                            #print("short search: projected underground missile trj point processed")
                            print("\r> Under < fi_opt = {:6.2f}, dist_opt = {:9.3f} mode2={}".format(degrees(fi_opt), dist_opt/1000, mode2), end='')
                            hit_found = False
                            m_rad_1 = mis_traj[i_m_beta - 1, 1]
                            m_height = sqrt(R_e*R_e + m_rad_1*m_rad_1 + 2 * R_e * m_rad_1 * sin(mis_traj[i_m_beta - 1, 2])) - R_e
                            if m_height >= h_int_min : #4
                                i2_beg = 0
                                for i1 in range(0, len(int_table[i_i_beta])) :
                                    i1r = int_table[i_i_beta][i1, 2]
                                    if i1r >= m_rad_1 :
                                        break
                                    for i2 in range (i2_beg, len(int_table[i_i_beta + 1])) :
                                        if int_table[i_i_beta][i1, 0] == int_table[i_i_beta + 1][i2, 0] : #3
                                            i2r = int_table[i_i_beta + 1][i2, 2]
                                            #if (i1r > m_rad_1) and (i2r > m_rad_1) :
                                            if i2r >= m_rad_1 :
                                                break
                                            i1x = i1r * cos(int_table[i_i_beta][i1, 3])
                                            i1y = i1r * sin(int_table[i_i_beta][i1, 3])
                                            i2x = i2r * cos(int_table[i_i_beta + 1][i2, 3])
                                            i2y = i2r * sin(int_table[i_i_beta + 1][i2, 3])
                                            #i1x = int_table[i_i_beta][i1, 2] * cos(int_table[i_i_beta][i1, 3])
                                            #i1y = int_table[i_i_beta][i1, 2] * sin(int_table[i_i_beta][i1, 3])
                                            #i2x = int_table[i_i_beta + 1][i2, 2] * cos(int_table[i_i_beta + 1][i2, 3])
                                            #i2y = int_table[i_i_beta + 1][i2, 2] * sin(int_table[i_i_beta + 1][i2, 3])
                                            m1x = m_rad * cos(mis_traj[i_m_beta, 2])
                                            m1y = m_rad * sin(mis_traj[i_m_beta, 2])
                                            #m2x = mis_traj[i_m_beta - 1, 1] * cos(mis_traj[i_m_beta - 1, 2])
                                            #m2y = mis_traj[i_m_beta - 1, 1] * sin(mis_traj[i_m_beta - 1, 2])
                                            m2x = m_rad_1 * cos(mis_traj[i_m_beta - 1, 2])
                                            m2y = m_rad_1 * sin(mis_traj[i_m_beta - 1, 2])
                                            hit = bm.segment_intersection([i1x, i1y], [i2x, i2y], [m1x, m1y], [m2x, m2y])
                                            if debug_hits :
                                                print(m1x, m1y, m2x, m2y)
                                                print(i1x, i1y, i2x, i2y)
                                            if hit : #2
                                                hit_found = True
                                                print("Underground hit")
                                                h_hit = sqrt((R_e + hit[1])*(R_e + hit[1]) + hit[0]*hit[0]) - R_e # local height of interception point
                                                if h_hit < h_int_min : #1
                                                    hit = False
                                                else :
                                                    r_hit = sqrt(hit[0]*hit[0] + hit[1]*hit[1])
                                                    beta_hit = asin(hit[1] / r_hit)
                                                    m1t = mis_traj[i_m_beta - 1, 0]
                                                    m2t = mis_traj[i_m_beta, 0]
                                                    t_hit_m = m1t + (m2t - m1t) * ((hit[0] - m1x) / (m2x - m1x))
                                                    i1t = int_table[i_i_beta][i1, 1]
                                                    i2t = int_table[i_i_beta + 1][i2, 1]
                                                    t_hit_i = i1t + (i2t - i1t) * ((hit[0] - i1x) / (i2x - i1x))
                            
                                                    psi1 = int_table[i_i_beta][i1, 0]
                                                    psi2 = int_table[i_i_beta + 1][i2, 0]
                                                    r1 = int_table[i_i_beta][i1, 2]
                                                    r2 = int_table[i_i_beta + 1][i2, 2]
                                                    psi_hit = psi1 + (psi2 - psi1) * (m_rad - r1) / (r2 - r1)
                                                    
                                                    #hit = [psi_hit, t_hit_i, r_hit, beta_hit, psi1, i1t, r1, psi2, i2t, r2], [t_hit_m, r_hit, beta_hit, r_hit * cos(beta_hit), r_hit * sin(beta_hit)]
                                                    hit = [psi_hit, t_hit_i, r_hit, beta_hit], [t_hit_m, r_hit, beta_hit, r_hit * cos(beta_hit), r_hit * sin(beta_hit)]
                                                    if debug_hits : #0
                                                        print("Debug hit")
                                                        print("coords: x={:.0f}, y={:.0f}, r={:.0f}, beta={:.3f}".format(hit[0], hit[1], r_hit, degrees(beta_hit)))
                                                        print("time: missile {:.3f},  interceptor {:.3f}, psi={:.3f}".format(t_hit_m, t_hit_i, psi_hit))
                                                        print("interceptor points:")
                                                        print("i1: ", int_table[i_i_beta][i1], "\ni2: ", int_table[i_i_beta + 1][i2])
                                                        print("missile points")
                                                        print("m1: ",mis_traj[i_m_beta - 1], "\nm2: ", mis_traj[i_m_beta])
                                                    else : pass #0
                                                
                                                    break
                                                # if #1 
                                            else : pass # if #2
                                        elif int_table[i_i_beta][i1, 0] > int_table[i_i_beta + 1][i2, 0] : # if #3 # all further will be ">" too
                                            i2_beg = i2
                                            break
                                    if hit_found :
                                        break
                                # } for i1
                            # if #4
                        # else # handling of partially underground segment
                        """   
            
                    elif int_table[i_i_beta][i_b, 2] == m_rad :
                    #elif (m_rad - int_table[i_i_beta][i_b, 2]) / m_rad < 0.01 : # miss by less than 0.1%  == hit $$$
                        #hit = int_table[i_i_beta][i_b], mis_traj[i_m_beta]
                        hit = int_table[i_i_beta][i_b], mis_traj[i_m_beta], int_table[i_i_beta][i_b], int_table[i_i_beta][i_b]
                elif int_table[i_i_beta][0, 2] == m_rad :
                    #hit = int_table[i_i_beta][0], mis_traj[i_m_beta]
                    hit = int_table[i_i_beta][0], mis_traj[i_m_beta], int_table[i_i_beta][0], int_table[i_i_beta][0]
                    
                if hit :
                    #hit_i, hit_m = hit
                    hit_i, hit_m, hit_i1, hit_i2 = hit
                    """ check that int flight time is less than time from the earliest possible int launch time and hit,"""
                    """ and hit is above acceptable interception altitude  >>>better would be to calculate local height<<< """ # TODO $$$
                    if hit_i[3] >= othi_r :
                        if (hit_i[1] <= (hit_m[0] - t_int_lnc)) and (hit_m[0] < t_end) :
                            if debug_hits :
                                print("hit i_angle={:f}, m_angle={:f}".format(degrees(hit_i[3]), degrees(hit_m[2])))
                            else :
                                break
                        else :
                            """
                            if i_m_beta > 674 :
                                print("hit_i[1] > (hit_m[0] - t_int_lnc)")
                                print("i_m_beta=", i_m_beta, mis_traj[i_m_beta], "\ni_i_beta=", i_i_beta, int_table[i_i_beta][i_b, 3], "det_range=", det_range/1000)
                                if hit_i[1] > (hit_m[0] - t_int_lnc) :
                                    print(hit_i, "\n", hit_m, t_int_lnc)
                                else :
                                    print("(hit_m[0] >= t_end", hit_m, t_end)
                            """
                            hit = False
                    else :
                        hit = False
                    
        # if np.any =====================

#        if i_m_beta == len(mis_traj) - 1 :
#                print("False: no hits")
#                return False
        if i_m_beta < (len(mis_traj) - 1) :
            m_beta = mis_traj[i_m_beta, 2] # assigned at the beginning of the cycle
            if (m_beta < m_beta_max) :
                if i_m_beta < i_m_beta_max :
                    i_i_beta += 1
                elif i_m_beta > i_m_beta_max:
                    i_i_beta -= 1
            else :
                if (mis_traj[i_m_beta + 1, 2] < m_beta) :
                    i_i_beta -= 1
                
        if i_i_beta == -1 :
            if debug_print : print("False: no hits (M beta lower than lowest I beta)")
            hit = False
            break
            #return False -- 221030 18:35
            
            
#        i_m_beta += 1
        """ Delete
        if mis_traj[i_m_beta, 2] == 0 :
            print("False by 0")
            return False                            
        """
    # end of for i_m_beta... 

    et = time.time()
    elapsed_time = et - st
    if debug_print :
        print("Full trajectory generation and search time = {:.3f}s, hit=>>>>>{}".format(elapsed_time, hit != False))
    
    if hit :
        #r_hit = (hit[1][3], hit[1][4])
        if debug_print :
            i_ohd -= 1
            print("hit=({:.0f},{:.0f}), m_time={:.3f} s, i_time={:.3f} s, it_delta={:.3f} s, i_beta={:.2f}".format(hit[1][3], hit[1][4], hit[1][0], hit[0][1], hit[1][0] - t_int_lnc - hit[0][1], hit[0][0]))
            print("t_int_lnc={:.2f}, t_bout={:.2f}, t_det={:.2f}, t_ohd={:.2f}, t_end={:.2f}".format(t_int_lnc, t_bout, tar_traj[i_dr + 1, 0] + t_delay, tar_traj[i_ohd, 0] + t_delay, t_end))
    else :
        #r_hit = False
        if debug_print :
            print("False")
            i_ohd -= 1
            print("t_int_lnc={:.2f}, t_bout={:.2f}, t_det={:.2f}, t_ohd={:.2f}, t_end={:.2f}".format(t_int_lnc, t_bout, tar_traj[i_dr + 1, 0] + t_delay, tar_traj[i_ohd, 0] + t_delay, t_end))

    return(hit) #r_hit

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
    """
    Mode2 (omega/shift) footprint. The search/bisection logic below is the
    original short_search_old.footprint_mode2 algorithm, unchanged, wrapped
    in _compute_part() so it can be run once against a classic table and
    once against a concentric table.

    t_itable_con :
        concentric interception table (tables_v2b_a-style angle-outer grid).
        If None, only the classic search (against t_itable) is performed and
        the concentric slot of the result is returned as empty placeholders.

    Returns
    -------
    [fp_part_std, fp_part_con]
        each element has the original footprint_mode2 shape [fp_part1, fp_part2]
        (or False, on the old exception paths below) -- fp_part_std from
        searching t_itable with concentric=False, fp_part_con from searching
        t_itable_con with concentric=True (or [zeros, zeros] if t_itable_con
        is None).
    """

    def _is_hit(result) :
        """
        s_func (short_search2/sls_search2) returns a numpy array on a
        concentric hit, and False/a tuple in classic mode. A bare `if result:`
        on a multi-element array raises numpy's "truth value of an array is
        ambiguous" error, so every boolean check on a search result goes
        through here instead. The raw value itself (array, tuple or False)
        is left untouched everywhere it's used for anything other than a
        yes/no check (e.g. when handed to fp.plot_hit2 below).
        """
        if isinstance(result, np.ndarray) :
            return result.size > 0
        return bool(result)

    def _compute_part(active_table, concentric) :
        """
        Original footprint_mode2 body (short_search_old), parameterised by
        which interceptor table to search (active_table) and whether s_func
        should run in concentric mode. Logic/structure is otherwise untouched,
        aside from routing hit/miss checks through _is_hit() above.
        """
        fp_calc_param = 3

        stop_par = 10

        #int_max_range = max([int_traj[len(int_traj) - 1][2] for int_traj in active_table if np.any(int_traj)])
        # this is in fact max r rather than max range

        if concentric :
            # active_table is theta-indexed (angle FROM the ILP, tables_v2b_a-style):
            # ray 0 is ~0 (the ILP itself), not a signed "how far behind vertical"
            # angle like classic beta, so beta_min's "-2*beta_min*R_e" doesn't carry
            # over -- it would collapse to ~0 here and starve the front-edge search
            # of any room to explore past the ILP. The table's actual reach is its
            # LAST ray (theta_max), so use that directly as the range bound instead.
            theta_max_val = active_table[-1][0, 3]
            int_max_range = theta_max_val * R_e
            print("theta_max={:.2f}, int_max_range={:.1f} km (concentric)".format(degrees(theta_max_val), int_max_range/1000))
        else :
            beta_min = active_table[0][0, 3] - radians(beta_step)
            int_max_range = -2 * beta_min * R_e # from basic geometry
            print("beta_min={:.2f}, int_max_range={:.1f} km".format(degrees(beta_min), int_max_range/1000))
        #print(m_type, i_type)

        f_alfa = t_trj[len(t_trj) - 1, 2]
        mrange = f_alfa * R_e
        fp_single = False   # assuming it is a two-part footprint unless proven otherwise
        dist_step = (shift_min - mrange) / num_dist # negative value

        ilp_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, mrange, op_range, det_range, t_delay, h_discr, concentric=concentric) # shift == mrange means ILP is the impact point
        if not _is_hit(ilp_ok) : # ILP not defendable, find a defendable position behind it
            print("Interceptor position undefendable. Attempting to find a footprint behind(?) it...")
            #shift_beg = mrange
            shift_beg = min( (mrange + int_max_range), dist_limit)
            shift_end = shift_min # smallest shift, corresponds to footprint's max distance behind the ILP
            dist_step = (shift_end - shift_beg) / num_dist # negative value

            stm2 = time.process_time()
            f_shift = shift_beg # f_shift is distance from ILP to MLP
            f_b = f_shift
            while f_shift >= shift_end - .1 :
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= 1 :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                if _is_hit(f_ok) :
                    f_b = f_shift
                    f_a = f_shift - dist_step
                    break
                f_shift += dist_step
            #while f_shift
            if f_b > mrange :
                print("\nFootprint in front of ILP", end='')

            print("\rProcessing distance complete", end='')

            if not _is_hit(f_ok) :
                print('\rNo footprint found behind undefendable interceptor position.\nIncreasing "Mode 2 Footprint number of steps" might help to find one.')
                fp_part = np.zeros((0, 5))
                return [fp_part, fp_part]

        else : # ilp_ok == True
            f_shift = mrange + int_max_range / 8 # f_shift is distance from ILP to impact point (??? -- must be to MLP)
        #search for an undefendable point in front of the ILP
        #beg        
            f_shift0 = mrange
            f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)

            while _is_hit(f_ok) :             
                f_shift0 = f_shift
                if f_shift >= int_max_range + mrange :
                    break
                f_shift = f_shift * 2 - mrange
                f_shift = min(f_shift, int_max_range + mrange)
                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)

            if True : #not f_ok :
                f_b = f_shift0
                f_a = f_shift
            else :
                print("\nProbing2 exception (a): f_shift = {:.2f} >= int_max_range + mrange = {:.2f}".format(f_shift, int_max_range + mrange))
                print("(meaning: front edge of the footprint is at the interceptor's max range)")
                return False

        # f_a, f_b is the initial range for finding the forward border of the footprint, f_a > f_b    
        #print("\nForward border between f_a = {:.2f} f_b = {:.2f}".format((f_a - mrange)/1000,(f_b - mrange)/1000))
        while True :
            f_x = f_a + (f_b - f_a) / 2
            f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric)
            if _is_hit(f_xy) :
                f_b = f_x
                f_y_keep = f_xy
            else :
                f_a = f_x
                #f_y_keep = f_xy
            if (abs(f_a - f_b) < abs(f_b - mrange) * t_acc) or (abs(f_a - f_b) < stop_par) : # "<10" to stop the cycle in case somehow f_a and f_b are on the same side
                #print("\nf_a = {:.2f} f_b = {:.2f} f_b - f_a = {:.2f}".format((f_a - mrange)/1000,(f_b - mrange)/1000, (f_b - f_a)/1000))
                break

        if f_a <= int_max_range + mrange : # footprint outer edge is NOT farther than max
            if plot_hit_charts and i_type :
                if f_a == dist_limit : #int_max_range + mrange :
                    fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                else :
                    pass
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)

        else :
            print("\nProbing2 exception (b): f_a = {:.2f} > int_max_range = {:.2f}".format(f_a/1000, int_max_range/1000))
            return False


        fp_front1 = f_b
        print("\rFront edge of the footprint found, location = {:.0f} km relative to ILP, looking for the back edge...".format((f_b - mrange)/1000))

        shift_beg = f_b
        shift_end = shift_min # smallest shift, corresponds to footprint's max distance behind the ILP
        dist_step = (shift_end - shift_beg) / num_dist # negative value

        stm2 = time.process_time()
        f_shift = shift_beg # f_shift is distance from ILP to MLP
        while f_shift >= shift_end - .1 :
            enm2 = time.process_time()
            dum2 = enm2 - stm2
            if dum2 >= out_time_freq :
                print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                stm2 = enm2

            f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
            #print(True if f_ok else False, f_shift)
            if not _is_hit(f_ok) :
                f_b = f_shift
                f_a = f_shift - dist_step
                break

            if f_shift > shift_end :
                f_shift += dist_step
                if f_shift < shift_end :
                    f_shift = shift_end
            else :
                f_shift += dist_step


        #print(True if f_ok else False, f_shift)

        print("\rProcessing distance complete", end='')
        #print("\nf_a={:.0f}km f_b={:.0f}km f_shift={:.0f}km".format(f_a/1000, f_b/1000, f_shift/1000))

        if _is_hit(f_ok) :
            print("\rFootprint's back edge is at the maximum missile range. Single part footprint.")
            fp_back1 = shift_end
            fp_single = True
        else :
            #print(f_ok, f_shift)
            while True :
                f_x = f_a + (f_b - f_a) / 2
                f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric)
                if _is_hit(f_xy) :
                    f_a = f_x
                    f_y_keep = f_xy
                else :
                    f_b = f_x
                if ((f_a - f_b) < (f_a - mrange) * t_acc) or ((f_a - f_b) < stop_par) : # "<10" to stop the cycle in case somehow f_a and f_b are on the same side
                    break
            fp_back1 = f_a
            print("\rBack  edge of the footprint found, location = {:.0f} km relative to ILP, looking for the footprint's second part...".format((f_a - mrange)/1000))

            if plot_hit_charts and i_type :
                if f_a == dist_limit : #int_max_range + mrange :
                    fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                else :
                    pass
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)

        """ Search for the second part of the footprint """
        if not fp_single :
            shift_beg = f_b
            #print("f_b = ", f_b)
            shift_end = shift_min # smallest shift, corresponds to footprint's max distance behind the ILP
            dist_step = (shift_end - shift_beg) / num_dist # negative value

            stm2 = time.process_time()
            f_shift = shift_beg # f_shift is distance from ILP to MLP
            while f_shift >= shift_end - .1 :
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= out_time_freq :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                #print("f_shift={:.3f} f_ok={}".format(f_shift, f_ok))
                if _is_hit(f_ok) :
                    f_b = f_shift
                    f_a = f_shift - dist_step
                    break
                f_shift += dist_step
                #print("f_shift={:.3f}".format(f_shift))
            #while
            print("\rProcessing distance complete", end='')

            if not _is_hit(f_ok) :
                print('\rNo second part of footprint found behind the first part.\nIncreasing "Mode 2 Footprint number of steps" might help to find one.')
                fp_single = True
            else :
                while True :
                    f_x = f_a + (f_b - f_a) / 2
                    f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    if _is_hit(f_xy) :
                        f_b = f_x
                        f_y_keep = f_xy
                    else :
                        f_a = f_x
                        #f_y_keep = f_xy
                    if (abs(f_a - f_b) < abs(f_b - mrange) * t_acc) or (abs(f_a - f_b) < stop_par) : # "<10" to stop the cycle in case somehow f_a and f_b are on the same side
                        break
                #while
                fp_front2 = f_b
                print("\rFront edge of the second part of the footprint found, location = {:.0f} km relative to ILP, looking for the back edge...".format((f_b - mrange)/1000))

                if plot_hit_charts and i_type :
                    if f_a == dist_limit : #int_max_range + mrange :
                        fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                    else :
                        pass
                        fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)

                shift_beg = f_b
                shift_end = shift_min # smallest shift, corresponds to footprint's max distance behind the ILP

                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, shift_end, op_range, det_range, t_delay, h_discr, concentric=concentric)        
                if _is_hit(f_ok) :
                    print("\nBack edge of the footprint's second part is at the maximum missile range")
                    fp_back2 = shift_end
                else :
                    f_a = shift_beg
                    f_b = shift_end
                    while True :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric)
                        if _is_hit(f_xy) :
                            f_a = f_x
                            f_y_keep = f_xy
                        else :
                            f_b = f_x
                        if ((f_a - f_b) < (f_a - mrange) * t_acc) or ((f_a - f_b) < stop_par) : # "<10" to stop the cycle in case somehow f_a and f_b are on the same side
                            break
                    fp_back2 = f_a
                    print("\rBack  edge of the second part of the footprint found, location = {:.0f} km relative to ILP.".format((f_a - mrange)/1000))

                    if plot_hit_charts and i_type :
                        if f_a == dist_limit : #int_max_range + mrange :
                            fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                        else :
                            pass
                            fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                #if f_ok
            # if not f_ok
        # if not fp_single

        """
        # function to calculate footprint part (works for both parts)           
        def fp_part_calc(fp_front, fp_back) : 

            #v1 -- calculates footprint border point (omega) for each shift, and connects them,
            #not good for big gaps between omegas for adjacent shifts


            fp_part = np.array([[0, fp_front, 0, 0, (fp_front - mrange)/1000]])

            t_ang_beg = 0
            t_ang_end = pi/2
            angle_step_r = radians(angle_step)
            stop_ang_par = stop_par / mrange

            shift_beg = fp_front
            shift_end = fp_back
            shift_step = (shift_end - shift_beg) / num_dist

            f_shift = shift_beg + shift_step

            stm2 = time.process_time()
            f_x  = pi/16
            while f_shift >= shift_end - shift_step - .1 :
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= out_time_freq :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_x0 = t_ang_beg
                f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                while _is_hit(f_ok) :             
                    f_x0 = f_x
                    if f_x >= t_ang_end :
                        break
                    f_x *= 2
                    f_x = min(f_x, t_ang_end)
                    f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                do_search = not _is_hit(f_ok) # == True only if f_x = pi/2, which shouldn't be impossible
                f_a = f_x0
                f_b = f_x

                while do_search :
                    f_x = f_a + (f_b - f_a) / 2
                    f_xy = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    if _is_hit(f_xy) :
                        f_a = f_x
                    else :
                        f_b = f_x
                    if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) : # "<stop_ang_par" to stop the cycle in case somehow f_a and f_b are on the same side
                        break

                f_x = f_b
                if abs(f_a) <= angle_eps :
                    x1 = 0
                    y1 = (f_shift - mrange) / 1000
                else :
                    f_delta_x = asin(sin(f_a) * sin(f_alfa))
                    f_gamma_y = asin(tan(f_delta_x) / tan(f_a))
                    x1 = - f_delta_x * R_e / 1000
                    y1 = (f_shift - f_gamma_y * R_e) / 1000

                fp_part = np.append(fp_part, [[f_a, f_shift, f_b - f_a, x1, y1]], axis=0)

                f_shift += shift_step
            pass #while

            # filling the back edge of the foot
            fp_tmp = np.empty([0, 5])
            f_ang = angle_step_r           
            while f_ang < 90 :
                f_ang_r = radians(f_ang)
                #print("\rProcessing hindmost line at angle {}".format(f_ang), end='')
                f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_ang_r, shift_end, op_range, det_range, t_delay, h_discr, concentric=concentric)
                if _is_hit(f_ok) :
                    if abs(f_ang_r) <= angle_eps :
                        x1 = 0
                        y1 = (shift_end - mrange) / 1000
                    else :
                        f_delta_x = asin(sin(f_ang_r) * sin(f_alfa))
                        f_gamma_y = asin(tan(f_delta_x) / tan(f_ang_r))
                        x1 = - f_delta_x * R_e / 1000
                        y1 = (shift_end - f_gamma_y * R_e) / 1000

                    fp_tmp = np.append(fp_tmp, [[f_ang_r, shift_end, 0, x1, y1]], axis=0)
                else :
                    break
                f_ang += angle_step_r
            # while

            fp_tmp_len = len(fp_tmp)
            for i in range(fp_tmp_len) :
                fp_part = np.append(fp_part, [fp_tmp[fp_tmp_len - 1 - i]], axis=0)

            fp_part = np.append(fp_part, [[0, fp_back, 0, 0, (fp_back - mrange)/1000]], axis=0)

            print("\rProcessing distance complete", end='')

            len_fp_part = len(fp_part)
            if len_fp_part > 0 :   
                for i_f in range (len_fp_part - 1) :
                    fp_part_if = fp_part[len_fp_part - 2 - i_f]
                    fp_part = np.append(fp_part, [fp_part_if], axis=0)
                    fp_part[len_fp_part + i_f, 0] = - fp_part_if[0]
                    fp_part[len_fp_part + i_f, 3] = - fp_part_if[3]
                # for
            else :
                print("\nFootprint not calculated, check input data")
                fp_part = np.zeros((0, 5))

            return fp_part
        """
        """ end of fp_part_calc """

        """
        def fp_part_calc_adv(fp_front, fp_back) : # v2
            """""""
            v2 -- calculates footprint border point (omega) for each shift, as well as 
            all points from omega of previous shift to the current omega, spaced by angle_step
            not very good since creates "stairs" close to omegas
            """""""

            #fp_part = np.array([[0, fp_front, 0, 0, (fp_front - mrange)/1000]])
            fp_part = np.empty([0, 5])

            t_ang_beg = 0
            t_ang_end = pi/2
            stop_ang_par = stop_par / mrange # stop angle (search) parameter

            shift_beg = fp_front
            shift_end = fp_back
            shift_step = (shift_end - shift_beg) / num_dist

            f_shift = shift_beg

            stm2 = time.process_time()
            f_x0 = t_ang_beg
            angle_step_r = radians(angle_step)
            f_x  = t_ang_beg
            count = 0 # so that the first point of the footprint is recorded
            count_rate = 3

            while f_shift >= shift_end - .1 :
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= out_time_freq :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                x, y = omegashift_2_xy(f_x, f_shift, mrange)
                #print("\rf_x = {:.2f} degr f_shift = {:.2f} km x = {:.2f} km y = {:.2f} km".format(degrees(f_x), (f_shift)/1000, x, y))
                if _is_hit(f_ok) :
                    while _is_hit(f_ok) :
                        if count % count_rate == 0  : # record every 5th 
                            x, y = omegashift_2_xy(f_x, f_shift, mrange)
                            fp_part = np.append(fp_part, [[f_x, f_shift, 0, x, y]], axis=0)
                            count = 1
                        else :
                            count += 1
                        f_x0 = f_x
                        if f_x >= t_ang_end :
                            break
                        f_x += angle_step_r
                        f_x = min(f_x, t_ang_end)
                        f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    #while
                    do_search = not _is_hit(f_ok)
                    f_a = f_x0
                    f_b = f_x

                    while do_search :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                        if _is_hit(f_xy) :
                            f_a = f_x
                        else :
                            f_b = f_x
                        if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) : # "<stop_ang_par" to stop the cycle in case somehow f_a and f_b are on the same side
                            break
                    #while
                    f_x = f_a
                    x, y = omegashift_2_xy(f_a, f_shift, mrange)                
                    fp_part = np.append(fp_part, [[f_a, f_shift, f_b - f_a, x, y]], axis=0)

                else : # f_x not in footprint at this shift
                    while not _is_hit(f_ok) :
                        if count % count_rate == 0 :
                            x, y = omegashift_2_xy(f_x, f_shift - shift_step, mrange)
                            #this needs to be previous f_shift, i.e. f_shift - shift_step
                            fp_part = np.append(fp_part, [[f_x, f_shift - shift_step, 0, x, y]], axis=0)
                            #print("\rf_x = {:.2f} degr f_shift - shift_step = {:.2f} km x = {:.2f} km y = {:.2f} km".format(degrees(f_x), (f_shift-shift_step)/1000, x, y))
                        else :
                            count += 1
                        f_x0 = f_x
                        if f_x <= t_ang_beg :
                            break
                        f_x -= angle_step_r
                        f_x = max(f_x, t_ang_beg)
                        f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    #while not f_ok

                    do_search = _is_hit(f_ok)
                    f_b = f_x0
                    f_a = f_x

                    while do_search :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                        if _is_hit(f_xy) :
                            f_a = f_x
                        else :
                            f_b = f_x
                        if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) : # "<stop_ang_par" to stop the cycle in case somehow f_a and f_b are on the same side
                            break
                    #while

                    f_x = f_a
                    x, y = omegashift_2_xy(f_a, f_shift, mrange)                
                    fp_part = np.append(fp_part, [[f_a, f_shift, f_b - f_a, x, y]], axis=0)
                #else
                count = 1
                f_shift += shift_step
            pass #while f_shift

            # filling the back edge of the foot
            fp_tmp = np.empty([0, 5])
            f_ang = angle_step           
            while f_ang < 90 :
                f_ang_r = radians(f_ang)
                #print("\rProcessing hindmost line at angle {}".format(f_ang), end='')
                f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_ang_r, shift_end, op_range, det_range, t_delay, h_discr, concentric=concentric)
                if _is_hit(f_ok) :
                    x, y = omegashift_2_xy(f_ang_r, shift_end, mrange)
                    fp_tmp = np.append(fp_tmp, [[f_ang_r, shift_end, 0, x, y]], axis=0)
                else :
                    break
                f_ang += angle_step
            # while

            fp_tmp_len = len(fp_tmp)
            for i in range(fp_tmp_len) :
                fp_part = np.append(fp_part, [fp_tmp[fp_tmp_len - 1 - i]], axis=0)

            fp_part = np.append(fp_part, [[0, fp_back, 0, 0, (fp_back - mrange)/1000]], axis=0)

            print("\rProcessing distance complete", end='')

            len_fp_part = len(fp_part)
            if len_fp_part > 0 :   
                for i_f in range (len_fp_part - 1) :
                    fp_part_if = fp_part[len_fp_part - 2 - i_f]
                    fp_part = np.append(fp_part, [fp_part_if], axis=0)
                    fp_part[len_fp_part + i_f, 0] = - fp_part_if[0]
                    fp_part[len_fp_part + i_f, 3] = - fp_part_if[3]
                # for
            else :
                print("\nFootprint not calculated, check input data")
                fp_part = np.zeros((0, 5))

            return fp_part
        """
        """ end of fp_part_calc_adv """

        def fp_part_calc_v3(fp_front, fp_back, num_steps=num_dist) :
            """
            v3 -- calculates footprint border point (omega) for each shift, as well as 
            all points from omega of previous shift to the current omega, spaced by angle_step
            and intrapolating shift value for those points -- creates a smooth line
            """

            fp_part = np.array([[0, fp_front, 0, 0, (fp_front - mrange)/1000]])
            #fp_part = np.empty([0, 5])

            t_ang_beg = 0
            t_ang_end = pi/2#*3/4 #/2 + line 43 above (if fi_opt > pi/2 :)
            stop_ang_par = stop_par / mrange # stop angle (search) parameter

            shift_beg = fp_front
            shift_end = fp_back
            shift_step = (shift_end - shift_beg) / num_steps
            #print("fp_front = {:.2f} km fp_back = {:.2f} km shift_step = {:.2f} km".format(fp_front/1000, fp_back/1000, shift_step/1000))

            f_shift = shift_beg

            stm2 = time.process_time()
            #f_x0 = t_ang_beg
            #angle_step_r = radians(angle_step)
            f_x  = t_ang_beg
            f_x_keep = t_ang_beg
            ang_quant_r = radians(min(angle_step, ang_quant))

            max_dist_line = False

            while f_shift - shift_end >= 0 : 
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= out_time_freq :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                #x, y = omegashift_2_xy(f_x, f_shift, mrange)
                #print("\rf_x = {:.2f} degr f_shift = {:.2f} km x = {:.2f} km y = {:.2f} km".format(degrees(f_x), (f_shift)/1000, x, y))
                if _is_hit(f_ok) :
                    """
                    while False : #f_ok :
                        f_x0 = f_x
                        if f_x >= t_ang_end :
                            break
                        f_x += angle_step_r
                        f_x = min(f_x, t_ang_end)
                        f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    #while
                    """
                    #do_search = not _is_hit(f_ok)
                    #f_a = f_x0
                    #f_b = f_x

                    f_a = f_x
                    f_b = t_ang_end

                    while True : #do_search :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                        #f_x_d = degrees(f_x) # for debugging
                        if _is_hit(f_xy) :
                            f_a = f_x
                        else :
                            f_b = f_x
                        if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) : # "<stop_ang_par" to stop the cycle in case somehow f_a and f_b are on the same side
                            #print("f_b = {} f_a = {}".format(f_b, f_a))
                            break
                    #while
                    fill_line = True
                    if f_b == t_ang_end :
                        if max_dist_line : # at least second successive point at max dist => put the zigzag between them
                            fill_line = False
                            #print(f_shift, degrees(f_b), degrees(f_a))
                            f_shift_z = f_shift - shift_step/2
                            x, y = omegashift_2_xy(f_a, f_shift_z, mrange * mrl_factor2)                
                            #fp_part = np.append(fp_part, [[f_a, f_shift_z, f_b - f_a, x, y]], axis=0)
                            fp_part = np.append(fp_part, [[f_a, f_shift_z, 2, x, y]], axis=0)
                            x, y = omegashift_2_xy(f_a, f_shift, mrange * mrl_factor1)
                            #fp_part = np.append(fp_part, [[f_a, f_shift, f_b - f_a, x, y]], axis=0)
                            fp_part = np.append(fp_part, [[f_a, f_shift, 3, x, y]], axis=0)
                        else :
                            max_dist_line = True # first successive point at max dist
                    else :
                        max_dist_line = False

                    if fill_line :    # fill line between points, which are not at max dist            
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
                            x, y = omegashift_2_xy(f_x_i, i_shift, mrange)                
                            if i == f_x_num - 1 :
                                fp_part = np.append(fp_part, [[f_x_i, i_shift, f_b - f_a, x, y]], axis=0) # real point
                            else :
                                fp_part = np.append(fp_part, [[f_x_i, i_shift, 1, x, y]], axis=0)         # intrapolated points (except for the points at fp_front)

                    f_x_keep = f_a
                    f_x = f_a                    

                else : # f_x not in footprint at this shift
                    """
                    while False : #not f_ok :
                        #f_x0 = f_x
                        if f_x <= t_ang_beg :
                            break
                        f_x -= angle_step_r
                        f_x = max(f_x, t_ang_beg)
                        f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    #while not f_ok
                    """
                    #do_search = _is_hit(f_ok)
                    #f_b = f_x0
                    #f_a = f_x

                    f_b = f_x
                    f_a = t_ang_beg

                    while True : #do_search :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                        if _is_hit(f_xy) :
                            f_a = f_x
                        else :
                            f_b = f_x
                        if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) : # "<stop_ang_par" to stop the cycle in case somehow f_a and f_b are on the same side
                            break
                    #while
                    f_x_num =  round(abs(f_a - f_x_keep) / ang_quant_r)
                    if not f_x_num :
                        f_x_num = 1
                    f_x_step = (f_a - f_x_keep) / f_x_num
                    for i in range(f_x_num) :
                        f_x_i = f_x_keep + (i + 1) * f_x_step
                        i_shift = f_shift - shift_step + shift_step * (i + 1) / f_x_num
                        x, y = omegashift_2_xy(f_x_i, i_shift, mrange)
                        if i == f_x_num - 1 :
                            fp_part = np.append(fp_part, [[f_x_i, f_shift, f_b - f_a, x, y]], axis=0) # real point
                        else :
                            fp_part = np.append(fp_part, [[f_x_i, i_shift, 1, x, y]], axis=0)         # intrapolated points

                    f_x_keep = f_a
                    f_x = f_a
                #else
                if f_shift > shift_end :
                    f_shift += shift_step
                    if f_shift < shift_end :
                        f_shift = shift_end
                else :
                    f_shift += shift_step

            pass #while f_shift

            # filling back edge of the footprint
            max_dist_line = False
            f_a = t_ang_beg # =0
            f_shift = shift_end
            if abs(f_shift - shift_min) < shift_min :
                ang_quant_r *= 2
                max_dist_line = True
            f_x_num =  round(abs(f_a - f_x_keep) / ang_quant_r)
            if not f_x_num :
                f_x_num = 1
            f_x_step = (f_a - f_x_keep) / f_x_num
            for i in range(0, f_x_num) :
                if max_dist_line :
                    f_x_i = f_x_keep + (i + 1/2) * f_x_step
                    x, y = omegashift_2_xy(f_x_i, shift_end, mrange * mrl_factor2)                
                    fp_part = np.append(fp_part, [[f_x_i, shift_end, 2, x, y]], axis=0)

                    f_x_i = f_x_keep + (i + 1) * f_x_step
                    x, y = omegashift_2_xy(f_x_i, shift_end, mrange * mrl_factor1)                
                    fp_part = np.append(fp_part, [[f_x_i, shift_end, 3, x, y]], axis=0)
                else :
                    f_x_i = f_x_keep + (i + 1) * f_x_step
                    x, y = omegashift_2_xy(f_x_i, shift_end, mrange)                
                    fp_part = np.append(fp_part, [[f_x_i, shift_end, 1, x, y]], axis=0)

            """
            fp_tmp = np.empty([0, 5])
            f_ang = angle_step           
            while f_ang <  :
                f_ang_r = radians(f_ang)
                #print("\rProcessing hindmost line at angle {}".format(f_ang), end='')
                f_ok = s_func(t_trj, active_table, h_min, t_lnc, f_ang_r, shift_end, op_range, det_range, t_delay, h_discr, concentric=concentric)
                if _is_hit(f_ok) :
                    x, y = omegashift_2_xy(f_ang_r, shift_end, mrange)
                    fp_tmp = np.append(fp_tmp, [[f_ang_r, shift_end, 0, x, y]], axis=0)
                else :
                    break
                f_ang += angle_step
            # while

            fp_tmp_len = len(fp_tmp)
            for i in range(fp_tmp_len) :
                fp_part = np.append(fp_part, [fp_tmp[fp_tmp_len - 1 - i]], axis=0)

            fp_part = np.append(fp_part, [[0, fp_back, 0, 0, (fp_back - mrange)/1000]], axis=0)
            """
            print("\rProcessing distance complete", end='')

            len_fp_part = len(fp_part)
            if len_fp_part > 0 :   
                for i_f in range (len_fp_part - 1) :
                    fp_part_if = fp_part[len_fp_part - 2 - i_f]
                    fp_part = np.append(fp_part, [fp_part_if], axis=0)
                    fp_part[len_fp_part + i_f, 0] = - fp_part_if[0]
                    fp_part[len_fp_part + i_f, 3] = - fp_part_if[3]
                # for
            else :
                print("\nFootprint not calculated, check input data")
                fp_part = np.zeros((0, 5))

            return fp_part
        """ end of fp_part_calc_v3 """

        if fp_single :
            fp_part2 = np.zeros((0, 5))
            #fp_part2 = []
            print("Calculating the footprint", end='')
        else :
            print("Calculating the first part of the footprint", end='')

        if fp_calc_param == 3 :
            fp_part1 = fp_part_calc_v3(fp_front1, fp_back1)
        """
        elif fp_calc_param == 1 :
            fp_part1 = fp_part_calc(fp_front1, fp_back1)
        elif fp_calc_param == 2 :
            fp_part1 = fp_part_calc_adv(fp_front1, fp_back1)
        else :
            print("\nFootprint mode2 calculation procedure not set")
            return False
        """    
        if not fp_single :
            print("\rCalculating the second (back) part of the footprint", end='')
            if fp_calc_param == 3 :
                #n_steps = max(round(num_dist * (fp_back2 - fp_front2) / (fp_back1 - fp_front1)), 20)
                fp_part2 = fp_part_calc_v3(fp_front2, fp_back2) #, n_steps)
            """
            elif fp_calc_param == 1 :
                fp_part2 = fp_part_calc(fp_front2, fp_back2)
            else :
                fp_part2 = fp_part_calc_adv(fp_front2, fp_back2)
            """


        print("\rFootprint calculation complete")

        return [fp_part1, fp_part2]

    fp_part_std = _compute_part(t_itable, False)

    if t_itable_con is not None :
        fp_part_con = _compute_part(t_itable_con, True)
    else :
        fp_part_con = [np.zeros((0, 5)), np.zeros((0, 5))]

    return [fp_part_std, fp_part_con]

""" End of footprint_mode2 """