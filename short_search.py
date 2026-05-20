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
                 concentric=False):
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

#    beta_n = 360
    
    """ Sampling rate for missile trajectory resampling, must be the same as int_table sampling rate $$$"""
    #beta_step = 0.025 #0.25

    # dist_opt is the metric distance between ILP and MLP (in meters)
    # fi_opt is the flight path plane aspect angle (phi)
    psi_d = degrees(dist_opt / R_e)
    phi_d = degrees(fi_opt)

    if concentric == True:
        mis_traj = tb_con.mis_traj_sample(tar_traj, psi_deg=psi_d, phi_deg=phi_d)
    else:
        mis_traj = tb.mis_traj_sample(tar_traj)
        
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
    max_valid_index = len(int_table) - 1 # NEU: Letzter gültiger Index im Array

    if (int_table[int_beta_il][0, 3] > m_beta_max) : # (beta_il - angle_eps)) : <- ???
        if debug_print : 
            print("All of the possibly interceptable missile trajectory is below lowest interceptor beta")
        return False
    
    # KORREKTUR 1: Zusätzliche Prüfung (int_beta_il < max_valid_index) direkt in der while-Bedingung
    while int_beta_il < max_valid_index and int_table[int_beta_il][0, 3] < (beta_il - angle_eps) : # TODO optimize search $$$
        int_beta_il += 1
        
        # KORREKTUR 2: Sicherheitsprüfung, falls wir trotz allem die Grenze sprengen
        if int_beta_il > max_valid_index:
            int_beta_il = max_valid_index
            break

        if not np.any(int_table[int_beta_il]) : # to account for empty betas near vertical
            if debug_print :
                print("!!!", end='')
            if abs(pi/2 -  beta_il) < angle_eps :
                int_beta_il = max_valid_index # KORREKTUR 3: Nutzt die dynamische Variable
                break
            else:
                m_beg += 1
                if m_beg >= len(mis_traj): # KORREKTUR 4: Zusätzliche Absicherung für die Raketenbahn
                    break
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
                  t_delay=0, h_discr=0, debug_hits=False, concentric=False) :
    return short_search(trj, int_table, h_int_min, maxia, t_int_lnc, omega, shift, op_range, det_range, 
                        t_delay, h_discr, True, debug_hits, False, concentric)


def sls_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, 
               t_delay=0, h_discr=0, mode2=False, debug_hits=False, 
               concentric=False) :
    """
    Finds interception points for shoot-look-shoot mode. First, runs short_search, then takes missile time at the first interception point
    and runs short_search again passing missile time instead of t_int_lnc. Returns both interception data points to check if this works.
    Both will not be needed when function is confirmed to be good.
    This is a valid procedure because short_search finds the earliest interception point, so if there is no second hit for the first
    found first hit, there won't be any for later ones.
    """

    hit_1 = short_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, t_delay, h_discr, mode2, debug_hits, False, concentric)
    
    if hit_1 :
        t_int_lnc = hit_1[1][0]
        hit_2 = short_search(trj, int_table, h_int_min, maxia, t_int_lnc, fi_opt, dist_opt, op_range, det_range, t_delay, h_discr, mode2, debug_hits, True, concentric)
        if hit_2 :
            sls_hit = hit_2
        else :
            sls_hit = False        
    else :
        sls_hit = False
        
    return sls_hit


def sls_search2(trj, int_table, h_int_min, maxia, t_int_lnc, omega, shift, op_range, det_range, 
                t_delay=0, h_discr=0, debug_hits=False, concentric=False) :
    return sls_search(trj, int_table, h_int_min, maxia, t_int_lnc, omega, shift, op_range, det_range, 
                      t_delay, h_discr, True, debug_hits, concentric)


                    
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
        beta_min = active_table - radians(beta_step)
        int_max_range = -2 * beta_min * R_e
        print("\n[{}] beta_min={:.2f}, int_max_range={:.1f} km".format(label_str, degrees(beta_min), int_max_range/1000))

        ilp_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, mrange, op_range, det_range, t_delay, h_discr)
        if ilp_ok :
            f_dist = min(int_max_range / 8 + mrange, dist_limit)
            f_dist0 = mrange
            f_ok = s_func(t_trj, active_table, h_min, maxia, t_lnc, 0, f_dist, op_range, det_range, t_delay, h_discr)
            
            while f_ok :             
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
                if f_xy :
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

def footprint_mode2(s_func, # s_func needs to be either short_search2 or sls_search2
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

    fp_calc_param = 3
    stop_par = 10
    is_compare_mode = t_itable_con is not None

    def fp_part_calc_v3(fp_front, fp_back, num_steps=num_dist, concentric=False) :
        """
        v3 -- calculates footprint border point (omega) for each shift, as well as 
        all points from omega of previous shift to the current omega, spaced by angle_step
        and intrapolating shift value for those points -- creates a smooth line
        """
        active_table = t_itable_con if concentric else t_itable

        fp_part = np.array([[0, fp_front, 0, 0, (fp_front - mrange)/1000]])
    
        t_ang_beg = 0
        t_ang_end = pi/2
        stop_ang_par = stop_par / mrange # stop angle (search) parameter
        
        shift_beg = fp_front
        shift_end = fp_back
        shift_step = (shift_end - shift_beg) / num_steps
        
        f_shift = shift_beg

        stm2 = time.process_time()
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
            if f_ok :
                f_a = f_x
                f_b = t_ang_end

                while True :
                    f_x = f_a + (f_b - f_a) / 2
                    f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    if f_xy :
                        f_a = f_x
                    else :
                        f_b = f_x
                    if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) :
                        break
                fill_line = True
                if f_b == t_ang_end :
                    if max_dist_line : 
                        fill_line = False
                        f_shift_z = f_shift - shift_step/2
                        x, y = omegashift_2_xy(f_a, f_shift_z, mrange * mrl_factor2)                
                        fp_part = np.append(fp_part, [[f_a, f_shift_z, 2, x, y]], axis=0)
                        x, y = omegashift_2_xy(f_a, f_shift, mrange * mrl_factor1)
                        fp_part = np.append(fp_part, [[f_a, f_shift, 3, x, y]], axis=0)
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
                        x, y = omegashift_2_xy(f_x_i, i_shift, mrange)                
                        if i == f_x_num - 1 :
                            fp_part = np.append(fp_part, [[f_x_i, i_shift, f_b - f_a, x, y]], axis=0) 
                        else :
                            fp_part = np.append(fp_part, [[f_x_i, i_shift, 1, x, y]], axis=0)         

                f_x_keep = f_a
                f_x = f_a                    

            else : 
                f_b = f_x
                f_a = t_ang_beg

                while True : 
                    f_x = f_a + (f_b - f_a) / 2
                    f_xy = s_func(t_trj, active_table, h_min, maxia, t_lnc, f_x, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric)
                    if f_xy :
                        f_a = f_x
                    else :
                        f_b = f_x
                    if (abs(f_b - f_a) < f_b  * t_acc) or (abs(f_b - f_a) < stop_ang_par) : 
                        break
                f_x_num =  round(abs(f_a - f_x_keep) / ang_quant_r)
                if not f_x_num :
                    f_x_num = 1
                f_x_step = (f_a - f_x_keep) / f_x_num
                for i in range(f_x_num) :
                    f_x_i = f_x_keep + (i + 1) * f_x_step
                    i_shift = f_shift - shift_step + shift_step * (i + 1) / f_x_num
                    x, y = omegashift_2_xy(f_x_i, i_shift, mrange)
                    if i == f_x_num - 1 :
                        fp_part = np.append(fp_part, [[f_x_i, f_shift, f_b - f_a, x, y]], axis=0) 
                    else :
                        fp_part = np.append(fp_part, [[f_x_i, i_shift, 1, x, y]], axis=0)         

                f_x_keep = f_a
                f_x = f_a

            if f_shift > shift_end :
                f_shift += shift_step
                if f_shift < shift_end :
                    f_shift = shift_end
            else :
                f_shift += shift_step

        pass 
        
        # filling back edge of the footprint
        max_dist_line = False
        f_a = t_ang_beg 
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

        print("\rProcessing distance complete", end='')

        len_fp_part = len(fp_part)
        if len_fp_part > 0 :   
            for i_f in range (len_fp_part - 1) :
                fp_part_if = fp_part[len_fp_part - 2 - i_f]
                fp_part = np.append(fp_part, [fp_part_if], axis=0)
                fp_part[len_fp_part + i_f, 0] = - fp_part_if[0]
                fp_part[len_fp_part + i_f, 3] = - fp_part_if[3]
        else :
            print("\nFootprint not calculated, check input data")
            fp_part = np.zeros((0, 5))

        return fp_part
    """ end of fp_part_calc_v3 """


    # --- EXECUTION PIPELINE ROUTER ---
    def run_footprint_pipeline(current_table, concentric_flag):
        beta_min = current_table[0][0, 3] - radians(beta_step)
        int_max_range = -2 * beta_min * R_e 
        print("beta_min={:.2f}, int_max_range={:.1f} km (Concentric={})".format(degrees(beta_min), int_max_range/1000, concentric_flag))

        f_alfa = t_trj[len(t_trj) - 1, 2]
        global mrange
        mrange = f_alfa * R_e
        fp_single = False   
        dist_step = (shift_min - mrange) / num_dist 
        
        ilp_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, mrange, op_range, det_range, t_delay, h_discr, concentric=concentric_flag) 
        if not ilp_ok : 
            print("Interceptor position undefendable. Attempting to find a footprint behind(?) it...")
            shift_beg = min( (mrange + int_max_range), dist_limit)
            shift_end = shift_min 
            dist_step = (shift_end - shift_beg) / num_dist 
            
            stm2 = time.process_time()
            f_shift = shift_beg 
            f_b = f_shift
            while f_shift >= shift_end - .1 :
                enm2 = time.process_time()
                dum2 = enm2 - stm2
                if dum2 >= 1 :
                    print("\rProcessing distance {:6.0f} km...   ".format((f_shift - mrange)/1000), end='')
                    stm2 = enm2

                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                if f_ok :
                    f_b = f_shift
                    f_a = f_shift - dist_step
                    break
                f_shift += dist_step
            if f_b > mrange :
                print("\nFootprint in front of ILP", end='')

            print("\rProcessing distance complete", end='')

            if not f_ok :
                print('\rNo footprint found behind undefendable interceptor position.\nIncreasing "Mode 2 Footprint number of steps" might help to find one.')
                fp_part = np.zeros((0, 5))
                return [fp_part, fp_part]
           
        else : 
            f_shift = mrange + int_max_range / 8 
            f_shift0 = mrange
            f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
            
            while f_ok :             
                f_shift0 = f_shift
                if f_shift >= int_max_range + mrange :
                    break
                f_shift = f_shift * 2 - mrange
                f_shift = min(f_shift, int_max_range + mrange)
                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_shift, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)

            if True : 
                f_b = f_shift0
                f_a = f_shift
            else :
                print("\nProbing2 exception (a): f_shift = {:.2f} >= int_max_range + mrange = {:.2f}".format(f_shift, int_max_range + mrange))
                return False
                
        while True :
            f_x = f_a + (f_b - f_a) / 2
            f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
            if f_xy :
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
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                    
        else :
            print("\nProbing2 exception (b): f_a = {:.2f} > int_max_range = {:.2f}".format(f_a/1000, int_max_range/1000))
            return False
            
        fp_front1 = f_b
        print("\rFront edge of the footprint found, location = {:.0f} km relative to ILP, looking for the back edge...".format((f_b - mrange)/1000))
        
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
            if not f_ok :
                f_b = f_shift
                f_a = f_shift - dist_step
                break
            
            if f_shift > shift_end :
                f_shift += dist_step
                if f_shift < shift_end :
                    f_shift = shift_end
            else :
                f_shift += dist_step

        print("\rProcessing distance complete", end='')

        if f_ok :
            print("\rFootprint's back edge is at the maximum missile range. Single part footprint.")
            fp_back1 = shift_end
            fp_single = True
        else :
            while True :
                f_x = f_a + (f_b - f_a) / 2
                f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                if f_xy :
                    f_a = f_x
                    f_y_keep = f_xy
                else :
                    f_b = f_x
                if ((f_a - f_b) < (f_a - mrange) * t_acc) or ((f_a - f_b) < stop_par) : 
                    break
            fp_back1 = f_a
            print("\rBack  edge of the footprint found, location = {:.0f} km relative to ILP, looking for the footprint's second part...".format((f_a - mrange)/1000))

            if plot_hit_charts and i_type :
                if f_a == dist_limit : 
                    fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                else :
                    fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                    
        """ Search for the second part of the footprint """
        if not fp_single :
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
                if f_ok :
                    f_b = f_shift
                    f_a = f_shift - dist_step
                    break
                f_shift += dist_step
            print("\rProcessing distance complete", end='')

            if not f_ok :
                print('\rNo second part of footprint found behind the first part.\nIncreasing "Mode 2 Footprint number of steps" might help to find one.')
                fp_single = True
            else :
                while True :
                    f_x = f_a + (f_b - f_a) / 2
                    f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                    if f_xy :
                        f_b = f_x
                        f_y_keep = f_xy
                    else :
                        f_a = f_x
                    if (abs(f_a - f_b) < abs(f_b - mrange) * t_acc) or (abs(f_a - f_b) < stop_par) : 
                        break
                fp_front2 = f_b
                print("\rFront edge of the second part of the footprint found, location = {:.0f} km relative to ILP, looking for the back edge...".format((f_b - mrange)/1000))

                if plot_hit_charts and i_type :
                    if f_a == dist_limit : 
                        fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                    else :
                        fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)

                shift_beg = f_b
                shift_end = shift_min 
                            
                f_ok = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, shift_end, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)        
                if f_ok :
                    print("\nBack edge of the footprint's second part is at the maximum missile range")
                    fp_back2 = shift_end
                else :
                    f_a = shift_beg
                    f_b = shift_end
                    while True :
                        f_x = f_a + (f_b - f_a) / 2
                        f_xy = s_func(t_trj, current_table, h_min, maxia, t_lnc, 0, f_x, op_range, det_range, t_delay, h_discr, concentric=concentric_flag)
                        if f_xy :
                            f_a = f_x
                            f_y_keep = f_xy
                        else :
                            f_b = f_x
                        if ((f_a - f_b) < (f_a - mrange) * t_acc) or ((f_a - f_b) < stop_par) : 
                            break
                    fp_back2 = f_a
                    print("\rBack  edge of the second part of the footprint found, location = {:.0f} km relative to ILP.".format((f_a - mrange)/1000))

                    if plot_hit_charts and i_type :
                        if f_a == dist_limit : 
                            fp.plot_hit2(f_ok, t_trj, m_type, i_type, 0, f_a, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)
                        else :
                            fp.plot_hit2(f_y_keep, t_trj, m_type, i_type, 0, f_b, int_max_range/1000, op_range, det_range, t_lnc - t_delay, h_min/1000, h_discr/1000)

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


    fp_part_std = run_footprint_pipeline(t_itable, concentric_flag=False)

    if is_compare_mode:
        fp_part_con = run_footprint_pipeline(t_itable_con, concentric_flag=True)
        return [fp_part_std, fp_part_con]
    else:
        return [fp_part_std, np.zeros((0, 5))]

""" End of footprint_mode2 """