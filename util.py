import numpy as np
import matplotlib.pyplot as plt
from constants import *

_IE24  = {'a': 6378388.0, 'invf': 297.0}
_IE67  = {'a': 6378160.0, 'invf': 298.247}
_WGS72 = {'a': 6378135.0, 'invf': 298.26}
_GRS80 = {'a': 6378137.0, 'invf': 298.257222101}
_WGS84 = {'a': 6378137.0, 'invf': 298.257223563}

#def preprocessor(agent_pos, sat_pos, residuals, delta_t, tm, offset):
def preprocessor(simEnv, residuals, tm, offset):
    """
    Input: simEnv -- TrainSimEnv object, 
              residuals -- list of Nsam residuals each array of size Nsat x Nm,
              tm -- time between measurements
              offset -- number of measurement sets to skip (Ndt measurements to skip)
    Output: a list of Nsam each is an array of pseudorange (Nsat x Nm)
    """

    agent_pos = simEnv.pos # true position (3 x (Ndt*Ntraj))
    sat_pos = simEnv.sat_pos # list of Nsat true sat positions (3 x (Ndt*Ntraj))
    delta_t = simEnv.delta_t # time between predict steps (each element in positions)
    Ndt = simEnv.Ndt
    Nsat = simEnv.Nsat

    iter_per_meas = int(tm / delta_t)
    if len(residuals) == 0:
       return []
    #Nsat = residuals[0].shape[0]
    Nm = residuals[0].shape[1]
    Nsam = len(residuals)

    # print(Nm)
    # Create initial array
    pseudoranges = [None]*Nsam
    
    for i_sam in range(Nsam):
        #print('i_sam', i_sam)
        curr_pseudoranges = np.nan * np.ones([Nsat, Nm])
        for i_meas in range(Nm):
            # print('i_meas', i_meas)
            # curr_agent_pos = agent_pos[:, offset*Ndt + i_meas*iter_per_meas]
            curr_agent_pos = agent_pos[:, (offset+1)*Ndt + i_meas*iter_per_meas - 1]
            for i_sat in range(Nsat):
                #print('  i_sat', i_sat)
                #   curr_sat_pos = sat_pos[i_sat][:, offset*Ndt + i_meas*iter_per_meas]
                curr_sat_pos = sat_pos[i_sat][:, (offset+1)*Ndt + i_meas*iter_per_meas - 1]
                true_range = np.linalg.norm(curr_agent_pos - curr_sat_pos)
                curr_res = residuals[i_sam][i_sat,i_meas]
                curr_pseudoranges[i_sat,i_meas] = curr_res + true_range
        pseudoranges[i_sam] = curr_pseudoranges

    return pseudoranges


def read_file(filename, label):
    # label = "v" for velocity or 
    #         "p" for trajectory or 
    #         "s" for satellite or
    #         "c" for covariance or
    #         "r" for r_ekf
    
    with open(filename, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Validate label_col argument
    allowed_label_cols = ('p', 'v', 's', 'c', 'r')
    if label not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label, allowed_label_cols))

    # Load stuff

    if label == 'v':
        v_cols = [i for i in range(len(headers)) if headers[i].startswith('v')]
        vels = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=v_cols)
        return vels.transpose()

    if label == 'p':
        x_cols = [i for i in range(len(headers)) if headers[i].startswith('p')]
        input = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=x_cols)
        return input

    if label == 's':
        s_cols = [i for i in range(len(headers)) if headers[i].startswith('s')]
        sats = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=s_cols)
        sat_pos_init = [sats[i,:] for i in range((np.shape(sats))[0])]
        return sat_pos_init

    if label == 'c':
        c_cols = [i for i in range(len(headers)) if headers[i].startswith('c')]
        cov_init = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=c_cols)
        return cov_init

    if label == 'r':
        r_cols = [i for i in range(len(headers)) if headers[i].startswith('r')]
        r_ekf = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=r_cols)
        return r_ekf

def simple_RAIM_1sample(residuals, threshold, sigma_range, simEnv, p, tm, offset):
    """
    Input: residuals -- array of measurement residuals for current position/sample (Nsat, Nm)
            threshold -- RAIM threshold for fault presence (scalar)
            sigma_range -- standard deviation for range measurements (scalar)
            simEnv -- simulation environment instance from TrainSimEnv
            p -- position estimates for each measurement update (3,Nm)
            tm -- time between measurements (s)
            offset -- number of measurement sets to skip (Ndt measurements to skip)
    Output: RAIM_alert -- binary values indicating whether RAIM threshold exceeded
                            for any measurement (alert -- True) or not (no alert -- False)
    """
    sat_pos = simEnv.sat_pos
    Nm = residuals.shape[1]
    Nsat = simEnv.Nsat
    Ndt = simEnv.Ndt
    delta_t = simEnv.delta_t   # time between predict steps (each element in positions)
    iter_per_meas = int(tm / delta_t)

    inv_var = 1/(sigma_range*sigma_range)
    W = inv_var*np.eye(Nsat)

    for i_meas in range(Nm):
        y = residuals[:,i_meas]
        p_curr = p[:,i_meas]
        G = np.nan*np.ones([Nsat,3])
        for i_sat in range(Nsat):
            curr_sat_pos = sat_pos[i_sat][:, (offset+1)*Ndt + i_meas*iter_per_meas - 1]
            est_range = p_curr - curr_sat_pos
            G[i_sat,:] = est_range / np.linalg.norm(est_range)
        S = W @ (np.eye(Nsat) - G @ np.linalg.inv(G.transpose() @ W @ G) @ G.transpose() @ W)
        if (y.transpose() @ S @ y > threshold):
            return True
    return False


def simple_RAIM(residuals, threshold, sigma_range, simEnv, p, tm, offset):
    """
    Input: residuals -- list of measurement residuals for each position/sample
                            length of Nsam, each an array of dim (Nsat, Nm)
            threshold -- RAIM threshold for fault presence (scalar)
            sigma_range -- standard deviation for range measurements (scalar)
            simEnv -- simulation environment instance from TrainSimEnv
            p -- list of position estimates for each sample & for each meas
                            length of Nsam, each an array of dim (3,Nm)
            tm -- time between measurements (s)
            offset -- number of measurement sets to skip (Ndt measurements to skip)
    Output: RAIM_alert -- list of binary values indicating whether RAIM threshold exceeded
                            length of Nm, (alert -- True) or not (no alert -- False)
    """
    Nsam = len(residuals)
    # RAIM_labels = [None]*Nsam
    RAIM_labels = np.nan * np.ones(Nsam)
    for i_sam in range(Nsam):
        res_samp = residuals[i_sam]
        pos_samp = p[i_sam]
        RAIM_labels[i_sam] = simple_RAIM_1sample(res_samp, threshold, \
            sigma_range, simEnv, pos_samp, tm, offset)
    return RAIM_labels

def save_to_file_in_NN_format(filename_data, filename_label, Nsat, Nsam, data_res, data_snr, labels):
    # converts to a np array

    input_dim = Nsat*2

    data = np.nan * np.ones([Nsam, input_dim])

    for i in range(Nsam):
        data[i, 0:Nsat] = np.reshape(data_res[i],[1,-1])
        data[i,Nsat: input_dim + 1] = np.reshape(data_snr[i], [1,-1])

    np.savetxt(filename_data, data, delimiter=',')
    np.savetxt(filename_label, labels, delimiter=',')

def save_to_file_in_NN_format_with_p(filename_data, filename_label, filename_p, filename_tp, Nsat, Nsam, data_res, data_snr, labels, ps, tp):
    # converts to a np array
    # ps - nsam * 3 * 10
    # tp - 3 * 1000
    # 

    # print(tp.shape)

    input_dim = Nsat*2

    data = np.nan * np.ones([Nsam, input_dim])

    pss = np.nan * np.ones([Nsam, 3, ps[0].shape[1]])

    tpp = np.zeros((3,10))
    # print(pss.shape)

    tp_re = np.reshape(tp[:,0], [-1,1])

    for i in range(Nsam):
        data[i, 0:Nsat] = np.reshape(data_res[i],[1,-1])
        data[i,Nsat: input_dim + 1] = np.reshape(data_snr[i], [1,-1])
        for j in range(10):
            p_re = np.reshape(ps[i][:,j], [-1,1])
            p_enu, R_enu = ECEF_to_ENU(tp_re, p_re)
            mu_enu = R_enu @ (p_re - tp_re)
            pss[i,:,j] = np.reshape(mu_enu, (3,))

            t_re = np.reshape(tp[:,j], [-1,1])
            tp_enu, tR_enu = ECEF_to_ENU(tp_re, t_re)
            pmu_enu = tR_enu @ (t_re - tp_re)
            tpp[:,j] = np.reshape(pmu_enu, (3,))

    np.save(filename_p, pss)
    np.savetxt(filename_tp, tpp, delimiter=',')
    np.savetxt(filename_data, data, delimiter=',')
    np.savetxt(filename_label, labels, delimiter=',')


def read_from_file_in_SVM_format(filename_data, filename_labels):
    # return data_res, data_snr, labels

    data = np.loadtxt(filename_data, delimiter=',')
    labels = np.loadtxt(filename_labels, delimiter=',')

    Nsam = data.shape[0]
    Nsat = int(data.shape[1]/2)

    data_res = [None] * Nsam
    data_snr = [None] * Nsam

    for i in range(Nsam):
        data_res[i] = np.reshape(data[i,0:Nsat], [Nsat,-1])
        data_snr[i] = np.reshape(data[i,Nsat:2*Nsat+1], [Nsat,-1])

    return data_res, data_snr, labels


def ECEF_to_LLA(posvel_ECEF, ellipsoid=_WGS84, normalize=False, in_degrees=True):
    """
    Returns lla position in a record array with keys 'lat', 'lon' and 'alt'. 
    lla is calculated using the closed-form solution.
    
    For more information, see
     - Datum Transformations of GPS Positions
       https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
       provides both the iterative and closed-form solution.

     >>> #ECE Building and Mount Everest
     >>> array([(40.11497089608554, -88.22793631642435, 203.9925799164921), 
                (27.98805865809616, 86.92527453636706, 8847.923165871762)], 
         dtype=[('lat', '<f8'), ('lon', '<f8'), ('alt', '<f8')])
    
    @type  posvel_ECEF: np.ndarray
    @param posvel_ECEF: ECEF position in array of shape (3,N)
    @type  ellipsoid: dict
    @param ellipsoid: Reference ellipsoid (eg. _WGS84 = {'a': 6378137.0, 'invf': 298.257223563}). 
    Class headers contains some pre-defined ellipsoids: _IE24, _IE67, _WGS72, _GRS80, _WGS84
    @type  normalize: bool
    @param normalize: Its default value is False; True will cause longitudes returned 
    by this function to be in the range of [0,360) instead of (-180,180]. 
    Setting to False will cause longitudes to be returned in the range of (-180,180].
    @rtype : numpy.ndarray
    @return: Position in a record array with keys 'lat', 'lon' and 'alt'.
    """
                                  
    from numpy import sqrt, cos, sin, pi, arctan2 as atan2
    #python subtlety only length 1 arrays can be converted to Python scalars
    #if we use the functions from math, we have to use map

    a    = ellipsoid['a']
    invf = ellipsoid['invf']
    f = 1.0/invf
    b = a*(1.0-f)
    e  = sqrt((a**2.0-b**2.0)/a**2.0)
    ep = sqrt((a**2.0-b**2.0)/b**2.0)
    
    xyz = np.asarray(posvel_ECEF)
    x = xyz[0,:]
    y = xyz[1,:]      
    z = xyz[2,:]
    
    # Create the record array.
    cols = np.shape(xyz)[1]
    lla  = np.zeros(cols, dtype = { 'names' : ['lat', 'lon', 'alt'], 
                                  'formats' : ['<f8', '<f8', '<f8']})
                                  
    lon = atan2(y, x)
    p = sqrt(x**2.0+y**2.0)
    theta = atan2(z*a,p*b)
    lat = atan2((z+(ep**2.0)*(b)*(sin(theta)**3.0)),(p-(e**2.0)*(a)*(cos(theta)**3.0)))
    N = a/sqrt(1.0-((e**2.0)*(sin(lat)**2.0)))
    alt = p/cos(lat)-N
    
    if normalize:
        lon = np.where(lon < 0.0, lon + 2.0*pi, lon)
    
    lla['alt'] = alt
    
    if in_degrees:
        lla['lat'] = lat*180.0/pi
        lla['lon'] = lon*180.0/pi
    else:
        lla['lat'] = lat
        lla['lon'] = lon        
    
    return lla

def LLA_to_ECEF(lat,lon,alt, ellipsoid=_WGS84):
    '''
    For more information, see
     - Datum Transformations of GPS Positions
       https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
       provides both the iterative and closed-form solution.

    '''
    from numpy import sqrt, cos, sin, pi, arctan2 as atan2
    #python subtlety only length 1 arrays can be converted to Python scalars
    #if we use the functions from math, we have to use map

    a    = ellipsoid['a']
    invf = ellipsoid['invf']
    f = 1.0/invf
    b = a*(1.0-f)
    e  = sqrt((a**2.0-b**2.0)/a**2.0)
    
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    alt = np.array(alt)
    
    N = a/sqrt(1-e**2*sin(lat)**2)
    X = (N+alt)*cos(lat)*cos(lon) 
    Y = (N+alt)*cos(lat)*sin(lon)
    Z = (((b**2)/(a**2))*N+alt)*sin(lat)
    
    posvel_ECEF = np.matrix(np.zeros([3,len(lat)]))
    posvel_ECEF[0,:] = X
    posvel_ECEF[1,:] = Y
    posvel_ECEF[2,:] = Z
    
    return posvel_ECEF

def ECI_to_ECEF(posvel_ECI, t_gps=None, t_c=None):
    """
    Returns rotated ECEF position and velocity coordinates
    due to the Earth's rotation during the given time duration from t_c to t_gps.
        
    For more information, see
     - Convert ECI to ECEF Coordinates by Darin Koblick
       http://www.mathworks.com/matlabcentral/fileexchange/28233-convert-eci-to-ecef-coordinates/content/ECI2ECEF.zip
     - Constell user manual for velocity rotation. 
       http://www.constell.org/Downloads/Manual-Version70.pdf
       (not sure if it is correct,
       there is a GPS World article that verified Constell)
     - NIST Technical Note 1385
       Global Position System Receivers and Relativity
    
    @type  posvel_ECI: np.matrix
    @param posvel_ECI: (8,N) np.matrix
    @type  t_gps: float
    @param t_gps: This is the time coordinate (s) of the rotating earth-fixed frame (ECEF).
    @type  t_c: float
    @param t_c: This is the time coordinate (s) of the non-rotating inertial reference frame (ECI).
    @rtype : np.matrix
    @return: (8,N) np.matrix with rotated position and velocity.
    """
    
    xyz    = posvel_ECI[0:3,:]
    xyzdot = posvel_ECI[4:7,:]

    # Compute the rotation angle.
    otau = OEDot*(t_gps-t_c)
    
    # Establish the rotation matrix.
    cotau  = np.cos(otau)
    sotau  = np.sin(otau)
    rot    = np.matrix([[cotau, sotau, 0], [-sotau, cotau, 0], [0, 0, 1]])
    rotdot = np.matrix([[0, - OEDot, 0], [OEDot, 0, 0], [0, 0, 0]])
    
    # Perform the rotation.
    rotxyz    = rot*xyz
    rotxyzdot = rot*(xyzdot - rotdot*xyz)
    
    #the following equations should give the same result
    #cotau_oedot = np.cos(otau)*OEDot
    #sotau_oedot = np.sin(otau)*OEDot
    #rotdot = np.matrix([[-sotau_oedot, cotau_oedot, 0], [-cotau_oedot, -sotau_oedot, 0], [0, 0, 0]])
    #rotxyzdot = rot*xyzdot + rotdot*xyz

    posvel_ECEF = np.matrix(np.zeros(np.shape(posvel_ECI)))
    posvel_ECEF[0:3,:] = rotxyz
    posvel_ECEF[3,:]   = posvel_ECI[3,:]
    posvel_ECEF[4:7,:] = rotxyzdot
    posvel_ECEF[7,:]   = posvel_ECI[7,:]
    
    return posvel_ECEF


def ECEF_to_ECI(posvel_ECEF, t_gps=None, t_c=None):
    """
    Returns rotated ECI position and velocity coordinates 
    due to the Earth's rotation during the given time duration from t_c to t_gps.
    
    For more information, see
     - IS-GPS-200H page 106 for position rotation.
     - Convert ECI to ECEF Coordinates by Darin Koblick
       http://www.mathworks.com/matlabcentral/fileexchange/28233-convert-eci-to-ecef-coordinates/content/ECI2ECEF.zip
     - Constell user manual for velocity rotation. 
       http://www.constell.org/Downloads/Manual-Version70.pdf
       (not sure if it is correct, 
       there is a GPS World article that verified Constell)
     - NIST Technical Note 7385
       Global Position System Receivers and Relativity
     - "Fundamentals of Inertial Navigation, Satellite-based 
       Positioning and their Integration" by A. Noureldin et al, 
       DOI: 10.1007/978-3-642-30466-8_2 for coordinate transformations

    @type  posvel_ECEF: np.matrix
    @param posvel_ECEF: (8,N) np.matrix
    @type  t_gps: float
    @param t_gps: This is the time coordinate (s) of the rotating earth-fixed frame (ECEF).
    @type  t_c: float
    @param t_c: This is the time coordinate (s) of the non-rotating inertial reference frame (ECI).
    @rtype : np.matrix
    @return: (8,N) np.matrix with rotated position and velocity.            
    """
    
    xyz    = posvel_ECEF[0:3,:]
    xyzdot = posvel_ECEF[4:7,:]
    
    # Compute the rotation angle.
    otau = OEDot*(t_gps-t_c)
    
    # Establish the rotation matrix.
    cotau = np.cos(otau)
    sotau = np.sin(otau)
    rot = np.matrix([[cotau, -sotau, 0], [sotau, cotau, 0], [0, 0, 1]])
    rotdot = np.matrix([[0, -OEDot, 0], [OEDot, 0, 0], [0, 0, 0]])
    # Perform the rotation.
    rotxyz = rot*xyz
    rotxyzdot = rot*xyzdot + rotdot*rotxyz 
    #following equations should give the same result
    #cotau_oedot = np.cos(otau)*OEDot
    #sotau_oedot = np.sin(otau)*OEDot
    #rotdot = np.matrix([[-sotau_oedot, -cotau_oedot, 0], [cotau_oedot, -sotau_oedot, 0], [0, 0, 0]])
    #rotxyzdot = rot*xyzdot + rotdot*xyz 
    
    posvel_ECI = np.matrix(np.zeros(np.shape(posvel_ECEF)))
    posvel_ECI[0:3,:] = rotxyz
    posvel_ECI[3,:]   = posvel_ECEF[3,:]
    posvel_ECI[4:7,:] = rotxyzdot
    posvel_ECI[7,:]   = posvel_ECEF[7,:]
    
    return posvel_ECI

def ECEF_to_ENU(refState=None, curState=None, diffState=None):
    """
    Returns 3xN matrix in enu order
    - Toolbox for attitude determination 
       Zhen Dai, ZESS, University of Siegen, Germany

    @type  curState: 3x1 or 8x1 matrix
    @param curState: current position in ECEF                 
    @type  refState: 3x1 or 8x1 matrix
    @param refState: reference position in ECEF
    @rtype : tuple
    @return: (numpy.matrix (3,N) (ENU), numpy.matrix (3,3) R_ECEF2ENU)
    """  
        
    lla = ECEF_to_LLA(refState,in_degrees=False)
    lat = lla['lat'][0]
    lon = lla['lon'][0]

    slon = np.sin(lon)
    clon = np.cos(lon)
    slat = np.sin(lat)
    clat = np.cos(lat)

    R_ECEF2ENU = np.mat([[ -slon, clon, 0.0],
                         [ -slat*clon, -slat*slon, clat],
                         [  clat*clon,  clat*slon, slat]])
    
    xyz0 = refState[0:3,:]

    if (curState is not None) and (diffState is None):    
        xyz  = curState[0:3,:]
        dxyz = xyz - np.tile(xyz0, (1,np.shape(curState)[1]))
    elif (curState is None) and (diffState is not None):
        dxyz = diffState[0:3,:]
    else:
        print("Unknown error, shape of states:")
        print('refState: '+str(np.shape(refState)))
        print('curState: '+str(np.shape(curState)))
        print('diffState: '+str(np.shape(diffState)))
        
    matENU = R_ECEF2ENU*dxyz

    return matENU, R_ECEF2ENU
    
def ENU_to_ECEF(refState=None, diffState=None, R_ECEF2ENU= None):
    """
    """  
    if R_ECEF2ENU is None:
        lla = geodetic(refState, in_degrees=False)
        lat = lla['lat'][0]
        lon = lla['lon'][0]
    
        slon = np.sin(lon)
        clon = np.cos(lon)
        slat = np.sin(lat)
        clat = np.cos(lat)
    
        R_ECEF2ENU = np.mat([[ -slon, clon, 0.0],
                             [ -slat*clon, -slat*slon, clat],
                             [  clat*clon,  clat*slon, slat]])
                             
    R_ENU2ECEF = R_ECEF2ENU.T
    
    xyz0 = refState[0:3,:]
    dxyz = diffState[0:3,:]
        
    matECEF = R_ENU2ECEF*dxyz+np.tile(xyz0, (1,np.shape(dxyz)[1]))

    return matECEF

def ENU_to_elaz(ENU):
    
    ENU = np.asarray(ENU)
    east  = ENU[0,:]
    north = ENU[1,:]      
    up    = ENU[2,:]
    
    # Create the record array.
    cols = np.shape(ENU)[1]
    elazd = np.zeros(cols, dtype = { 'names' : ['ele', 'azi', 'dist'], 
                                   'formats' : ['<f8', '<f8', '<f8']})
    
    horz_dist = np.sqrt(east**2 + north**2)
    elazd['ele']  = np.arctan2(up, horz_dist)        
    elazd['azi']  = np.arctan2(east, north)
    elazd['dist'] = np.sqrt(east**2 + north**2 + up**2)

    return elazd

### plotting utils

def myplot(x,y,xllim, xulim, yllim, yulim):
    plt.plot(x,y, '*')
    plt.axis([xllim, xulim, yllim, yulim])
    plt.show()