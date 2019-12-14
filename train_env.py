import numpy as np
import util

class TrainSimEnv:

    def __init__(self, pos_init, vel, sat_pos_init, Ndt, Ntraj, sigma_init, delta_t, Nsat, R_ekf, Qa, Qb, v_sigma):
        """
        Constructor
        """
        
        self.Nsat = Nsat
        self.pos_init = pos_init
        self.vel = vel
        self.sat_pos_init = sat_pos_init # is a list
        self.Ndt = Ndt 
        self.Ntraj = Ntraj
        self.Nsat = Nsat
        self.sigma_init = sigma_init
        self.delta_t = delta_t
        self.R_ekf = R_ekf
        self.Qa = Qa
        self.Qb = Qb
        self.sat_pos = [None] * Nsat
        self.v_sigma = v_sigma

    def gen_agent_traj(self):
        """
        uses init_pos and vel (vel has size 3 x (Ndt x Ntraj)) to create agent trajectory
        output: array of agent positions 3 x (Ndt x Ntraj)
        """

        pos = np.zeros((3,self.Ndt * self.Ntraj))
        vel = self.vel
        delta_t = self.delta_t

        pos[:,0] = self.pos_init
        # print(pos[:,0])
        for i in range(0, (self.Ndt * self.Ntraj)):
            # if i == 0: 
            #     pos[:,i] = self.pos_init
            # else:
            #     pos[:,i] = self.linear_motion_model(pos[:,i-1], vel[:,i-1], delta_t)
            if i == 0:
                pos[:,0] = self.linear_motion_model(pos[:,0], vel[:,0], delta_t)
            else:
                pos[:,i] = self.linear_motion_model(pos[:,i-1], vel[:,i-1], delta_t)
        
        self.pos = pos

    def gen_sat_traj(self):
        """
        uses sat_pos to creat sat_traj
        output: list of length Nsat each element is an array of size 3 x (Ndt x Ntraj)
        """
        for j in range(0, self.Nsat):
            self.sat_pos[j] = (np.tile(self.sat_pos_init[j], ((self.Ndt) * (self.Ntraj), 1))).transpose()    

    def linear_motion_model(self, pos, vel, delta_t):
        """
        linear velocity motion model
        Outputs: updated p each are 3 dimensional
        """
        pos = pos + delta_t * vel
        return pos
