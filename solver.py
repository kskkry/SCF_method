import numpy as np


class SCFSolver:
    '''
    Reference: A.Szabo, N.S.Ostlund
    https://www.amazon.co.jp/-/en/Attila-Szabo/dp/4130621114/ref=pd_lpo_14_t_0/357-6877730-3564413?_encoding=UTF8&pd_rd_i=4130621114&pd_rd_r=21208b56-0531-4686-a004-b21b0d615dbe&pd_rd_w=r2Xpu&pd_rd_wg=oGSC6&pf_rd_p=cb2cef9d-b0a3-4b58-a575-45abfc5e07e8&pf_rd_r=3C5KDGNZX2WMMXDKWT0T&psc=1&refRID=3C5KDGNZX2WMMXDKWT0T
    '''
    def __init__(self, Hcore: np.array, S: np.array, two_elec_integral: np.array, num_elec: int, num_AO: int):
        '''
        parameter Hcore: Hamiltonial Matrix
        parameter S: Overlap Integral Matrix
        parameter two_elec_integral: 2 Electron Integral Matrix
        parameter num_elec: Number of Electrons
        parameter num_AO: Number of Atomic Orbitals
        '''
        self.Hcore = Hcore
        self.S = S
        self.two_elec_integral = two_elec_integral
        self.num_elec = num_elec
        self.num_AO = num_AO
        assert self.Hcore.shape == self.S.shape
        assert self.Hcore.shape == self.two_elec_integral.shape

    def compute_transformation_matrix(self, mode='canonical'):
        '''
        parameter mode should be 'canonical' or 'symmetric'
        return: regular matrix
        '''
        X = np.array(self.S.shape)
        if mode=='canonical':
            pass
        elif mode=='symmetric':
            pass
        return X
    
    def update_P_matrix(self, P, C_dash):
        '''
        parameter P: density matrix
        parameter C_dash: X*C(C: regular matrix computed from Fock matrix)
        return: updated density matrix
        '''
        new_P = np.array(P.shape)
        ### 占有軌道の数
        num_occupied_orbital = self.num_elec // 2 
        for i in range(self.num_AO):
            for j in range(self.num_AO):
                tmp = 0
                for k in range(self.num_occupied_orbital):
                    tmp += C_dash[i,k]*C_dash[j,k]
                new_P = tmp
        return new_P

    def judge_scf_end(self, new_P, P):
        '''
        parameter P: density matrix
        return: T/F (if True: Finish SCF loops)
        '''
        THRESHOLD=1e-4
        diff_val = 0.5 * ((new_P-P)**2).sum()
        do_finish = False
        if diff_val < THRESHOLD:
            do_finish = True
        return do_finish

    def scf(self):
        pass

