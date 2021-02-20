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

    def compute_transformation_matrix(self, mode='canonical'):
        '''
        parameter mode should be 'canonical' or 'symmetric'
        return: regular matrix
        '''
        X = np.zeros(self.S.shape)
        
        #固有値の配列、固有ベクトルをそれぞれ返す
        e, e_v = np.linalg.eigh(self.S)
        if mode=='canonical':
            #正準直交化
            for i in range(self.num_AO):
                for j in range(self.num_AO):
                    X[i,j] = e_v[i,j] / np.sqrt(e[j])
        elif mode=='symmetric':
            #対称直交化
            regular_matrix = np.zeros(self.S.shape)
            for i in range(len(e)):
                regular_matrix[i,i] = 1/np.sqrt(e[i])
            X = e_v.dot(regular_matrix.dot(np.conjugate(e_v)))
        return X
    
    def update_P_matrix(self, P, C):
        '''
        parameter P: density matrix
        parameter C_dash: X*C(C: regular matrix computed from Fock matrix)
        return: updated density matrix
        '''
        new_P = np.zeros(P.shape)
        ### 占有軌道の数
        num_occupied_orbital = self.num_elec // 2 
        for i in range(self.num_AO):
            for j in range(self.num_AO):
                tmp = 0
                for k in range(num_occupied_orbital):
                    tmp += 2*(C[i,k]*C[j,k])
                new_P[i,j] = tmp
        return new_P

    def compute_term_2_elec_integral(self, P: np.array):
        '''
        2電子積分(self.two_elec_integral)がエルミート行列であることを確認
        '''
        new_G = np.zeros((self.num_AO,self.num_AO))

        for i in range(self.num_AO):
            for j in range(self.num_AO):
                for k in range(self.num_AO):
                    for l in range(self.num_AO):
                        J = self.two_elec_integral[i][j][k][l]
                        K = 0.5 * self.two_elec_integral[i][l][k][j]
                        new_G[i,j] += P[k,l]*(J-K)
        return new_G

    def judge_scf_end(self, new_P: np.array, P: np.array):
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
        X = self.compute_transformation_matrix()
        X_adj = np.matrix.getH(X) #共役転置行列を得る
        P = np.zeros(self.S.shape)
        MAX_ITER = 10

        for n_iter in range(MAX_ITER):
            print('-'*50)
            print(f"ITERATION={n_iter+1}/{MAX_ITER}")
            G = self.compute_term_2_elec_integral(P)
            Fock = self.Hcore + G
            Fock_dash = X_adj.dot(Fock.dot(X))
            eigen_val, C_dash = np.linalg.eigh(Fock_dash)
            C = X.dot(C_dash)
            new_P = self.update_P_matrix(P, C)

            #参考: https://qiita.com/hatori_hoku/items/867fa793488ebe1d2beb
            energy = (new_P * (self.Hcore + Fock)).sum() * 0.5 
            print(f'Electronic Energy = {energy} hartrees')
            do_finish = self.judge_scf_end(new_P, P)
            if do_finish:
                print('Finish SCF Calculation')
                print('\nComputed Density Matrix')
                print(new_P)
                print('-'*50)
                break
            P = new_P
            print('-'*50)
