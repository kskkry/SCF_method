import math
import numpy as np
import matplotlib.pyplot as plt
from h2_config import InitializedParameter

N_elec = 2
N_AO = 2
THRESHOLD = 1e-3

parameter = InitializedParameter()
parameter.initialize()

def compute_transformation_matrix():
    '''
    canonical orthogonalization
    重なり積分の値は固定されているので計算結果を代入する
    '''
    transformation_matrix = np.zeros((N_AO, N_AO))

    transformation_matrix[0,0] = -0.7071 / np.sqrt(0.3407)
    transformation_matrix[0,1] = 0.7071 / np.sqrt(1.6593)
    transformation_matrix[1,0] = 0.7071 / np.sqrt(0.3407)
    transformation_matrix[1,1] = 0.7071 / np.sqrt(1.6593)
    return transformation_matrix

def eri_table(u,v,p,q):
    '''
    参考より引用
    parameter.initialize_two_elec_integral()により得られる2電子積分の値と同じ
    '''
    if u < v:
        u,v = v,u
    if p < q:
        p,q = q,p
    if (u+1)*(v+1) < (p+1)*(q+1):
        u,v,p,q = p,q,u,v
    
    # Szabo. pp162(p3.235)
    if (u,v,p,q) == (0,0,0,0) or (u,v,p,q) == (1,1,1,1):
        return 0.7746
    elif (u,v,p,q) == (1,1,0,0):
        return 0.5697
    elif (u,v,p,q) == (1,0,0,0) or (u,v,p,q) == (1,1,1,0):
        return 0.4441
    elif (u,v,p,q) == (1,0,1,0):
        return 0.2970
    else:
        # Never get here.
        print(u,v,p,q)
        raise

def compute_term_two_elec_integral(P: np.array):
    '''
    密度行列Pと2電子積分を示す行列との積の項を算出する
    計算の際での参考: https://github.com/YukiSakamoto/HartreeFock_py/blob/master/step1/hf.py
    '''
    G = np.zeros((2,2))
    tei = parameter.two_elec_integral #2電子積分行列

    for i in range(N_AO):
        for j in range(N_AO):
            for k in range(N_AO):
                for l in range(N_AO):
                    J = eri_table(i,j,k,l)
                    K = 0.5 * eri_table(i,l,k,j)
                    G[i,j] += P[k,l] * (J-K)
    return G

def compute_transformation_fock_matrix(fock: np.array, X: np.array):
    '''
    fock行列は(核-1)電子ハミルトニアン行列とGを足し合わせて得られる行列である
    変換行列を用いてfock行列を変換する
    '''
    adj_X = np.matrix.getH(X)
    return adj_X.dot(fock.dot(X))

def compute_updated_P_matrix(D):
    '''
    密度行列Pを更新する
    '''
    new_P = np.zeros((N_AO, N_AO))
    N_trial = N_elec // 2
    for i in range(N_AO):
        for j in range(N_AO):
            temp = 0
            for k in range(N_trial):
                temp += 2*D[i,k]*D[j,k] 
            new_P[i,j] = temp
    return new_P

def get_eval_convergence(new_P, P):
    '''
    計算の収束を判定する
    電荷密度の変化が微小であるかどうかで収束を決定する
    参考:https://qiita.com/hatori_hoku/items/867fa793488ebe1d2beb
    '''
    val = 0.5 * ((new_P - P)**2).sum()
    finish_scf = False
    if val < THRESHOLD:
        finish_scf = True
    return finish_scf, val

def main():
    X = compute_transformation_matrix()
    X_adj = np.matrix.getH(X)
    P = np.zeros((N_AO, N_AO))
    MAX_ITER = 10

    for n_iter in range(MAX_ITER):
        print('-' * 50)
        print(f"ITERATION : {n_iter+1}.")

        G = compute_term_two_elec_integral(P)
        F = parameter.H_core_init + G
        F_dash = X_adj.dot(F.dot(X))

        eigen_value, C = np.linalg.eigh(F_dash)
        D = X.dot(C)

        # 密度行列Pを更新
        new_P = compute_updated_P_matrix(D)
        judge_scf, val = get_eval_convergence(new_P, P)
        print(f"error value={val}")
        if judge_scf:
            print("\nH2 Density Matrix\n", new_P)
            print('-'*50)
            break
    
        P = new_P
        print('-' * 50, '\n'*5)



if __name__=="__main__":
    main()
