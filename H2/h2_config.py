import numpy as np


class InitializedParameter:
    '''
    パラメータの値はザボの教科書より引用(https://www.amazon.co.jp/%E6%96%B0%E3%81%97%E3%81%84%E9%87%8F%E5%AD%90%E5%8C%96%E5%AD%A6%E2%80%95%E9%9B%BB%E5%AD%90%E6%A7%8B%E9%80%A0%E3%81%AE%E7%90%86%E8%AB%96%E5%85%A5%E9%96%80%E3%80%88%E4%B8%8A%E3%80%89-Attila-Szabo/dp/4130621114)
    初版 pp.167-179
    '''
    def __init__(self):
        self.S_init = np.zeros((2,2))
        self.kinetic_energy_init = np.zeros((2,2))
        self.potential_energy1_init = np.zeros((2,2))
        self.potential_energy2_init = np.zeros((2,2))
        self.H_core_init = np.zeros((2,2))
        self.two_elec_integral = np.zeros((2,2,2,2))

    def initialize_S_H2(self):
        self.S_init[0, 0] = 1
        self.S_init[1, 1] = 1
        self.S_init[0, 1] = 0.6593
        self.S_init[1, 0] = 0.6593

    def initialize_Hcore_H2(self):
        self.kinetic_energy_init[0,0] = self.kinetic_energy_init[1,1] = 0.7600
        self.kinetic_energy_init[0,1] = self.kinetic_energy_init[1,0] = 0.2365

        self.potential_energy1_init[0,0] = -1.2266
        self.potential_energy1_init[1,1] = -0.6538
        self.potential_energy1_init[0,1] = self.potential_energy1_init[1,0] = -0.5974

        self.potential_energy2_init[0,0] = -0.6538
        self.potential_energy2_init[1,1] = -1.2266
        self.potential_energy2_init[0,1] = self.potential_energy2_init[1,0] = -0.5974

        self.H_core_init = self.kinetic_energy_init + self.potential_energy1_init + self.potential_energy2_init
        
    def initialize_two_elec_integral(self):
        self.two_elec_integral[0][0][0][0] = self.two_elec_integral[1][1][1][1] = 0.7746
        self.two_elec_integral[1][1][0][0] = 0.5697
        self.two_elec_integral[1][0][0][0] = self.two_elec_integral[1][1][1][0] = 0.4441
        self.two_elec_integral[1][0][1][0] = 0.2970
        

    def initialize(self):
        self.initialize_S_H2()
        self.initialize_Hcore_H2()
        self.initialize_two_elec_integral()



