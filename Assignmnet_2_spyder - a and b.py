### This script uses fsolve to solve the msv model with exogene shocks


##### module imports
###
#
#import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
#
###
##### end module iports



##### define parameters
###
#
sigma = 2
kappa = 0.3
beta = 0.99
phi_1 = 1.5
phi_2 = 0.2
#
###
##### end define parameters



##### define functions 
###
#
def calculate_z_with_shock(C_0_guess, sigma, kappa, beta, phi_1, phi_2, e_t):
    A_inv = np.linalg.inv(np.array([[1, 0, 1/sigma], [-kappa, 1, 0], [0, 0, 1]]))
    B = np.array((C_0_guess[0] - 1 / sigma * (C_0_guess[1]) + e_t, #shock added
                  beta * C_0_guess[1],
                  phi_1 * C_0_guess[1] + phi_2 * C_0_guess[0]))
    B = B.reshape((3, -1))
    z_t = np.dot(A_inv, B)
    return z_t

def return_diff_between_zt_and_input(C_0_and_C_1_guess, sigma, kappa, beta, phi_1, phi_2):
    #split input vector into c1 and c0
    C_0_guess = C_0_and_C_1_guess[:3]
    #calculate expectations (i assume expected val of white noise is zero, so it does not affect expectations)
    exp_y_pi_i = C_0_guess
    # calculate c0 and c1 via expected value trick
    e_t = 0
    z_t_zero_shock = calculate_z_with_shock(exp_y_pi_i, sigma, kappa, beta, phi_1, phi_2, e_t)
    C_0 = z_t_zero_shock
    e_t = 1
    z_t_one_shock = calculate_z_with_shock(exp_y_pi_i, sigma, kappa, beta, phi_1, phi_2, e_t)
    C_1 = z_t_one_shock - C_0
    # calculate diff
    C_0_and_C_1 = []
    for w in C_0:
        C_0_and_C_1.append(w[0])
    for w in C_1:
        C_0_and_C_1.append(w[0])
    C_0_and_C_1 = np.array(C_0_and_C_1).reshape((6,))
    diff = np.array(C_0_and_C_1_guess).reshape((6),) - C_0_and_C_1
    return diff
#
###
##### end define functions

##### use fsolve  
###
#
tuple_of_start_vals = (10000, 1000, 5, -3, 5, 10)

solution = fsolve(return_diff_between_zt_and_input, list(tuple_of_start_vals), 
                  args=(sigma, kappa, beta, phi_1, phi_2), full_output = False)
print('fsolve',solution)

print('the results are the same as before, because the shock does not change the expectation of t+1 in Y')

###
##### end use fsolve
