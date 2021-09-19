### This script uses adaptive learning to solve the msv model with exogene shocks


##### module imports
###
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#
###
##### end module iports


##### define input variables
###
#
sigma = 2
kappa = 0.3
beta = 0.99
phi_1 = 1.5
phi_2 = 0.2
c_hat_old = np.array([0.5, 2, -0.2, 0.6, 0.3, 1]).reshape((2,3))
e = np.random.normal(0, 1, size=10000)
R_old = np.array([1,1,1,1]).reshape((2,2))
gamma = 0.05
#
###
##### end define input variables

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

def plot_results(result):
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].plot(result['y'])
    axes[0, 1].plot(result['pi'])
    axes[0, 2].plot(result['i'])
    axes[1, 0].plot(result['y_exp'])
    axes[1, 1].plot(result['pi_exp'])
    axes[1, 2].plot(result['e'])    
    axes[0, 0].set_title('y')
    axes[0, 1].set_title('pi')
    axes[0, 2].set_title('i')
    axes[1, 0].set_title('y_exp')
    axes[1, 1].set_title('pi_exp')
    axes[1, 2].set_title('e')
    fig.tight_layout()    

def msv_adaptive_learner(R_old, gamma, e, C_hat_old, sigma, kappa, beta, phi_1, phi_2):
    y, pi, i, y_exp, pi_exp = [], [], [], [], []
    for e_t in e:
        C0_hat_old = C_hat_old[0]
        z_t = calculate_z_with_shock(C0_hat_old, sigma, kappa, beta, phi_1, phi_2, e_t)
        z_t = z_t.reshape((1,3))
        v = np.array([1,e_t]).reshape((2,1))
        R_new = R_old + gamma * (v * v.T - R_old)
        #in the next line i use @ instead of np.dot() to multiply matrices of different ranks
        C_hat_new = C_hat_old + gamma * np.linalg.inv(R_new) @ v * (z_t - v.T @ C_hat_old)
        C0_hat_new = C_hat_new[0]
        C1_hat_new = C_hat_new[1]
        y.append((C0_hat_new + C1_hat_new * e_t)[0]) #gives same as y.append(z_t[0][0])
        pi.append((C0_hat_new + C1_hat_new * e_t)[1]) #gives same as y.append(z_t[0][1])
        i.append((C0_hat_new + C1_hat_new * e_t)[2]) #gives same as y.append(z_t[0][2])
        y_exp.append(C0_hat_new[0])
        pi_exp.append(C0_hat_new[0])
        C_hat_old = C_hat_new
        R_old = R_new
    print('final values of the C_hat array:',C_hat_new, 'The results are the same as last week. Agents manage to learn the MSV solution')
    return pd.DataFrame({'y':y, 'pi':pi, 'i':i, 'y_exp':y_exp, 'pi_exp':pi_exp, 'e':e})
#
###
##### end define functions


##### use msv_adaptive_learner function
###
#
result = msv_adaptive_learner(R_old, gamma, e, c_hat_old, sigma, kappa, beta, phi_1, phi_2)
print(result)
plot_results(result)
#
###
##### end use function