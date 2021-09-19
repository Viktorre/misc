### This script uses adaptive learning to solve the msv model with exogene shocks


##### module imports
###
#
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
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
gamma = 0.05
M = np.array([[1,1/sigma,0],[0,beta,0],[phi_1,phi_2,0]])
A = np.array([[1,0,1/sigma],[-kappa,1,0],[0,0,1]])
D = np.zeros((3,3))
#
###
##### end define input variables

##### define guess values 
###
#
F = np.ones((3,3))
F_new = np.zeros((3,3))
#
###
##### end define gues values


##### 1.2
###
#
F_new = np.dot((np.linalg.inv(A-np.dot(M,F))),D)
F = F_new
#print(F)

Q = np.linalg.inv(A-np.dot(M,F_new))
#print(Q)
#
###
##### end 1.2


##### 1.4. add shock with AC
###
#
rho_e = 0.8
A= np.array([ [1,0,0,0],
              [-1,1,0,1/sigma],
              [0,-kappa,1,0],
              [0,0,0,1]])
M = np.array([[0,0,0,0],
              [0,1,1/sigma,0],
              [0,0,beta,0],
              [0,phi_1,phi_2,0]])
D = np.zeros((4,4))
D[0,0] = rho_e
F= np.ones((4,4))
F_new = np.zeros((4,4))

#F_new = np.dot((np.linalg.inv(A-np.dot(M,F))),D)
#F = F_new
#print(F)


#
###
##### 



##### 1.5
###
#
F_new = np.ones((4,4)) #redefine starts vals, as the results are start values dependent
F = np.zeros((4,4))
while np.max(np.abs(F-F_new))>0.000001:
    F = F_new
    F_new = np.dot(np.linalg.inv(A-np.dot(M,F)),D)
print(F_new,'F_new')

    
#
###
##### 1.5



##### 1.6
###
#
Q = np.linalg.inv(A-np.dot(M,F_new))
print(Q,'Q')
#
###
##### 1.6



##### 1.7
###
#
C1 = Q[:,0]
C5 = F[:,0] 
#C1 = np.array([1,1.759,2.537,3.526])
#C5 = np.array([0.8,1.487,2.029,2.6686])
#
###
#####



##### 1.8
###
#
N = 20 #save number of iterations in variable N
# define vectors of 20 zeros for epsilon_e,e,y,pi.ni
epsilon_e, e, y, pi, ni = np.zeros(20),np.zeros(20),np.zeros(20),np.zeros(20),np.zeros(20) 
epsilon_e[0] = 0.01
for i in range(N):
    e[i] = C1[0]*epsilon_e[i]+C5[0]*e[i-1]
    y[i] = C1[1]*epsilon_e[i]+C5[1]*e[i-1]
    pi[i] = C1[2]*epsilon_e[i]+C5[2]*e[i-1]
    ni[i] = C1[3]*epsilon_e[i]+C5[3]*e[i-1]
#for i in range(N):
#    [e[i],y[i],pi[i],ni[i]] = [C1[hh]*epsilon_e[i]+C5[hh]*e[i-1] for hh in range(4)] 
#
###
##### 1.9


##### 1.9
###
#
fig = plt.figure()
plt.subplot(2,2,1)
plt.plot([y[i]*100 for i in range(1,N)],linewidth=2)
plt.title('Output')
plt.xlabel('t')
plt.ylabel('Y_t')

plt.subplot(2,2,2)
plt.plot([pi[i]*100 for i in range(1,N)],linewidth=2)
plt.title('pi')
plt.xlabel('t')
plt.ylabel('pi_t')

plt.subplot(2,2,3)
plt.plot([ni[i]*100 for i in range(1,N)],linewidth=2)
plt.title('ni')
plt.xlabel('t')
plt.ylabel('ni_t')

plt.subplot(2,2,4)
plt.plot([e[i]*100 for i in range(1,N)],linewidth=2)
plt.title('e')
plt.xlabel('t')
plt.ylabel('e_t')

plt.tight_layout()
#
###
#####

'''Dear Joep,

thank you! I found the error.

Am I correct that in assignment 4, the logic of the C-factors is as follows:

if Q and F (subsequently A,M,D too) are 4x4 matrices, then for both their four columns account for (7), (1), (2) and (3) in that particular order.

Hence, the first column of Q shows the impact factors for z_t by the shock e_t. The second (there is none in the 4th assignment) would be the impact factors for z_t by an shock on Y_t. Third the shock on pi_t and fourth on i_t. 

For the F the column order (e,Y,pi,i) is the same, but here we have the impact factors by the lags (t-1).

Is that correct? I ask because I use that logic to identify the C-factors in the final assignment. See the screenshot I sent in my last mail.

Best regards,
Viktor'''






 