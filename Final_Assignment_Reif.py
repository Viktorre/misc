'''
Computational Macroeconomics
Final Assignemnt 
Viktor Reif
March 2021 
'''

##### module imports
###
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from statsmodels.formula.api import ols
#
###
##### end module iports


##### define input variables
###
#
sigma= 1 #code copied from sheet, simple variable definitions
kappa=0.3
beta=0.995
phi_pi=1.5
phi_y=0.1
rho_mu=0.7
#
###
##### end define input variables
    

##### 1.
###
#
A = np.array([[1,0,0,0], # define 4x4 numpy array containing the values of the matrix A
              [0,1,0,1/sigma],
              [-1,-kappa,1,0],
              [0,0,0,1]])

M = np.array([[0,0,0,0], # define 4x4 numpy array containing the values of the matrix M
              [0,1,1/sigma,0],
              [0,0,beta,0],
              [0,0,0,0]])

# define 4x4 numpy array containing the values of the matrix D
D = np.array([[rho_mu,0,0,0], 
              [0,0,0,0],
              [0,0,0,0],
              [0,phi_y,phi_pi,0]])
#
###
##### end 1.
    
##### 2.
###
#
# function calculates z_t given A,M,D and a guess and returns the diff between the guess
# and z_t. Arguments are the before defined matrices and the guessed values for z_t.
def calculate_z_t_via_AMD_and_return_diff(z_guess,A,M,D):
    # z_t is calculated via (6). I moved the A matrix to the right hand side via the
    # inverse. I skip the steps to define E_t*t_t+1 and z_t-1 and instead use z_guess
    # directly. u_t is 0, so I leave it out alltogether. "@" is a shorter way to do
    # what np.dot() does.
    z_t = np.linalg.inv(A) @ (M @ z_guess + D @ z_guess)
    return z_t - z_guess #return difference between calculation and guess
#
###
##### end
    
##### 3. use fsovle 
###
#
z_guess = [0.1, 0.9, 0.5, -0.3] #define vector of guessed values for z_t
#call fsolve to find a solution to our defined function and save whatever it 
# returns into variable with the name "solution". 
solution = fsolve(calculate_z_t_via_AMD_and_return_diff, z_guess, 
                  args=(A,M,D), full_output = False) 
print('fsolve:',solution) #print out what fsolve returned
#
###
##### end 3.
    

##### 4. 
###
#
# define function to calculate F and Q. Its arguments are the before defined matrices 
# and the guessed values for F_new.
def use_time_iteration_to_calculate_F_and_Q(A,M,D,F_guess):
    F = np.zeros((4,4)) #define start values for F
    F_new = F_guess #rename F_guess into F_new to match format from assignment 4
    while np.max(np.abs(F-F_new))>0.0000001: # keep repeating next two lines until
        # difference between last and second-last iteration of F is smaller than
        # 0.0000001.
        F = F_new #update F from F_new of last iteration
        F_new = np.dot(np.linalg.inv(A-np.dot(M,F)),D) #calculate F_new given old F_new
    Q = np.linalg.inv(A-np.dot(M,F_new)) #calculate Q given iterated F_new
    return F, Q #return results for F and Q
#
###
##### end 4.
    

##### 5.  
###
#
F_guess = np.ones((4,4)) #define start values for F_new for the time iteration
F, Q = use_time_iteration_to_calculate_F_and_Q(A,M,D,F_guess) #call our function and
# save what it returns (two matrices) as the the variables F and Q
print('F:',F) # print out the two matrices that we just calculated and their names
print('Q:',Q)
#
###
##### end 5.
    

##### 6.
###
#
# [:,X] slices the X-th column of a 2D-matrix. I will not go into detail how array-
# slicing works because it gets complex very fast. It is common practice to just
# google what you want, eg in my case "numpy get column of matrix", and use code 
# from stackoverflow. 
C_epsilon_mu = Q[:,0] 
C_epsilon_i = Q[:,3]
C_mu = F[:,0]
C_y = F[:,1]
C_pi = F[:,2]
print('C_epsilon_mu:',C_epsilon_mu) # print out the sliced columns togehter with name
print('C_epsilon_i:',C_epsilon_i)
print('C_mu:',C_mu)
print('C_y:',C_y)
print('C_pi:',C_pi)

#
###
##### end 6


##### 7.
###
#
N = 30 #save number of iterations in variable N
# define vectors of N zeros for epsilon_e, epsilon_i, mu, Y, pi, ni (nominal i):
epsilon_mu = np.zeros(N)
epsilon_i = np.zeros(N)
mu = np.zeros(N)
Y = np.zeros(N)
pi = np.zeros(N)
ni = np.zeros(N)
epsilon_mu[0] = 0.01 #epsilon_mu_0 is set to 0.01, ie the first entry of the vector
for i in range(N): #start loop of N periods, where i indicateds the time period t.
# I took the code from assignment 4, but left the list comprehension out, because it
# is unintuitive. In evey iteration i (ie the time period) I calculate mu,Y,pi and ni
# seperately via (5). In the first iteration (i=0) there exist no lagged values for
# z_t. mu[0-1] gives the last entry of the mu vector (last time period), which does 
# not hinder the iteration to converge, but is actually incorrect. I did not handle
# this issue because neither did assignment 4.
    mu[i] = C_mu[0]*mu[i-1] + C_y[0]*Y[i-1] + C_pi[0]*pi[i-1] + \
            C_epsilon_mu[0]*epsilon_mu[i] + C_epsilon_i[0]*epsilon_i[i] 
    Y[i]  = C_mu[1]*mu[i-1] + C_y[1]*Y[i-1] + C_pi[1]*pi[i-1] + \
            C_epsilon_mu[1]*epsilon_mu[i] + C_epsilon_i[1]*epsilon_i[i]
    pi[i] = C_mu[2]*mu[i-1] + C_y[2]*Y[i-1] + C_pi[2]*pi[i-1] + \
            C_epsilon_mu[2]*epsilon_mu[i] + C_epsilon_i[2]*epsilon_i[i]
    ni[i] = C_mu[3]*mu[i-1] + C_y[3]*Y[i-1] + C_pi[3]*pi[i-1] + \
            C_epsilon_mu[3]*epsilon_mu[i] + C_epsilon_i[3]*epsilon_i[i]
z_t_excercise_7 = [mu,Y,pi,ni] #save results in variable to acess them later and re-
# use the namespaces of mu,Y etc in question 8.
#
###
##### end 7 


##### 8.
###
#
N = 30 #save number of iterations in variable N
# define vectors of N zeros for epsilon_e, epsilon_i, mu, Y, pi, ni (nominal i):
epsilon_mu = np.zeros(N)
epsilon_i = np.zeros(N)
mu = np.zeros(N)
Y = np.zeros(N)
pi = np.zeros(N)
ni = np.zeros(N)
epsilon_i[0] = 0.01 #epsilon_i_0 is set to 0.01, ie the first entry of the vector
for i in range(N): #same loop as in 7.
    mu[i] = C_mu[0]*mu[i-1] + C_y[0]*Y[i-1] + C_pi[0]*pi[i-1] + \
            C_epsilon_mu[0]*epsilon_mu[i] + C_epsilon_i[0]*epsilon_i[i] 
    Y[i]  = C_mu[1]*mu[i-1] + C_y[1]*Y[i-1] + C_pi[1]*pi[i-1] + \
            C_epsilon_mu[1]*epsilon_mu[i] + C_epsilon_i[1]*epsilon_i[i]
    pi[i] = C_mu[2]*mu[i-1] + C_y[2]*Y[i-1] + C_pi[2]*pi[i-1] + \
            C_epsilon_mu[2]*epsilon_mu[i] + C_epsilon_i[2]*epsilon_i[i]
    ni[i] = C_mu[3]*mu[i-1] + C_y[3]*Y[i-1] + C_pi[3]*pi[i-1] + \
            C_epsilon_mu[3]*epsilon_mu[i] + C_epsilon_i[3]*epsilon_i[i]
z_t_excercise_8 = [mu,Y,pi,ni] #save results in variable
#
###
##### end 8


##### 9.
###
#
# to not repeat myself, I define a function that takes an array containing the results
# form 7. and 8. respectively and plots them in subplots.
def plot_z_t_in_subplots(z_t):
    mu, Y, pi, ni = z_t[0],z_t[1],z_t[2],z_t[3]# I redefine mu,Y,pi and ni locally to
    # have the input format for plotting like the code in assignment 4.
    plt.figure() #creates a new figure, ie a new blank plot to be filled with graphs.
    plt.subplot(2,2,1) # a figure already exists, this creates subplots in this figure.
    # 2,2 indicates that the figure will have 2x2 grid. ,1 indicates to draw one of the
    # for subplots in position "1", ie upper left. As long as no other subplot() is
    # called, all other plt functions will take place in the last subplot.
    plt.plot([Y[i]*100 for i in range(1,N)],linewidth=4,color='red') # plot() draws a
    # 2D graph, where the y axis is the vector of input data and the x axis (by default
    # if no other data is passed) is an ascending index of integers with the same lentgh
    # as the y data. Here, as input data I pass all but the first elements in the vector
    # Y, where each element is multiplied by 100. linewdith= changes the width of the
    # line of the graph and color= changes the color of the line.
    plt.title('Output') #writes title of subplot
    plt.xlabel('t') #labels x axis
    plt.ylabel('Y_t') #labels y axis
    plt.subplot(2,2,2) #create subplot in postion "2", ie upper right
    plt.plot([pi[i]*100 for i in range(1,N)],linewidth=4,color='blue') #same as berfore,
    # but with pi as input data
    plt.title('pi') #...
    plt.xlabel('t') #...
    plt.ylabel('pi_t') #...
    plt.subplot(2,2,3) #create subplot in postion "3", ie lower left
    plt.plot([ni[i]*100 for i in range(1,N)],linewidth=4, color='green') #...
    plt.title('ni')
    plt.xlabel('t')
    plt.ylabel('ni_t')
    plt.subplot(2,2,4) #create subplot in postion "4", ie lower right
    plt.plot([mu[i]*100 for i in range(1,N)],linewidth=4,color='orange')
    plt.title('mu')
    plt.xlabel('t')
    plt.ylabel('mu_t')
    plt.tight_layout() #this automatically reformats the entire figure to make labels
    # more readable.
# In the next line I call the new function. Once I pass the array of the results from
# from question 7 as the argument when calling the function, and once the results from
# question 8. 
plot_z_t_in_subplots(z_t_excercise_7)
plot_z_t_in_subplots(z_t_excercise_8)
#
###
##### end 9 


##### 10
###
#
print('10.: See pdf for interpretation.')
#
###
##### end 10

##### 11
###
#
epsilon_mu = np.random.normal(0, 1, size=500) #save array of 500 random numbers drawn 
# from a normal distribution with zero mean in the variable epsilon_mu.
epsilon_i = np.random.normal(0, 1, size=500) # do the same for epsilon_i
# loop throuhg both epsilons via the iterator "i" and in each iteration calculate z_t
# via (6) using the c-factors from before. For the lagged values of z_t I use z_t from
# the last iteration. In the first iteration I use start values for z_t.
y, pi, mu, ni = [],[],[],[] # define empty arrays that will be filled in the for loop
# and later turned into the columns of a pandas dataframe.
z_t = np.array([12,-2,1.5,3]) #define start values for z_t.
for i in range(len(epsilon_mu)): # I loop "numerically" to access each value of the two
# shocks via index slicing, ie [i], at the same time.
    #before calculating the new z_t, I save the values from the last iteration (in i=0 
    # the start values) in the arrays y,pi,mu and ni.
    y.append(z_t[1]) 
    pi.append(z_t[2])
    mu.append(z_t[0])
    ni.append(z_t[3])
    # I define the lags of mu, Y and pi from z_t of the last iteration. I do this to for
    # better visibility.
    mu_lag = z_t[0]
    Y_lag = z_t[1] 
    pi_lag= z_t[2]
    # z_t is calculated via (5).
    z_t =   C_epsilon_mu*epsilon_mu[i] + C_epsilon_i*epsilon_i[i] + C_mu*mu_lag + \
            C_y*Y_lag + C_pi*pi_lag
simulation_result = pd.DataFrame({'y':y, 'pi':pi,'mu':mu,'ni':ni}) #turn arrays into
# dataframe via dictionary.

#
###
##### end 11

##### 12
###
#
Et_pi_t_plus_one = [] #define empty array where values for forecasts will be saved.
for i in range(len(epsilon_i)): #same loop logic as before. I access the needed values
# of the simulation_result dataframe via the column name and the row number. the row
# number will be i. Thus I loop trough the entire dataframe. In this loop I will use
# (5), but use t+1 in each element. Also, I take the expected value of each element in
# the equation. Thus, the epsilons will be zero and the hird element of z_t+1 will be 
# the exptected inflation for t+1 forecasted in t.
    # again for better visibility I define the needed variables with the respective 
    # values for Y,pi and mu from the dataframe.
    Y_t = simulation_result['y'][i] 
    pi_t = simulation_result['pi'][i]
    mu_t = simulation_result['mu'][i]
    # I calculate period t's forecast for z in t+1 via (5). I left the elements of u_t
    # in the equation even though they are zero for more visbility. Note that all the
    # elements of (5) are "shifted" towards the future by one period.
    exp_z_t_plus_one =   C_epsilon_mu*0 + C_epsilon_i*0 + C_mu*mu_t + \
            C_y*Y_t + C_pi*pi_t
    Et_pi_t_plus_one.append(exp_z_t_plus_one[2]) #third element in z_t is inflation. I 
    # append it to the array in each iteration.
simulation_result['Et_pi_t_plus_one'] = Et_pi_t_plus_one #add array as column to the
# dataframe.
#
###
##### end 12


##### 13
###
#
# define new column that shifts values of pi one period into the future so that
# in each row we have the realised pi and expected pi of the same period
simulation_result['pi_t_plus_one'] =    simulation_result['pi']\
                                        .shift(periods=-1)
# define new column that is the difference between the realized pi of a period t and its
# expected value from the t-1.
simulation_result['pi_t_plus_one_minus_Et_pi_t_plus_one'] =  \
            simulation_result['pi_t_plus_one'] - simulation_result['Et_pi_t_plus_one']
#
###
##### end 13


##### 14
###
#
# call ols object passing our dataframe as input data and the wanted regression formula
# as arguments and save it as the variable mod.
mod = ols(formula='pi_t_plus_one_minus_Et_pi_t_plus_one ~ pi', data=simulation_result)
# call the method fit() as save what it returns (in this case another object) in the 
# variable fit.
fit = mod.fit()
# call fit's method summary and print in in the console
print(fit.summary())
print('\nSee pdf for interpretation.')
#
###
##### end 14