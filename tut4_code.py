import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

### Fsolve

# define function with non-linear equations
 
def nonlin(X):
    x,y,z = X
    z1 = z-x-y
    z2 = y*z*x-8
    z3 = 0 # see hint in exercise 
    return [z1,z2,z3] 

init=[1,0,0]
coefficients,inf,ier,mes=fsolve(nonlin,init,maxfev=100000,full_output=True)
print (coefficients,ier, mes)

nonlin(coefficients)

init=[1,0,0]
[coefficients,inf,ier,mes]=fsolve(nonlin,init,maxfev=100000,full_output=True)
print (coefficients,ier, mes)

init=[-3,1.5,-1.5]
[coefficients,inf,ier,mes]=fsolve(nonlin,init,maxfev=100000,full_output=True)
print (coefficients,ier, mes)

init=[0,0,0]
[coefficients,inf,ier,mes]=fsolve(nonlin,init,maxfev=100000,full_output=True)
print (coefficients,ier, mes)

# non-linear system of equations has multiple solutions
# depending on initial guess, we may converge to different ones


















### simple linear regression

# set a random seed in order to reproduce your results
np.random.seed(456)

# numer of observations
n = 20

# create artificial data set x
x = np.random.random(n)
print(x)

# create normally distributed errors
epsilon= np.random.randn(n)

# create your artifical y data
y = 2.3 + 5.5*x + epsilon
# y[19] = 20 # see how outlier affect estimation



print(epsilon)







# plot x and y
plt.figure()
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')










# define a function that computes both beta1 and beta2
# given data for x and y
# let the function return the estimated coefficients

print(np.cov(x,y,ddof=1))
print(np.var(x,ddof=1))


def simlinreg(y_data,x_data):
    beta1 = np.cov(x_data,y_data)[0,1]/np.cov(x_data,y_data)[0,0]
    beta0 = np.mean(y_data) - beta1*np.mean(x_data)
    coefs = [beta0,beta1]
    return coefs

coefs = simlinreg(y,x)
print(coefs)

# define a function that predicts y given the estimated coefficients and 
# x data, return the predictions










def ypredict(x,coefs):
    beta0,beta1 = coefs
    yhat = beta0 + beta1*x
    return yhat

# plot the predicted line in the scatter plot
yhat = ypredict(x,coefs)
#plt.figure()
#plt.scatter(x,y)
plt.plot(x, yhat)
#plt.xlabel('x')
#plt.ylabel('y')







