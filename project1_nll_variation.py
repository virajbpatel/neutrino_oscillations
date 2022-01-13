# Project 1: Neutrino Oscillations
# CID: 01709594

import numpy as np
import matplotlib.pyplot as plt

data_to_fit = np.loadtxt("data_to_fit.txt",skiprows=2)
unoscillated_data = np.loadtxt("unoscillated_flux.txt",skiprows=2)
energies = np.arange(0.025,10,0.05)

def P(E,params):
    theta,dm,L = params[0],params[1],params[2]
    A = np.sin(2*theta)
    B = np.sin((1.267*dm*L)/E)
    return 1 - (A*B)**2

def NLL(l,m):
    n = len(l)
    nll = 0
    for i in range(n):
        if l[i] != 0 and m[i] != 0:
            nll += l[i] - m[i] + m[i]*np.log(m[i]/l[i])
    return nll

def find_nll(params):
    p = P(energies,params)
    return NLL(np.multiply(p,unoscillated_data),data_to_fit)

def NLL_minimise(params,m,index,arg_range):
    arg = np.arange(arg_range[0],arg_range[1],arg_range[2])
    nll_array = []
    for a in arg:
        params[index] = a
        p = P(energies,params)
        l = np.multiply(unoscillated_data,p)
        nll = NLL(l,m)
        nll_array.append(nll)
    return arg, np.array(nll_array)

def nll_theta(t):
    p = P(energies,[t,2e-3,295])
    return NLL(np.multiply(p,unoscillated_data),data_to_fit)

def nll_dm(dm):
    p = P(energies,[np.pi/4,dm,295])
    return NLL(np.multiply(p,unoscillated_data),data_to_fit)

def parabolic_minimiser(f,p_0,axis):
    x_0 = [p_0[0][axis],p_0[1][axis],p_0[2][axis]]
    x_min = x_0[1]
    err = 1
    while err > 1e-6:
        x_0 = sorted(x_0)
        y_0 = []
        for i in x_0:
            temp = p_0[0]
            temp[axis] = i
            y_0.append(f(temp))
        x = 0.5*((x_0[2]**2-x_0[1]**2)*y_0[0]+(x_0[0]**2-x_0[2]**2)*y_0[1]+(x_0[1]**2-x_0[0]**2)*y_0[2])/((x_0[2]-x_0[1])*y_0[0]+(x_0[0]-x_0[2])*y_0[1]+(x_0[1]-x_0[0])*y_0[2])
        err = abs((x-x_min)/x_min)
        x_min = x
        px = p_0[0]
        px[axis] = x
        y = f(px)
        y_0.append(y)
        x_0.append(x)
        p = y_0.index(max(y_0))
        y_0.pop(p)
        x_0.pop(p)
    x_0 = sorted(x_0)
    y_0 = []
    for i in x_0:
        temp = p_0[0]
        temp[axis] = i
        y_0.append(f(temp))
    return x_min,x_0,y_0

def find_error(t,lim):
    t0 = t
    while nll_theta(t) < lim:
        t -= 0.0001
    t_m = t
    while nll_theta(t0) < lim:
        t0 += 0.0001
    t_p = t0
    return t_m,t_p

def curvature_err(x,y):
    d2x = 2*(y[0]/((x[0]-x[1])*(x[0]-x[2])) + y[1]/((x[1]-x[0])*(x[1]-x[2])) + y[2]/((x[2]-x[1])*(x[2]-x[0])))
    std = np.sqrt(1/abs(d2x))
    return std

theta = np.pi/4
dm = 2e-3 # initial: 2.4e-3
L = 295
args = [theta,dm,L]
# Vary theta for fixed dm and L, gave theta_min = 0.661 and 0.884
theta_arr, nll_arr = NLL_minimise(args,data_to_fit,0,[0,2*np.pi,0.01])

plt.plot(theta_arr,nll_arr)
plt.xlabel(r'$\theta_{23}$')
plt.ylabel("NLL")
plt.title(r'NLL variation with $\theta_{23}$')
#plt.savefig("theta_variation.png")
plt.show()

# Vary dm for fixed theta and L, gave dm_min = 2.41e-3
dm_arr, nll_arr_dm = NLL_minimise(args,data_to_fit,1,[0,5e-3,1e-4])

plt.plot(dm_arr,nll_arr_dm)
plt.xlabel(r'$\Delta m_{23}^2$')
plt.ylabel("NLL")
plt.title(r'NLL variation with $\Delta m_{23}^2$')
#plt.savefig("dm_variation.png")
plt.show()

# Parabolic minimisation
e_theta = 1
e_dm = 1
theta = 0.661
dm = 2.41e-3
while e_theta > 1e-6 or e_dm > 1e-6:
    params = [theta,dm,L]
    # Parabolic minimisation of theta
    theta_vals = [0.9*theta,theta,1.1*theta]
    p_t = [[t,dm,L] for t in theta_vals]
    theta_min,t,nll = parabolic_minimiser(find_nll,p_t,0)
    '''
    # Find error of theta_min: ~0.0108
    nll_lim = nll_theta(theta_min) + 0.5
    theta_minus, theta_plus = find_error(theta_min,nll_lim)
    err1, err2 = abs(theta_min-theta_minus), abs(theta_min-theta_plus)
    print(err1,err2)
    '''
    # Find error of theta_min using curvature: ~0.0108
    curve_err = curvature_err(t,nll)
    #print(curve_err)
    e_theta = abs((theta-theta_min)/theta_min)
    theta = theta_min

    # Parabolic minimisation of dm
    dm_vals = [0.9*dm,dm,1.1*dm]
    p_d = [[theta,d,L] for d in dm_vals]
    dm_min,d,nll = parabolic_minimiser(find_nll,p_d,1)
    # Find error of dm_min using curvature: ~2.31e-5
    curve_err_dm = curvature_err(d,nll)
    #print(curve_err_dm)
    e_dm = abs((dm-dm_min)/dm_min)
    dm = dm_min

print("Univariate theta_min: ", theta_min) # 0.705
print("theta_min error: ", curve_err) # 0.0108
print("Univariate dm_min: ", dm_min) # 2.56e-3
print("dm_min error: ", curve_err_dm) # 3.97e-5
