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

def finite_diff(f,x,h,axis):
    x_1 = np.copy(x)
    x_1[axis] += h
    x_2 = np.copy(x)
    x_2[axis] -= h
    return (f(x_1)-f(x_2))/(2*h)

def grad(f,x,dx):
    Df = []
    e = []
    for i in range(len(x)-1):
        Df.append(finite_diff(f,x,dx[i],i))
        e.append(dx[i]**2)
    return Df,e

def hessian(f,x,dx):
    H = np.zeros((len(x)-1,len(x)-1))
    e = np.copy(H)
    for i in range(len(x)-1):
        for j in range(len(x)-1):
            x_1 = np.copy(x)
            x_1[i] += dx[i]
            x_2 = np.copy(x)
            x_2[i] -= dx[i]
            H[i][j] = (finite_diff(f,x_1,dx[j],j) - finite_diff(f,x_2,dx[j],j))/(2*dx[i])
            e[i][j] = (dx[i]+dx[j])**3
    return H,e

def check_convergence(p,p_0,criteria):
    for i in range(len(p_0)-1):
        if abs((p_0[i]-p[i])/p_0[i]) > criteria:
            return False
    return True

def newton_method(f,params,step):
    p = [0]*(len(params)-1)
    p_0 = np.array(params[:-1])
    Df,e_Df = np.array(grad(f,params,step))
    H,e_H = hessian(f,params,step)
    H_inv = np.linalg.inv(H)
    e = np.matmul(H_inv,e_Df) + np.matmul(e_H,Df)
    p = p_0 - np.matmul(H_inv,Df)
    while not check_convergence(p,params,1e-6):
        p_0 = np.copy(p)
        params = np.append(p_0,params[-1])
        Df,e_Df = np.array(grad(f,params,step))
        H,e_H = hessian(f,params,step)
        H_inv = np.linalg.inv(H)
        e += np.matmul(H_inv,e_Df) + np.matmul(e_H,Df)
        p = p_0 - np.matmul(H_inv,Df)
    return p,e

theta = 0.689
dm = 2.41e-3
L = 295
params = [theta,dm,L]
step = [0.001,0.01e-3]
params_min, err = newton_method(find_nll,params,step)
print("Minimised parameters (theta,dm,L): ", params_min)
print("Errors (theta,dm): ", err)
'''
Gives theta_min = 0.705 +/- 9.02e-5
dm_min = 2.57e-3 +/- 7.76e-8
NLL = 327.5
'''
