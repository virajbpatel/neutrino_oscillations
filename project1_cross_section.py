# Project 1: Neutrino Oscillations
# CID: 01709594

import numpy as np
import matplotlib.pyplot as plt

data_to_fit = np.loadtxt("data_to_fit.txt",skiprows=2)
unoscillated_data = np.loadtxt("unoscillated_flux.txt",skiprows=2)
energies = np.arange(0.025,10,0.05)

def P(E,params):
    theta,dm,alpha,L = params[0],params[1],params[2],params[3]
    A = np.sin(2*theta)
    B = np.sin((1.267*dm*L)/E)
    return alpha*(1 - (A*B)**2)

def NLL(l,m):
    n = len(l)
    nll = 0
    for i in range(n):
        if l[i] != 0 and m[i] != 0:
            nll += l[i] - m[i] + m[i]*np.log(m[i]/l[i])
    return nll

def find_nll(params):
    p = P(energies,params)
    temp = np.multiply(unoscillated_data,energies)
    return NLL(np.multiply(p,temp),data_to_fit)

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
alpha = 1
L = 295
params = [theta,dm,alpha,L]
step = [0.001,0.01e-3,0.01]
params_min, err = newton_method(find_nll,params,step)
print("------------------\nNewton's Method\n------------------")
print("Minimised parameters (theta,dm,alpha,L): ", params_min)
print("Errors (theta,dm,alpha): ", err)
print("NLL: ", find_nll(np.append(params_min,295)))

'''
Gives theta_min = 0.699 +/- 4.96e-5
dm_min = 2.84e-3 +/- 1.30e-5
alpha_min = 1.08 +/- 6.63e-2
NLL = 58.92
'''
import random as rnd
from tqdm import tqdm
import time

num_itt = 100000

def update_func(parameters):
    t = parameters[0] + np.random.normal(0.0,0.001)
    m = parameters[1] + np.random.normal(0.0,0.01e-3)
    a = parameters[2] + np.random.normal(0.0,0.01)
    return np.array([t,m,a])

def accept_func(dE,T):
    if dE < 0.0:
        return 1
    elif rnd.random() < np.exp(-1*dE/T):
        return 1
    return 0

def run_chain(t,s):
    T = t
    temps = np.arange(T,0.0,-s)
    parameter_chain = np.zeros((num_itt*len(temps)+1,3),float,'C')
    parameter_chain[0,0] = 0.689
    parameter_chain[0,1] = 2.41e-3
    parameter_chain[0,2] = 1
    parameters = np.zeros(3)
    currentNLL = find_nll(np.append(parameter_chain[0],L))
    for i in range(len(temps)):
        temp = temps[i]
        print("T: ",temp)
        parameters[0] = parameter_chain[num_itt*i,0]
        parameters[1] = parameter_chain[num_itt*i,1]
        parameters[2] = parameter_chain[num_itt*i,2]
        currentNLL = find_nll(np.append(parameters,L))
        for st in tqdm(range(1,num_itt+1)):
            new_parameters = update_func(parameters)
            newNLL = find_nll(np.append(new_parameters,L))
            dE = newNLL - currentNLL
            if accept_func(dE,temp):
                currentNLL = newNLL
                parameters = new_parameters
            parameter_chain[num_itt*i+st,0] = parameters[0]
            parameter_chain[num_itt*i+st,1] = parameters[1]
            parameter_chain[num_itt*i+st,2] = parameters[2]
    return parameter_chain

print("------------------\nMetropolis Algorithm\n------------------")
paramChain = run_chain(3,0.5)
print("Minimised parameters (theta, dm, alpha): ", paramChain[-1])
theta_arr = paramChain[4*num_itt:, 0]
dm_arr = paramChain[4*num_itt:, 1]
alpha_arr = paramChain[4*num_itt:, 2]
err = [np.std(theta_arr),np.std(dm_arr),np.std(alpha_arr)]
print("Errors (theta, dm, alpha): ", err)
print("NLL: ",find_nll(np.append(paramChain[-1],295)))

paramChain = run_chain(5,1)
print("Minimised parameters (theta, dm, alpha): ", paramChain[-1])
theta_arr = paramChain[4*num_itt:, 0]
dm_arr = paramChain[4*num_itt:, 1]
alpha_arr = paramChain[4*num_itt:, 2]
err = [np.std(theta_arr),np.std(dm_arr),np.std(alpha_arr)]
print("Errors (theta, dm, alpha): ", err)
print("NLL: ",find_nll(np.append(paramChain[-1],295)))

'''
T = 5:
Gave theta_min = 0.704 +/- 0.015
dm_min = 2.77e-3 +/- 4.57e-5
alpha_min = 1.04 +/- 0.06
NLL = 60.89

T = 3:
theta_min = 0.702 +/- 0.012
dm_min = 2.81e-3 +/- 3.87e-5
alpha_min = 1.09 +/- 0.05
NLL = 59.18
'''
#print(find_nll([0.702,2.81e-3,1.09,295]))