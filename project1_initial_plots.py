import numpy as np
import matplotlib.pyplot as plt

data_to_fit = np.loadtxt("data_to_fit.txt",skiprows=2)
unoscillated_data = np.loadtxt("unoscillated_flux.txt",skiprows=2)
energies = np.arange(0.025,10,0.05)

def plot_hists(experimental,unoscillated,energies):
    plt.bar(energies, data_to_fit,align = 'center',width = 0.05,alpha = 0.8)
    plt.title("Experimental Data")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Number")
    #plt.savefig('experimental_data.png')
    plt.show()
    plt.bar(energies, unoscillated_data,align='center',width = 0.05,alpha = 0.8)
    plt.title("Unoscillated Data")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Energy Rate")
    #plt.savefig('unoscillated_events.png')
    plt.show()

def P(E,params):
    theta,dm,L = params[0],params[1],params[2]
    A = np.sin(2*theta)
    B = np.sin((1.267*dm*L)/E)
    return 1 - (A*B)**2

plot_hists(data_to_fit,unoscillated_data,energies)

'''
Variation of parameters:
theta changes periodically, max at pi/4, min at pi/2. Changes range of P
dm and L change periodically, stretches P horizontally and creates more fluctuations from E=0
'''
# Keep theta as pi/4 because some values need to go to 0
'''
Fairly optimum parameters:
theta = 0.7
dm = 2e-3
'''
theta = np.pi/4
dm = 2.4e-3
L = 295

args = [theta,dm,L]
P_neutrino = P(energies,args)
plt.plot(energies,P_neutrino)
plt.xlabel("E (GeV)")
plt.ylabel(r'$P(\nu_\mu \to \nu_\mu)$')
plt.title("Probability Function with initial parameters")
#plt.savefig("initial_P.png")
plt.show()

fitted_unoscillated_data = np.multiply(unoscillated_data,P_neutrino)
plt.bar(energies,fitted_unoscillated_data,align='center',width=0.05,alpha=0.8)
plt.xlabel("Energy (GeV)")
plt.ylabel("Number of events")
plt.title("Energy Distribution for initial parameters")
#plt.savefig("inital_param_dist.png")
plt.show()

theta = 0.7 # initial: np.pi/4
dm = 2e-3 # initial: 2.4e-3
L = 295

args = [theta,dm,L]
P_neutrino = P(energies,args)
plt.plot(energies,P_neutrino)
plt.xlabel("E (GeV)")
plt.ylabel(r'$P(\nu_\mu \to \nu_\mu)$')
plt.title("Probability Function with optimised parameters")
#plt.savefig("P1.png")
plt.show()

fitted_unoscillated_data = np.multiply(unoscillated_data,P_neutrino)
plt.bar(energies,fitted_unoscillated_data,align='center',width=0.05,alpha=0.8)
plt.xlabel("Energy (GeV)")
plt.ylabel("Number of events")
plt.title("Energy Distribution for optimised parameters")
#plt.savefig("dist1.png")
plt.show()
