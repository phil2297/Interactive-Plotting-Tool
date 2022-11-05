import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw

def linear_data(x, a, b, *args):
    y = a*x + b
    noise = np.std(y)/5 * np.random.normal(size=x.size)
    return y + noise, noise

def polynomial_data(x, *args):
    consts = [constant for constant in args]
    tempy = np.array([consts[i]*(x**(len(consts)-i)) for i in range(len(consts))])
    y = 0
    for i in range(len(tempy)):
        y += tempy[i]
    noise = np.std(y)/5 * np.random.normal(size=x.size)
    return y + noise, noise

def exponential_data(x, a, b, *args):
    y = b * np.exp(a*x)
    noise = np.std(y)/5 * np.random.normal(size=x.size)
    return y + noise, noise

def powerlaw_data(x, a, b, *args):
    y = np.power(x, 6)
    noise = np.std(y)/5 * np.random.normal(size=x.size)
    return y + noise, noise

def sinusodial_data(x, *args):
    y = np.sin(x)
    noise = np.std(y)/5 * np.random.normal(size=x.size)
    return y + noise, noise

def gaussian_data(x, *args):
    y = np.random.normal(size=x.size)
    noise = np.zeros(len(x))
    return y + noise, noise

def randconst():
    return np.random.uniform(low=-10, high=10, size=1)

def save_data(F, filename):
    seed = np.random.seed(int(np.random.random()*100))
    x = np.random.uniform(low=-10, high=10, size=100)
    a = randconst()
    b = randconst()
    c = randconst()
    x = np.sort(x)
    Fdat, Ferr = F(x, a, b, c)
    np.savetxt(f'Test Data/Fit Tests/{filename}.dat', np.array([x, Fdat, Ferr]).T)
    
save_data(linear_data, 'linear')
save_data(polynomial_data, 'polynomial')
save_data(exponential_data, 'exponential')
save_data(powerlaw_data, 'powerlaw')
save_data(sinusodial_data, 'sine')
save_data(gaussian_data, 'gaussian')
