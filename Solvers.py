import numpy as np
import scipy.optimize as sp

def explicitEuler(df, y0, t):
    y = np.empty([t.size, y0.size])
    y[0, :] = y0
    
    for i in range(t.size - 1):
        dt = t[i+1] - t[i]
        y[i+1,:] = y[i,:] + df(y[i,:],t[i]) * dt
    return y

def implicitEuler(df, y0, t):
    y = np.empty([t.size, y0.size])
    y[0, :] = y0
    
    for i in range(t.size - 1):
        dt = t[i+1] - t[i]
        y1 = y[i,:] + df(y[i,:],t[i+1]) * dt
        def R(yn): 
            return y[i,:] + df(yn,t[i+1]) * dt - yn
        y[i+1,:] = sp.fsolve(R, y1)
    return y

def heun(df, y0, t):
    y = np.empty([t.size, y0.size])
    y[0, :] = y0
    
    for i in range(t.size - 1):
        dt = t[i+1] - t[i]

        k1 = df(y[i, :], t[i])
        k2 = df(y[i, :] + k1 * dt, t[i+1])
        y[i+1, :] = y[i,:] + dt/2 * (k1 + k2)

    return y

def rungeKutta(df, y0, t):
    y = np.empty([t.size, y0.size])
    y[0, :] = y0
    
    for i in range(t.size - 1):
        dt = t[i+1] - t[i]
        ti = t[i] + dt/2
        k1 = df(y[i,:], t[i])
        k2 = df(y[i, :] + dt/2 * k1, ti)
        k3 = df(y[i, :] + dt/2 * k2, ti)
        k4 = df(y[i, :] + dt * k3, t[i+1])
        y[i+1, :] = y[i,:] + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y