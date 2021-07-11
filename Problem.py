import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as pl

from Solvers import explicitEuler
from Solvers import implicitEuler
from Solvers import heun
from Solvers import rungeKutta


class Lorenz:
    def __init__(self):
        self.sgm = 10.0
        self.rho = 28.0
        self.bta = 8.0/3.0
        
        self.count = 0
        
    def __call__(self,X,t):
        x,y,z = X
        
        self.count += 1
        
        xd = self.sgm*(y-x)
        yd = x*(self.rho-z)-y
        zd = x*y-self.bta*z
        return np.array([xd,yd,zd])

class Transport:
    def __init__(self):
        self.Q = np.array([[-200,0,0],[200,-200,0],[0,200,-200]])
        self.G = np.array([2000 + 200*2,1000,2000])
        self.V = np.array([150,150,300])
        self.E = np.array([[-25,25,0],[25,-75,50],[0,50,-50]])
        self.T = self.Q + self.E
        
    def __call__(self, y, t):
        return (self.T @ y + self.G)/self.V
    
class Orbit:
    au = 149.6e9
    G = 6.674e-11
    M = 1.99e30
    
    def __call__(self, y, t):
        # y = [x, y, u ,v]
        def acc(x):
            r = np.sqrt((x*x).sum())
            return -self.G * self.M * x[0:2] / r**3
        x = y[0:2]
        v = y[2:4]
        a = acc(x)
        df = np.concatenate((v,a))
        return df

def TransportPlot():
    y = np.array([0, 0, 0])
    t = np.linspace(0, 12, 20)
    transport = Transport()
    sol = [explicitEuler(transport, y, t), implicitEuler(transport, y, t), heun(transport, y, t), rungeKutta(transport, y, t)]
    solNames = ["Explicit Euler", "Implicit Euler", "Heun", "Runge Kutta"]
    
    fig = pl.figure('Transport Problem', tight_layout=True, figsize = (8, 5))
    ax = fig.subplots(3, 1, sharex = True)
    ax[-1].set_xlabel('Time, t [-]')
    rooms = ['Kitchen', 'Bar', 'Seating']
    
    for k in range(3):
        ax[k].set_ylabel(rooms[k])
        for i in range(4):
            ax[k].plot(t, sol[i][:, k], label = solNames[i])
            
    ax[0].legend(title = 'Solver', loc = 'upper left', bbox_to_anchor=(1, 1))
    
    return fig
    
def LorenzPlot():
    t = np.linspace(0,10,500)
    x0 = np.array([5,5,5])
    lorenz = Lorenz()
    sol = [explicitEuler(lorenz, x0, t), implicitEuler(lorenz, x0, t), heun(lorenz, x0, t), rungeKutta(lorenz, x0, t)]
    solNames = ["Explicit Euler", "Implicit Euler", "Heun", "Runge Kutta"]
    fig = pl.figure(tight_layout = True)
    ax = fig.subplots(3, 1)
    ax[-1].set_xlabel('Time, t [-]')
    for k in range(3):
        ax[k].set_ylabel(f'${"xyz"[k]}$ [-]')
        for i in range(4):
            ax[k].plot(t, sol[i][:, k], label = solNames[i])
    

    
    ax[0].legend(title = 'Solver', loc='upper left', bbox_to_anchor=(1,1))
    
    return fig

def OrbitPlot(years=1):
    au = 149.6e9
    G = 6.674e-11
    M = 1.99e30
    
    x0 = np.array([au, 0, 0, np.sqrt(G*M/au)])
    y0 = 60*60*24*365 # seconds in a year
    days = 365
    t = np.linspace(0, years * y0, days * years)
    orbit = Orbit()
    
    sol = [explicitEuler(orbit, x0, t), implicitEuler(orbit, x0, t), heun(orbit, x0, t), rungeKutta(orbit, x0, t)]
    solNames = ["Explicit Euler", "Implicit Euler", "Heun", "Runge Kutta"]
    
    fig = pl.figure(tight_layout = True, figsize = (8, 5))
    ax = fig.subplots(1 , 1)
    ax.set_xlabel('Position, x [au]')
    ax.set_ylabel('Position, y [au]')
    ax.set_aspect('equal')
    for i in range(4):
        ax.plot(sol[i][:, 0]/au, sol[i][:, 1]/au, label = solNames[i])
    ax.legend(title = 'Solver', loc='upper left', bbox_to_anchor=(1,1))
    
    return fig
    

def main():
    fig1 = LorenzPlot()
    fig2 = TransportPlot()
    fig3 = OrbitPlot(3)
    pl.show()
    print("Done")

if __name__=='__main__': main()
