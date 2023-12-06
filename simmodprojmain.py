import numpy as np
import numpy.linalg as la
import simmodprojforces as fo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import count
from matplotlib.animation import FuncAnimation



# Constants

G = 6.6743 * 10 ** (-11) # Gravitational constant


class Planet():
    def __init__(self, mass, radius0, v0, theta0):
        self.mass = mass
        self.radius0 = radius0 # This is its position from the sun not the radius of the planet, they are points to us
        self.v0 = np.array([0, v0])
        self.theta0 = theta0
        self.x = radius0 * np.cos(theta0)
        self.y = radius0 * np.sin(theta0)
        
        self.poslist = [np.array([self.x, self.y])]
        self.v = [self.v0]
        self.a = []

        self.x = [self.x]
        self.y = [self.y]



class Simulator():
    def __init__(self, sun, planets, N = 365, dt = 3600*24):
        self.N = N # This is weird, changing T and dt changes stuff crazy
        self.sun = sun
        self.planets = planets
        self.dt = dt
        self.T = N * dt

    def RK3(self):
        for i in range(0,3):
            for planet in self.planets:
                hat = -fo.hat(planet.poslist[i], (0, 0))
                k1 = fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat)
                k1v = planet.v[i]

                k2 = fo.acc(self.sun.mass, G,np.add(planet.poslist[i], self.dt*k1v), np.array([0, 0]), hat)
                k2v = np.add(planet.v[i], k1*self.dt)

                k3 = fo.acc(self.sun.mass, G, np.add(planet.poslist[i], self.dt*k1v / 4, self.dt*k2v / 4), np.array([0,0]), hat)
                k3v = np.add(planet.v[i], self.dt * k1 / 4, self.dt * k2 / 4)

                hold = [np.add(planet.v[i], self.dt / 6 * (k1 + k2 + 4*k3)), np.add(planet.poslist[i], self.dt / 6 * (k1v + k2v + 4*k3v))]
                planet.v.append(hold[0])
            
                planet.poslist.append(hold[1])
                planet.x.append(hold[1][0])
                planet.y.append(hold[1][1])

    def AB4(self):
        #planet.poslist = np.zeros(self.N)
        #planet.poslist[0] = (initpos[0], initpos[1])
        a = []
        
        self.RK3()
        
        for i in range(0, 3):
            for planet in self.planets:
                hat = -fo.hat(planet.poslist[i], (0, 0))
                planet.a.append(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0,0]), hat))

        for i in range(3, self.N):
            for planet in self.planets:
                hat = -fo.hat(planet.poslist[i], (0, 0))
                planet.a.append(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat))
                # Maybe make v0 an arrey so that AB4 on the velocity as well
                hold = [np.add(planet.poslist[i], self.dt * (55 * planet.v[i] - 59 * planet.v[i-1] + 37 * planet.v[i-2] - 9 * planet.v[i-3]) / 24), np.add(planet.v[i], self.dt * (55*planet.a[i] - 59*planet.a[i-1] + 37*planet.a[i-2] - 9*planet.a[i-3]) / 24)]
                planet.v.append(hold[1]) # TODO: V IS WAY TO LARGE I THINK
                
                planet.poslist.append(hold[0])
                planet.x.append(hold[0][0])
                planet.y.append(hold[0][1])

            

    
    def solarsys_anim(self):
        # subplots() function you can draw 
        # multiple plots in one figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        
        # set limit for x and y axis
        axes.set_ylim(-250 * 10 ** 9, 250 * 10 ** 9)
        axes.set_xlim(-250 * 10 ** 9, 250 * 10 ** 9)
        
        # style for plotting line
        plt.style.use("ggplot")
        
        # create 5 list to get store element 
        # after every iteration
        x1, x2, x3, x4, y1, y2, y3, y4 = [], [], [], [], [], [], [], []
        
        def animate(i):
            x1.append(self.planets[0].x[i])
            x2.append(self.planets[1].x[i])
            x3.append(self.planets[2].x[i])
            x4.append(self.planets[3].x[i])
            y1.append(self.planets[0].y[i])
            y2.append(self.planets[1].y[i])
            y3.append(self.planets[2].y[i])
            y4.append(self.planets[3].y[i])
            axes.plot((0,0), 'y*')
            axes.plot(x1, y1, color="red")
            axes.plot(x2, y2, color="gray")
            axes.plot(x3, y3, color="blue")
            axes.plot(x4, y4, color="green")
            axes.set_title('Day ' + str(i))
        
            axes.legend(['Sun', 'Mercury Orbit','Venus Orbit', 'Earths Orbit', 'Mars Orbit'])
        
        
        # set ani variable to call the 
        # function recursively
        anim = FuncAnimation(fig, animate, interval=20)
        plt.show()

        


def main():
    # TODO: Add forces between asteroid and plaents and sun, add all planets, maybe make animation fade, idk though this might be good, also have the normal plot whcich is good for final to show pictures in report
    sun = Planet(1.98892 * 10**30, 0, 0, 0)
    mercury = Planet(0.03301 * 10 ** 24 , 69.818 * 10 ** 9, 38860, 0)
    venus = Planet(4.8673 * 10 ** 24, 108.941 * 10 ** 9, 34780, 0)
    earth = Planet(5.9722 * 10 ** 24, 152.1 * 10 ** 9, 29290, 0)
    mars = Planet(6.4174 * 10 ** 23, 249.261 * 10 ** 9, 21970, 0)
    planets = [mercury, venus, earth, mars]

    sim = Simulator(sun, planets, N=365)
    sim.AB4()
    sim.solarsys_anim()
    
    #plt.plot((0, 0), '*y')
    #plt.plot(*zip(*mercury.poslist))
    #plt.plot(*zip(*venus.poslist))
    #plt.plot(*zip(*earth.poslist))
    #plt.plot(*zip(*mars.poslist))
    #plt.legend(['Sun', 'Mercury Orbit','Venus Orbit', 'Earths Orbit', 'Mars Orbit'])
    #ax = plt.gca()
    #ax.set_aspect('equal', adjustable='box')
    #plt.show()



if __name__ == "__main__":
    main()
