import numpy as np
import numpy.linalg as la
import simmodprojforces as fo
import matplotlib.pyplot as plt
import matplotlib as animation


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
        self.v = [v0]

        self.xlist = [self.x]
        self.ylist = [self.y]


    def attraction(self, other):
        other_x, other_y = other.x, other.y
        distance_x = other_x - self.x
        distance_y = other_y - self.y
        distance = (distance_x ** 2 + distance_y ** 2) ** (-.5)



class Simulator():
    def __init__(self, sun, earth, N = 300, dt = 3600*24):
        self.N = N # This is weird, changing T and dt changes stuff crazy
        self.sun = sun
        self.earth = earth
        self.dt = dt
        self.T = N * dt

    def RK3(self):
        for i in range(0,3):
            hat = -fo.hat(self.earth.poslist[i], (0, 0))
            k1 = fo.acc(self.sun.mass, G, self.earth.poslist[i], np.array([0, 0]), hat)
            k1v = self.earth.v0

            k2 = fo.acc(self.sun.mass, G,np.add(self.earth.poslist[i], self.dt*k1v), np.array([0, 0]), hat)
            k2v = np.add(self.earth.v0, k1*self.dt)

            k3 = fo.acc(self.sun.mass, G, np.add(self.earth.poslist[i], self.dt*k1v / 4, self.dt*k2v / 4), np.array([0,0]), hat)
            k3v = np.add(self.earth.v0, self.dt * k1 / 4, self.dt * k2 / 4)

            hold = [np.add(self.earth.v0, self.dt / 6 * (k1 + k2 + 4*k3)), np.add(self.earth.poslist[i], self.dt / 6 * (k1v + k2v + 4*k3v))]
            self.earth.v.append(hold[0])
        
            self.earth.poslist.append(hold[1])

    def AB4(self):
        initpos = self.earth.poslist[0]
        #self.earth.poslist = np.zeros(self.N)
        #self.earth.poslist[0] = (initpos[0], initpos[1])
        a = []
        self.RK3()

        for i in range(0, 3):
            hat = -fo.hat(self.earth.poslist[i], (0, 0))
            a.append(fo.acc(self.sun.mass, G, self.earth.poslist[i], np.array([0,0]), hat))

        for i in range(3, self.N):
            hat = -fo.hat(self.earth.poslist[i], (0, 0))
            a.append(fo.acc(self.sun.mass, G, self.earth.poslist[i], np.array([0, 0]), hat))
            # Maybe make v0 an arrey so that AB4 on the velocity as well
            hold = [np.add(self.earth.poslist[i], self.dt / 24 * (55 * self.earth.v[i] -59 * self.earth.v[i-1] + 37 * self.earth.v[i-2] - 9 * self.earth.v[i-3])), np.add(self.earth.v0, self.dt / 24 * (55*a[i] - 59*a[i-1] + 37*a[i-2] - 9*a[i-3]))]
            
            self.earth.v.append(hold[1]) # TODO: V IS WAY TO LARGE I THINK
            print(la.norm(a[i]))
            self.earth.poslist.append(hold[0])
            
            


def main():
    sun = Planet(1.98892 * 10**30, 0, 0, 0)
    earth = Planet(5.972 * 10 ** 24, 149.6e6 * 1000, 29783, 0)
    Simulator(sun, earth).AB4()
    plt.plot((0, 0), '*y')
    plt.scatter(*zip(*earth.poslist))
    plt.legend(['Sun', 'Earths orbit'])
    plt.show()



if __name__ == "__main__":
    main()
