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
    def __init__(self, mass, radius0, v, theta0):
        self.mass = mass
        self.radius0 = radius0 # This is its position from the sun not the radius of the planet, they are points to us
        self.theta0 = theta0
        self.x = radius0 * np.cos(theta0)
        self.y = radius0 * np.sin(theta0)
        
        self.poslist = [np.array([self.x, self.y])]
        self.v = [v]
        self.a = []

        self.x = [self.x]
        self.y = [self.y]



class Simulator():
    def __init__(self, sun, planets, asteroid, N = 365, dt = 3600*24):
        self.N = N # This is weird, changing T and dt changes stuff crazy
        self.sun = sun
        self.planets = planets
        self.dt = dt
        self.T = N * dt
        self.asteroid = asteroid

    def RK3(self):
        for i in range(0,3):
            
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0, 0]))
            k1 = fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0, 0]), hata)
            k1tot = k1
            k1v = np.add(self.asteroid.v[i], np.array([0,0]))
            k1vtot = k1v

            k2 = fo.acc(self.sun.mass, G, np.add(self.asteroid.poslist[i], self.dt*k1v), np.array([0, 0]), hata)
            k2tot = k2
            k2v = np.add(self.asteroid.v[i], k1*self.dt)
            k2vtot = k2v

            k3tot = fo.acc(self.sun.mass, G, np.add(np.add(self.asteroid.poslist[i], self.dt*k1v / 4), self.dt*k2v / 4), np.array([0,0]), hata)
            k3vtot = np.add(np.add(self.asteroid.v[i], self.dt * k1 / 4), self.dt * k2 / 4)
            


            for planet in self.planets:
                
                hat = -fo.hat(planet.poslist[i], np.array([0, 0]))
                k1 = fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat)
                k1v = np.add(planet.v[i], [0, 0])

                k2 = fo.acc(self.sun.mass, G,np.add(planet.poslist[i], self.dt*k1v), np.array([0, 0]), hat)
                k2v = np.add(planet.v[i], k1*self.dt)

                k3 = fo.acc(self.sun.mass, G, np.add(np.add(planet.poslist[i], self.dt*k1v / 4), self.dt*k2v / 4), np.array([0,0]), hat)
                k3v = np.add(np.add(planet.v[i], self.dt * k1 / 4), self.dt * k2 / 4)

                hold = [np.add(planet.v[i], self.dt / 6 * (k1 + k2 + 4*k3)), np.add(planet.poslist[i], self.dt / 6 * (k1v + k2v + 4*k3v))]
                planet.v.append(hold[0])
            
                planet.poslist.append(hold[1])
                planet.x.append(hold[1][0])
                planet.y.append(hold[1][1])

                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])

                k1 =  fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata)
                k1tot += k1
                k1v = self.asteroid.v[i]
                k1vtot += k1v


                k2 = fo.acc(planet.mass, G, np.add(self.asteroid.poslist[i], self.dt*k1v), planet.poslist[i], hata)
                k2tot += k2
                k2v = np.add(self.asteroid.v[i], k1*self.dt)
                k2vtot += k2v
                

                k3 = fo.acc(planet.mass, G, np.add(np.add(self.asteroid.poslist[i], self.dt*k1v / 4), self.dt*k2v / 4), planet.poslist[i], hata)
                k3tot += k3
                k3v = np.add(np.add(self.asteroid.v[i], self.dt * k1 / 4), self.dt * k2 / 4)
                k3vtot += k3v
                

            

            holda = [np.add(self.asteroid.v[i], self.dt / 6 * (k1tot + k2tot + 4*k3tot)), np.add(self.asteroid.poslist[i], self.dt / 6 * (k1vtot + k2vtot + 4*k3vtot))]
            
            
            self.asteroid.v.append(holda[0])
            
            self.asteroid.poslist.append(holda[1])
            self.asteroid.x.append(holda[1][0])
            self.asteroid.y.append(holda[1][1])

    def AB4(self):
        self.RK3()
        
        for i in range(0, 3):
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0,0]))
            atot = fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0,0]), hata)
            for planet in self.planets:
                hat = -fo.hat(planet.poslist[i], np.array([0, 0]))
                planet.a.append(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0,0]), hat))
                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])
                atot += fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata)
            self.asteroid.a.append(atot)
            

        for i in range(3, self.N):
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0,0]))
            atot = fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0,0]), hata)
            for planet in self.planets:
                
                hat = -fo.hat(planet.poslist[i], (0, 0))
                planet.a.append(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat))
                
                hold = [np.add(planet.poslist[i], self.dt * (55 * planet.v[i] - 59 * planet.v[i-1] + 37 * planet.v[i-2] - 9 * planet.v[i-3]) / 24),
                         np.add(planet.v[i], self.dt * (55*planet.a[i] - 59*planet.a[i-1] + 37*planet.a[i-2] - 9*planet.a[i-3]) / 24)]
                planet.v.append(hold[1]) 
                planet.poslist.append(hold[0])
                planet.x.append(hold[0][0])
                planet.y.append(hold[0][1])


                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])
                atot += fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata)
                
            self.asteroid.a.append(atot)
            
            holda = [np.add(self.asteroid.poslist[i], self.dt * (55 * self.asteroid.v[i] - 59 * self.asteroid.v[i-1] + 37 * self.asteroid.v[i-2] - 9 * self.asteroid.v[i-3]) / 24),
                     np.add(self.asteroid.v[i], self.dt * (55*self.asteroid.a[i] - 59*self.asteroid.a[i-1] + 37*self.asteroid.a[i-2] - 9*self.asteroid.a[i-3]) / 24)]
            self.asteroid.v.append(holda[1]) 
            
            self.asteroid.poslist.append(holda[0])
            self.asteroid.x.append(holda[0][0])
            self.asteroid.y.append(holda[0][1])
            
    def AB4adaptive(self, constant):
        self.RK3()
        
        for i in range(0, 3):
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0,0]))
            atot = fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0,0]), hata)
            for planet in self.planets:
                hat = -fo.hat(planet.poslist[i], np.array([0, 0]))
                planet.a.append(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0,0]), hat))
                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])
                atot += fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata)
            self.asteroid.a.append(atot)
            

        for i in range(3, self.N):
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0,0]))
            atot = fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0,0]), hata)
            for planet in self.planets:
                self.dt = constant / planet.v[i]
                
                hat = -fo.hat(planet.poslist[i], (0, 0))
                planet.a.append(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat))
                
                hold = [np.add(planet.poslist[i], self.dt * (55 * planet.v[i] - 59 * planet.v[i-1] + 37 * planet.v[i-2] - 9 * planet.v[i-3]) / 24),
                         np.add(planet.v[i], self.dt * (55*planet.a[i] - 59*planet.a[i-1] + 37*planet.a[i-2] - 9*planet.a[i-3]) / 24)]
                planet.v.append(hold[1]) 
                planet.poslist.append(hold[0])
                planet.x.append(hold[0][0])
                planet.y.append(hold[0][1])


                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])
                atot += fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata)
                
            self.asteroid.a.append(atot)
            
            holda = [np.add(self.asteroid.poslist[i], self.dt * (55 * self.asteroid.v[i] - 59 * self.asteroid.v[i-1] + 37 * self.asteroid.v[i-2] - 9 * self.asteroid.v[i-3]) / 24),
                     np.add(self.asteroid.v[i], self.dt * (55*self.asteroid.a[i] - 59*self.asteroid.a[i-1] + 37*self.asteroid.a[i-2] - 9*self.asteroid.a[i-3]) / 24)]
            self.asteroid.v.append(holda[1]) 
            
            self.asteroid.poslist.append(holda[0])
            self.asteroid.x.append(holda[0][0])
            self.asteroid.y.append(holda[0][1])


            
    
    def solarsys_anim_no_trail(self, frames_per_update):
        # subplots() function you can draw 
        # multiple plots in one figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        
        
        # set limit for x and y axis
        
        # style for plotting line
        plt.style.use("ggplot")
        
        # create 5 list to get store element 
        # after every iteration
        x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        def animate(i):
            i = i * frames_per_update
            
            x1.append(self.planets[0].x[i])
            x2.append(self.planets[1].x[i])
            x3.append(self.planets[2].x[i])
            x4.append(self.planets[3].x[i])
            x5.append(self.planets[4].x[i])
            x6.append(self.planets[5].x[i])
            x7.append(self.planets[6].x[i])
            x8.append(self.planets[7].x[i])
            x9.append(self.asteroid.x[i])
            
            y1.append(self.planets[0].y[i])
            y2.append(self.planets[1].y[i])
            y3.append(self.planets[2].y[i])
            y4.append(self.planets[3].y[i])
            y5.append(self.planets[4].y[i])
            y6.append(self.planets[5].y[i])
            y7.append(self.planets[6].y[i])
            y8.append(self.planets[7].y[i])
            y9.append(self.asteroid.y[i])


            
            if i>=10:
                x1.pop(0)
                x2.pop(0)
                x3.pop(0)
                x4.pop(0)
                x5.pop(0)
                x6.pop(0)
                x7.pop(0)
                x8.pop(0)
                x9.pop(0)
                y1.pop(0)
                y2.pop(0)
                y3.pop(0)
                y4.pop(0)
                y5.pop(0)
                y6.pop(0)
                y7.pop(0)
                y8.pop(0)
                y9.pop(0)

            axes.clear()

            

            axes.plot((0,0), 'y*')
            axes.plot(x1, y1, color="red")
            axes.plot(x2, y2, color="gray")
            axes.plot(x3, y3, color="blue")
            axes.plot(x4, y4, color="green")
            axes.plot(x5, y5, color="brown")
            axes.plot(x6, y6, color="blue")
            axes.plot(x7, y7, color="green")
            axes.plot(x8, y8, color="brown")
            axes.plot(x9, y9, color="black")
            axes.set_title('Day ' + str(i))
            axes.set_ylim(-4600 * 10 ** 9, 4600 * 10 ** 9)
            axes.set_xlim(-4600 * 10 ** 9, 4600 * 10 ** 9)
        
            axes.legend(['Sun', 'Mercury Orbit','Venus Orbit', 'Earths Orbit', 'Mars Orbit', 'Jupiter Orbit', 'Saturn Orbit', 'Uranus Orbit', 'Neptune Orbit', 'Asteroid'])
        
        
        
        # set ani variable to call the 
        # function recursively
        anim = FuncAnimation(fig, animate, interval=10, save_count=1000)
        #anim.save('test.gif', writer='imagemagick', fps=60)
        plt.show()


    def solarsys_anim_with_trail(self, frames_per_update):
        # subplots() function you can draw 
        # multiple plots in one figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        
        
        # set limit for x and y axis
        
        # style for plotting line
        plt.style.use("ggplot")
        
        # create 5 list to get store element 
        # after every iteration
        x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        
        def animate(i):
            i = i * frames_per_update
            
            x1.append(self.planets[0].x[i])
            x2.append(self.planets[1].x[i])
            x3.append(self.planets[2].x[i])
            x4.append(self.planets[3].x[i])
            x5.append(self.planets[4].x[i])
            x6.append(self.planets[5].x[i])
            x7.append(self.planets[6].x[i])
            x8.append(self.planets[7].x[i])
            x9.append(self.asteroid.x[i])
            
            y1.append(self.planets[0].y[i])
            y2.append(self.planets[1].y[i])
            y3.append(self.planets[2].y[i])
            y4.append(self.planets[3].y[i])
            y5.append(self.planets[4].y[i])
            y6.append(self.planets[5].y[i])
            y7.append(self.planets[6].y[i])
            y8.append(self.planets[7].y[i])
            y9.append(self.asteroid.y[i])
            

            axes.plot((0,0), 'y*')
            axes.plot(x1, y1, color="red", linewidth=0.5)
            axes.plot(x2, y2, color="gray", linewidth=0.5)
            axes.plot(x3, y3, color="blue", linewidth=0.5)
            axes.plot(x4, y4, color="green", linewidth=0.5)
            axes.plot(x5, y5, color="brown")
            axes.plot(x6, y6, color="blue")
            axes.plot(x7, y7, color="green")
            axes.plot(x8, y8, color="brown")
            axes.plot(x9, y9, color="black")
            axes.set_title('Day ' + str(i))
            axes.set_ylim(-4600 * 10 ** 9, 4600 * 10 ** 9)
            axes.set_xlim(-4600 * 10 ** 9, 4600 * 10 ** 9)
        
            axes.legend(['Sun', 'Mercury Orbit','Venus Orbit', 'Earths Orbit', 'Mars Orbit', 'Jupiter Orbit', 'Saturn Orbit', 'Uranus Orbit', 'Neptune Orbit', 'Asteroid'])
        
        
        
        # set ani variable to call the 
        # function recursively
        anim = FuncAnimation(fig, animate, interval=10, save_count=1000)
        #anim.save('test.gif', writer='imagemagick', fps=60)
        plt.show()

        
 

def main():
    # TODO: Forces on astroid does not seem to be working, test this, also test adding jupiter and make them pass close, it is much bigger!
    sun = Planet(1.98892 * 10**30, 0, 0, 0)
    mercury = Planet(0.03301 * 10 ** 24 , 69.818 * 10 ** 9, np.array([0, 38860]), 0)
    venus = Planet(4.8673 * 10 ** 24, 108.941 * 10 ** 9, np.array([0, 34780]), 0)
    earth = Planet(5.9722 * 10 ** 24, 152.1 * 10 ** 9, np.array([0, 29290]), 0)
    mars = Planet(6.4174 * 10 ** 23, 249.261 * 10 ** 9, np.array([0, 21970]), 0)
    jupiter = Planet(1898.1 * 10 ** 24, 816.363 * 10 ** 9, np.array([0, 12440]), 0)
    saturn = Planet(568.32 * 10 ** 24, 1506.527 * 10 ** 9, np.array([0, 9140]), 0)
    uranus = Planet(86.811 * 10 ** 24, 3001.390 * 10 ** 9, np.array([0, 6900]), 0)
    neptune = Planet(102.409 * 10 ** 24, 4558.857 * 10 ** 9, np.array([0, 5370]), 0)

    asteroid = Planet(10, 830 * 10 ** 9, np.array([0, -15000]), np.pi/20)


    planets = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

    sim = Simulator(sun, planets, asteroid, N=2000)
    sim.AB4()
    sim.solarsys_anim_with_trail(20)
    # TODO: ADD ADAPTIVE TIMESTEP TEST IT, COULD BE WEIRD WHEN SIMULATING IT BECAUSE IT WOULD SLOW IT DOWN BUT SHOULD BE GOOD IN STILL PLOT...
    
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
