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
    def __init__(self, sun, planets, asteroid, dt = 3600*24, T = 3600*24*100):
        
        self.sun = sun
        self.planets = planets
        self.dt = dt
        self.T = T
        self.N = int(self.T / self.dt)
        self.asteroid = asteroid
        self.timelist = [0]

    def RK4(self):
        # TODO: Försök fixa detta och om inget funkar gör via numpy arrays att allt görs via vektorer samtidigt.
        for i in range(0,self.N):
            self.timelist.append(i * self.dt / (3600 * 24))
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0, 0]))
            k1 = self.dt * fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0, 0]), hata)
            k1tot = k1
            k1v = self.dt * np.add(self.asteroid.v[i], np.array([0,0]))
            k1vtot = k1v

            k2 = self.dt * fo.acc(self.sun.mass, G, np.add(self.asteroid.poslist[i], k1v/2), np.array([0, 0]), hata)
            k2tot = k2
            k2v = self.dt * np.add(self.asteroid.v[i], k1/2)
            k2vtot = k2v

            k3 = self.dt * fo.acc(self.sun.mass, G, np.add(self.asteroid.poslist[i], k2v / 2), np.array([0,0]), hata)
            k3tot = k3
            k3v = self.dt * np.add(self.asteroid.v[i], k2 / 2)
            k3vtot = k3v

            k4 = self.dt * fo.acc(self.sun.mass, G, np.add(self.asteroid.poslist[i], k3v), np.array([0,0]), hata)
            k4tot = k4
            k4v = self.dt * np.add(self.asteroid.v[i], k3)
            k4vtot = k4v
            

            for planet in self.planets:
                hat2 = -fo.hat(planet.poslist[i], self.asteroid.poslist[i])
                hat = -fo.hat(planet.poslist[i], np.array([0, 0]))
                k1 = self.dt * np.add(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, planet.poslist[i], self.asteroid.poslist[i], hat2))
                k1v = self.dt * np.add(planet.v[i], [0, 0])

                k2 = self.dt * np.add(fo.acc(self.sun.mass, G, np.add(planet.poslist[i], k1v/2), np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, np.add(planet.poslist[i], k1v/2), self.asteroid.poslist[i], hat2))
                k2v = self.dt * np.add(planet.v[i], k1/2)

                k3 = self.dt * np.add(fo.acc(self.sun.mass, G, np.add(planet.poslist[i], k2v/2), np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, np.add(planet.poslist[i], k2v/2), self.asteroid.poslist[i], hat2))
                k3v = self.dt * np.add(planet.v[i], k2 / 2)
                
                k4 = self.dt * np.add(fo.acc(self.sun.mass, G, np.add(planet.poslist[i], k3v), np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, np.add(planet.poslist[i], k3v), self.asteroid.poslist[i], hat2))
                k4v = self.dt * np.add(planet.v[i], k3)

                hold = [np.add(planet.v[i], (k1 + 2*k2 + 2*k3 + k4) / 6), np.add(planet.poslist[i], (k1v + 2*k2v + 2*k3v + k4v) / 6)]
                planet.v.append(hold[0])
            
                planet.poslist.append(hold[1])
                planet.x.append(hold[1][0])
                planet.y.append(hold[1][1])



                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])

                k1 =  self.dt * fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata)
                k1tot += k1
                k1v = self.dt * self.asteroid.v[i]
                k1vtot += k1v


                k2 = self.dt * fo.acc(planet.mass, G, np.add(self.asteroid.poslist[i], k1v/2), planet.poslist[i], hata)
                k2tot += k2
                k2v = self.dt * np.add(self.asteroid.v[i], k1/2)
                k2vtot += k2v
                

                k3 = self.dt * fo.acc(planet.mass, G, np.add(self.asteroid.poslist[i], k1v / 2), planet.poslist[i], hata)
                k3tot += k3
                k3v = self.dt * np.add(self.asteroid.v[i], k1 / 2)
                k3vtot += k3v

                k4 = self.dt * fo.acc(planet.mass, G, np.add(self.asteroid.poslist[i], k3v), planet.poslist[i], hata)
                k4tot += k4
                k4v = self.dt * np.add(self.asteroid.v[i], k3)
                k4vtot += k4v
                

            

            holda = [np.add(self.asteroid.v[i], (k1tot + 2*k2tot + 2*k3tot + k4tot) / 6), np.add(self.asteroid.poslist[i], (k1vtot + 2*k2vtot + 2*k3vtot + k4vtot) / 6)]
            
            
            self.asteroid.v.append(holda[0])
            
            self.asteroid.poslist.append(holda[1])
            self.asteroid.x.append(holda[1][0])
            self.asteroid.y.append(holda[1][1])

    def RK3helper(self):
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
        self.RK3helper()
        
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
                hat2 = -fo.hat(planet.poslist[i], self.asteroid.poslist[i])
                planet.a.append(np.add(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, planet.poslist[i], self.asteroid.poslist[i], hat2)))
                
                hold = [np.add(planet.poslist[i], self.dt * (55 * planet.v[i] - 59 * planet.v[i-1] + 37 * planet.v[i-2] - 9 * planet.v[i-3]) / 24),
                         np.add(planet.v[i], self.dt * (55*planet.a[i] - 59*planet.a[i-1] + 37*planet.a[i-2] - 9*planet.a[i-3]) / 24)]
                planet.v.append(hold[1]) 
                planet.poslist.append(hold[0])
                planet.x.append(hold[0][0])
                planet.y.append(hold[0][1])

                hat = -fo.hat(planet.poslist[i], (0, 0))
                

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
        self.RK3helper()
        
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


    def verlet(self):
        for i in range(0,self.N):
            self.timelist.append(i * self.dt / (3600 * 24))
            hata = -fo.hat(self.asteroid.poslist[i], np.array([0, 0]))
            self.asteroid.poslist.append(np.add(self.asteroid.poslist[i], self.dt * np.add(self.asteroid.v[i], self.dt/2 * fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0, 0]), hata))))
            
            hatanew = -fo.hat(self.asteroid.poslist[i+1], np.array([0, 0]))
            self.asteroid.v.append(np.add(self.asteroid.v[i], self.dt * 1/2*np.add(fo.acc(self.sun.mass, G, self.asteroid.poslist[i+1], np.array([0, 0]), hatanew), fo.acc(self.sun.mass, G, self.asteroid.poslist[i], np.array([0, 0]), hata))))
            
            for planet in self.planets:
                # TODO: adding the wrong acceleration because a(i+1) is calculated after
                hat2 = -fo.hat(planet.poslist[i], self.asteroid.poslist[i])
                hat = -fo.hat(planet.poslist[i], np.array([0, 0]))
                planet.poslist.append(np.add(planet.poslist[i], self.dt * np.add(planet.v[i], self.dt/2 * np.add(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, planet.poslist[i], self.asteroid.poslist[i], hat2)))))
                

                hata = -fo.hat(self.asteroid.poslist[i], planet.poslist[i])
                self.asteroid.poslist[i+1] += self.dt * np.add(self.asteroid.v[i], self.dt/2 * fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata))
                
                hatanew = -fo.hat(self.asteroid.poslist[i+1], planet.poslist[i+1])
                self.asteroid.v[i+1] += self.dt * 1/2*np.add(fo.acc(planet.mass, G, self.asteroid.poslist[i+1], planet.poslist[i+1], hatanew), fo.acc(planet.mass, G, self.asteroid.poslist[i], planet.poslist[i], hata))

                hat2new = -fo.hat(planet.poslist[i+1], self.asteroid.poslist[i+1])
                hatnew = -fo.hat(planet.poslist[i+1], np.array([0, 0]))
                planet.v.append(np.add(planet.v[i], self.dt * 1/2*np.add(fo.acc(self.sun.mass, G, planet.poslist[i+1], np.array([0, 0]), hatnew), fo.acc(self.asteroid.mass, G, planet.poslist[i+1], self.asteroid.poslist[i+1], hat2new)), np.add(fo.acc(self.sun.mass, G, planet.poslist[i], np.array([0, 0]), hat), fo.acc(self.asteroid.mass, G, planet.poslist[i], self.asteroid.poslist[i], hat2))))
                hold = [planet.v[i+1], planet.poslist[i+1]]
                planet.x.append(hold[1][0])
                planet.y.append(hold[1][1])


            self.asteroid.x.append(self.asteroid.poslist[i+1][0])
            self.asteroid.y.append(self.asteroid.poslist[i+1][1])      
            # TODO: Also remember to try and fix RK4 code!!!
            # TODO: Prio 1: Accyracy of velet code is 1, this is good, jsut small error in code possobly or this is value to be reported. should be 2.
            # TODO: DO this by wring on paper the solution and then checking the code step by step,
            
    
    def solarsys_anim_no_trail(self, frames_per_update, axmin, axmax):


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

            if i>=10 * frames_per_update:
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
            axes.plot(x1, y1, color="red", linewidth=0.5)
            axes.plot(x2, y2, color="gray", linewidth=0.5)
            axes.plot(x3, y3, color="blue", linewidth=0.5)
            axes.plot(x4, y4, color="green", linewidth=0.5)
            axes.plot(x5, y5, color="brown")
            axes.plot(x6, y6, color="blue")
            axes.plot(x7, y7, color="green")
            axes.plot(x8, y8, color="brown")
            axes.plot(x9, y9, color="black")
            axes.set_title('Day ' + str(int(self.timelist[i])))
            axes.set_ylim(axmin[1], axmax[1])
            axes.set_xlim(axmin[0], axmax[0])
        
            axes.legend(['Sun', 'Mercury Orbit','Venus Orbit', 'Earths Orbit', 'Mars Orbit', 'Jupiter Orbit', 'Saturn Orbit', 'Uranus Orbit', 'Neptune Orbit', 'Asteroid'])
        
        
        
        # set ani variable to call the 
        # function recursively
        anim = FuncAnimation(fig, animate, interval=10, save_count=1000)
        #anim.save('test.gif', writer='imagemagick', fps=60)
        plt.show()
        

    def solarsys_anim_with_trail(self, frames_per_update, axmin, axmax):
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
            axes.set_title('Day ' + str(int(self.timelist[i])))
            axes.set_ylim(axmin[1], axmax[1])
            axes.set_xlim(axmin[0], axmax[0])
        
            axes.legend(['Sun', 'Mercury Orbit','Venus Orbit', 'Earths Orbit', 'Mars Orbit', 'Jupiter Orbit', 'Saturn Orbit', 'Uranus Orbit', 'Neptune Orbit', 'Asteroid'])
        
        
        
        # set ani variable to call the 
        # function recursively
        anim = FuncAnimation(fig, animate, interval=10, save_count=1000)
        #anim.save('test.gif', writer='imagemagick', fps=60)
        plt.show()

        
def initialize_planets():
    mercury = Planet(0.03301 * 10 ** 24 , 69.818 * 10 ** 9, np.array([0, 38860]), 0)
    venus = Planet(4.8673 * 10 ** 24, 108.941 * 10 ** 9, np.array([0, 34780]), 0)
    earth = Planet(5.9722 * 10 ** 24, 152.1 * 10 ** 9, np.array([0, 29290]), 0)
    mars = Planet(6.4174 * 10 ** 23, 249.261 * 10 ** 9, np.array([0, 21970]), 0)
    jupiter = Planet(1898.1 * 10 ** 24, 816.363 * 10 ** 9, np.array([0, 12440]), 0)
    saturn = Planet(568.32 * 10 ** 24, 1506.527 * 10 ** 9, np.array([0, 9140]), 0)
    uranus = Planet(86.811 * 10 ** 24, 3001.390 * 10 ** 9, np.array([0, 6900]), 0)
    neptune = Planet(102.409 * 10 ** 24, 4558.857 * 10 ** 9, np.array([0, 5370]), 0)
    moon = Planet(0.07346 * 10**24, 0.4055 * 10**6 + earth.radius0, np.array([0, 0.97 * 10**3 + 29290]), 0)
    planets = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]
    sun = Planet(1.98892 * 10**30, 0, 0, 0)    
    # TODO: Add moon in RK calculations!!!
    return planets, sun


def accuracy_test_RK4():
    final_posx = []
    final_posy = []
    finalxdiff = []
    finalydiff = []
    finaldiff = []
    ilist = []
    dtlist = [360, 360*2, 360*4, 360*8, 380*16, 360*32] # Fixa även så att den möter på samma ställe varje gång!!!
    for i in dtlist:
        
        print(i)
        planets, sun = initialize_planets()
        asteroid = Planet(10**29, 250 * 10 ** 9, np.array([5000, 0]), np.pi/2)
        sim = Simulator(sun, planets, asteroid, dt=i, T = 3600*24*300) 
        sim.RK4()
        final_posx.append(planets[3].poslist[-1][0])
        final_posy.append(planets[3].poslist[-1][1])
        if i != 360 and i != 360*32:
            ilist.append(i)
            finalxdiff.append(abs(final_posx[-2] - final_posx[-1]))
            finalydiff.append(abs(final_posy[-2] - final_posy[-1]))
            finaldiff.append(abs(abs(final_posx[-2] + final_posy[-2])**.5) - abs(final_posx[-1] + final_posy[-1])**.5)

    plt.plot(dtlist, final_posx)
    plt.show()
    plt.plot(dtlist, final_posy)
    plt.show() # TODO: not 4th order accurate!!!!!!!

    ilistsq = []
    for i in ilist:
        ilistsq.append(i**4)
    plt.loglog(ilist, finaldiff)
    plt.loglog(ilist, ilistsq)
    plt.show()
    print("Accuracy x: " + str(np.log2(finalxdiff[2]/finalxdiff[1])))
    print("Accuracy y: " + str(np.log2(finalydiff[2]/finalydiff[1])))
    print("Accuracy: " + str(np.log2(finaldiff[2]/finaldiff[1])))


def run_verlet():
    planets, sun = initialize_planets()
    asteroid = Planet(10**29, 250 * 10 ** 9, np.array([5000, 0]), np.pi/2)
    # TODO: Velocity for asteroid too large

    axmin = [-250 * 10 ** 9, -250 * 10 ** 9]
    axmax = [250 * 10 ** 9, 250 * 10 ** 9]

    # TODO: Add setting that follows the earths path instead of entire window

    sim = Simulator(sun, planets, asteroid, dt=2*360, T = 3600*24*366)
    sim.verlet()
    sim.solarsys_anim_no_trail(100, axmin, axmax)


def accuracy_test_verlet():
    final_posx = []
    final_posy = []
    finalxdiff = []
    finalydiff = []
    finaldiff = []
    ilist = []
    dtlist = [90, 180, 360, 360*2, 360*4, 360*8, 360*16] # Fixa även så att den möter på samma ställe varje gång!!!
    for i in dtlist:
        
        print(i)
        planets, sun = initialize_planets()
        asteroid = Planet(10**29, 250 * 10 ** 9, np.array([5000, 0]), np.pi/2)
        sim = Simulator(sun, planets, asteroid, dt=i, T = 360*64*100) 
        sim.verlet()
        final_posx.append(planets[3].poslist[-1][0])
        final_posy.append(planets[3].poslist[-1][1])
        if i != 90 and i != 360*16:
            ilist.append(i)
            finalxdiff.append(abs(final_posx[-2] - final_posx[-1]))
            finalydiff.append(abs(final_posy[-2] - final_posy[-1]))
            finaldiff.append(abs(abs(final_posx[-2] + final_posy[-2])**.5) - abs(final_posx[-1] + final_posy[-1])**.5)


    print("Accuracy x: " + str(np.log2(finalxdiff[2]/finalxdiff[1])))
    print("Accuracy y: " + str(np.log2(finalydiff[2]/finalydiff[1])))
    print("Accuracy: " + str(np.log2(finaldiff[2]/finaldiff[1])))

    plt.plot(dtlist, final_posx)
    plt.show()
    plt.plot(dtlist, final_posy)
    plt.show() # TODO: not 2nd order accurate!!!!!!!

    ilistsq = []
    for i in ilist:
        ilistsq.append(i**4)
    plt.loglog(ilist, finaldiff)
    plt.loglog(ilist, ilistsq)
    plt.show()



def main():
    # TODO: Make a function that looks at different masses for asteroid and how much planets are effected
    # TODO: Maybe also do this for different starting positions and different initial speeds!!!
    #accuracy_test_RK4()
    accuracy_test_verlet() # TODO: THis should be 2nd order

# TODO: Clean up code!


if __name__ == "__main__":
    main()
