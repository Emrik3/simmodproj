# Simulating asteroid passing cloasly to the solar system
### Introduction
In this simulation of the solar system the Runge-Kutta method and Verlet method will be used to simulate the behavois of planets when a massive asteroid passes though the solar system. The Runge-Kutta method is not finished yet because of small accuracy error. The Verlet method is the main method for simulating the problem because it is a better method for conserving energy in long simulations. The behavior and energy is analysied further using functions listed below and then the chaos of the system is analysed using difference evaluation.

## Documentation

    class Planet():
        def __init__(self, mass, radius0, v, theta0):
            self.mass = mass
            self.radius0 = radius0
            self.theta0 = theta0
            self.x = radius0 * np.cos(theta0)
            self.y = radius0 * np.sin(theta0)
            
            self.poslist = [np.array([self.x, self.y])]
            self.v = [v]
            self.a = []

            self.x = [self.x]
            self.y = [self.y]



    class Simulator():
        def __init__(self, sun, planets, asteroid, dt = 3600*24, T = 3600*24*100)
            self.sun = sun
            self.planets = planets
            self.dt = dt
            self.T = T
            self.N = int(self.T / self.dt)
            self.asteroid = asteroid
            self.timelist = [0]

        
        def RK4(self):
            # Runge Kutta 4 Method

        def AB4(self):
            # Adam-Bachforth 4 method

        def verlet(self):
            # Velocity Verlet method

        def solarsys_anim_no_trail(self, frames_per_update, axmin, axmax):
            # Animation

        def solarsys_anim_with_trail(self, frames_per_update, axmin, axmax):
            # Animation with trail always there
            
    def initialize_planets():
        # Initialatising plant values

    def accuracy_test_RK4():
        # RK4 accuracy

    def run_verlet():
        # Running Verlet method


    def run_RK4():
        # Running 


    def accuracy_test_verlet():
        # Testing verlet accuracy

    def posvstimestep(k):
        # Plotting position vs timestep


    def massvary(k):
        # Plotting mass vs distance from sun


    def posvstime(k):
        # Plotting position vs time

    def rvstime(k):
        # Plotting distance from sun vs time

    def diffeval(k):
        # Distance evaluation

    def sysen(k):
        # Energy of the system