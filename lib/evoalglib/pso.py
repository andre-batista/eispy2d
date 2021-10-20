import numpy as np
from evoalglib.mechanism import Mechanism
from numpy.random import rand


class ParticleSwarmOptimization(Mechanism):
    def __init__(self, boundary_condition, acceleration=2., inertia=.4):
        super().__init__(boundary_condition)
        self.w = inertia
        if type(acceleration) is int or type(acceleration) is float:
            self.c1, self.c2 = acceleration, acceleration
        else:
            self.c1, self.c2 = acceleration[0], acceleration[0]
        self.v = None
        self.pbest, self.gbest = None, None
        self.fp, self.fg = None, None
    def reset_variables(self, population_size, representation):
        super().reset_variables(population_size, representation)
        self.v = rand(population_size, representation.nvar)
        self.pbest = np.zeros((population_size, representation.nvar),
                              dtype=representation.dtype)
        self.gbest = np.zeros(representation.nvar, dtype=representation.dtype)
        self.fp, self.fg = np.inf*np.ones(population_size), np.inf
    def run(self, population, population_fitness, objective_function,
            current_nevals):
        _ = super().run(population, population_fitness, objective_function,
                        current_nevals)
        x, fx, objfun = population, population_fitness, objective_function
        NPOP, NVAR = population.shape
        for i in range(NPOP):
            if fx[i] < self.fp[i]:
                self.pbest[i, :] = x[i, :]
                self.fp[i] = fx[i]
            if self.fp[i] < self.fg:
                self.gbest = self.pbest[i, :]
                self.fg = self.fp[i]
        U1 = rand(NPOP, NVAR)
        U2 = rand(NPOP, NVAR)
        self.v = (self.w*self.v
                  + self.c1*U1*(self.pbest-x)
                  + self.c2*U2*(np.tile(self.gbest.reshape((1, -1)),
                                         (NPOP, 1)) - x))
        x[:, :] = x + self.v
        self.bc.run(x)
        for i in range(NPOP):
            fx[i] = objfun.eval(x[i, :])
        imin = np.argmin(fx)
        if fx[imin] < self.fg:
            self.xopt = np.copy(x[imin, :])
            self.fopt = fx[imin]
        else:
            self.xopt = np.copy(self.gbest)
            self.fopt = self.fg
        return x, fx, current_nevals + NPOP

    def copy(self, new=None):
        if new is None:
            new = ParticleSwarmOptimization(self.bc, (self.c1, self.c2),
                                            self.w)
            new.v = np.copy(self.v)
            new.pbest, new.gbest = np.copy(self.pbest), np.copy(self.gbest)
            new.fp, new.fg = np.copy(self.fp), self.fg
            new.bc, new.xopt, new.fopt = self.bc, self.xopt, self.fopt  
            return new         
        else:
            super().copy(new)
            self.v = np.copy(new.v)
            self.pbest, self.fp = np.copy(new.pbest), np.copy(new.fp)
            self.gbest, self.fg = np.copy(new.gbest), new.fg

    def __str__(self):
        message = super().__str__()
        message += 'Particle Swarm Optimization\n'
        message += 'Inertia: %.2f\n' % self.w
        message += 'Acceleration: %.2f (c1), ' % self.c1
        message += '%.2f\n' % self.c2 
        message += str(self.bc)
        return message
