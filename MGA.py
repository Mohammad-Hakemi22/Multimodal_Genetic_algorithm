from random import randint, random
import numpy as np
from operator import xor
import math
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


class MGA():
    def __init__(self, pop_shape, pc=0.9, pm=0.005, max_round=100, chrom_l=[0, 0], low=[0, 0], high=[0, 0]):
        self.pop_shape = pop_shape
        self.pc = pc
        self.pm = pm
        self.max_round = max_round
        self.chrom_l = chrom_l
        self.low = low
        self.high = high

    def initialization(self):  # initialize first population
        pop = np.random.randint(
            low=0, high=2, size=self.pop_shape)  # random number 0,1
        return pop

    def crossover(self, ind_0, ind_1):  # cross over for two individual (one point crossover)
        new_0, new_1 = [], []
        # check two individuals have same lenght
        assert(len(ind_0) == len(ind_1))
        p_pc = np.random.random_sample(1)
        if p_pc < self.pc:  # doing crossover
            point = np.random.randint(len(ind_0))
            new_0 = list(np.hstack((ind_0[:point], ind_1[point:])))
            new_1 = list(np.hstack((ind_1[:point], ind_0[point:])))
        else:  # Transfer without crossover
            new_0 = list(ind_0)
            new_1 = list(ind_1)
        # check two new childs have same lenght
        assert(len(new_0) == len(new_1))

        return new_0, new_1

    def mutation(self, pop):
        # Calculate the number of bits that must mutation
        num_mut = math.ceil(self.pm * pop.shape[0] * pop.shape[1])
        for m in range(0, num_mut):
            i = np.random.randint(0, pop.shape[0])
            j = np.random.randint(0, pop.shape[1])
            pop[i][j] = xor(pop[i][j], 1)
        return pop

    def fitnessFunc(self, real_val):
        fitness_val = 21.5 + \
            real_val[0]*np.sin(4*np.pi*real_val[0]) + \
            real_val[1]*np.sin(20*np.pi*real_val[1])
        return fitness_val

    def b2d(self, list_b):  # convert binary number to decimal number
        l = len(list_b)
        sum = 0
        for i in range(0, l):
            p = ((l-1)-i)
            sum += (pow(2, p) * list_b[i])
        return sum

    def d2r(self, b2d, lenght_b, m):  # Change the decimal number to fit in the range of problem variables
        norm = b2d/(pow(2, lenght_b) - 1)
        match m:
            case 0:
                real = self.low[0] + (norm * (self.high[0] - self.low[0]))
                return real
            case 1:
                real = self.low[1] + (norm * (self.high[1] - self.low[1]))
                return real

    # decoding the chromosome value for calculate fitness
    def chromosomeDecode(self, pop):
        gen = []
        for i in range(0, pop.shape[0]):
            l1 = pop[i][0:self.chrom_l[0]]
            l2 = pop[i][self.chrom_l[0]:]
            gen.append(self.d2r(self.b2d(list(l1)), len(l1), 0))
            gen.append(self.d2r(self.b2d(list(l2)), len(l2), 1))
        return np.array(gen).reshape(pop.shape[0], 2)

    def roulette_wheel_selection(self, population):
        chooses_ind = [i for i in range (0,100)]
        return chooses_ind  # return selected individuals

    def selectInd(self, chooses_ind, pop):  # Perform crossover on the selected population
        new_pop = []
        for i in range(0, len(chooses_ind), 2):
            a, b = self.crossover(pop[chooses_ind[i]], pop[chooses_ind[i+1]])
            res = self.preSelection(pop[chooses_ind[i]],pop[chooses_ind[i+1]],a,b)
            new_pop.append(res[0])
            new_pop.append(res[1])
        npa = np.asarray(new_pop, dtype=np.int32)
        return npa

    def preSelection(self, par1, par2, off1, off2):
        par_off = []
        res = []
        par_off.append(par1)
        par_off.append(par2)
        par_off.append(off1)
        par_off.append(off2)
        par_off = np.asarray(par_off, dtype=np.int32)
        par_decoded = self.chromosomeDecode(par_off)
        fit_par_off = [self.fitnessFunc(i) for i in par_decoded]
        res.append(par_off[np.argmax(fit_par_off)])
        np.delete(par_off,np.argmax(fit_par_off))
        res.append(par_off[np.argmax(fit_par_off)])
        return res
    


    def bestResult(self, population):  # calculate best fitness, avg fitness
        population_fitness = [self.fitnessFunc(
            population[i]) for i in range(0, population.shape[0])]
        population_best_fitness = np.sort(population_fitness, kind="heapsort")[-5:]
        agents_index = np.argsort(population_fitness, kind="heapsort")[-5:]
        agents = population[agents_index]
        avg_population_fitness = sum(
            population_fitness) / len(population_fitness)
        return population_best_fitness, avg_population_fitness, population_fitness, agents

    def run(self):  # start algorithm
        avg_population_fitness = []
        population_best_fitness = []
        population_fitness = []
        agents = []
        ga = MGA((100, 33), chrom_l=[18, 15],
                 low=[-3, 4.1], high=[12.1, 5.8])
        n_pop = ga.initialization()  # initial first population
        for i in range(0, self.max_round):
            chrom_decoded = ga.chromosomeDecode(n_pop)
            b_f, p_f, p, a = ga.bestResult(chrom_decoded)
            avg_population_fitness.append(p_f)
            population_best_fitness.append(b_f)
            population_fitness.append(p)
            agents.append(a)
            selected_ind = ga.roulette_wheel_selection(chrom_decoded)
            new_child = ga.selectInd(selected_ind, n_pop)
            new_pop = ga.mutation(new_child)
            n_pop = new_pop  # Replace the new population
        return population_best_fitness, avg_population_fitness, agents, chrom_decoded

    def plot(self, population_best_fitness, avg_population_fitness, agents, chrom_decoded):
        fig, ax = plt.subplots()
        ax.plot(avg_population_fitness, linewidth=2.0, label="avg_fitness")
        ax.plot(population_best_fitness, linewidth=2.0 ,label=["best_fitness 1","best_fitness 2","best_fitness 3","best_fitness 4","best_fitness 5"], linestyle=':')
        plt.legend(loc="lower right")
        print(f"best solution: {population_best_fitness[-1]}")
        print(
            f"best solution agents: {agents[-1]}")
        plt.show()
