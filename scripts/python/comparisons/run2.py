import operator
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

pset = gep.PrimitiveSet('Main', input_names=['x'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
# each ENC is a random integer within [-10, 10]
pset.add_ephemeral_terminal(
    name='enc', gen=lambda: random.uniform(-10.0, 10.0))


# to minimize the objective (fitness)
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

h = 5  # head length
n_genes = 1   # number of genes in a chromosome

toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual,
                 gene_gen=toolbox.gene_gen, n_genes=n_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

X = np.loadtxt("data/datax_2.txt")
Y = np.loadtxt("data/datay_2_1.txt")



toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform,
                 pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.4)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 1p: expected one point mutation in an individual
toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')
# we can also give the probability via the pbs property
toolbox.pbs['mut_ephemeral'] = 1

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# size of population and number of generations
n_pop = 40
n_gen = 100

pop = toolbox.population(n=n_pop)
# only record the best three individuals ever found in all generations
hof = tools.HallOfFame(40)

error_list = []

for pp in range(100):
    X = np.loadtxt("data/datax_2.txt")
    Y = np.loadtxt("data/datay_2_"+str(pp+1)+".txt")
    
    def evaluate2(individual):
        """Evalute the fitness of an individual: MAE (mean absolute error)"""
        func = toolbox.compile(individual)
        Yp = np.array(list(map(func, X)))
        return np.mean(np.abs(Y - Yp)/(np.abs(Y)+1)),

    def evaluate(individual):
        """Evalute the fitness of an individual: MAE (mean absolute error)"""
        func = toolbox.compile(individual)
        Yp = np.array(list(map(func, X)))
        return np.mean(np.abs(Y - Yp)**2),
    
    toolbox.register('evaluate', evaluate)

    # start evolution
    pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                            stats=stats, hall_of_fame=hof, verbose=True)

    print(hof[0])

    temp = []
    for j in pop:
        temp.append(evaluate2(j)[0])
    error_list.append(np.min(temp))

np.savetxt("result/result_2_comp.txt",error_list)