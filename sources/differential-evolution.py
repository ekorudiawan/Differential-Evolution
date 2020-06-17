import numpy as np
import matplotlib.pyplot as plt 

solution = np.array([0.5, 0.1, -0.3, 0.9, 0.4, -0.6, 0.2])
dim_theta = solution.shape[0]

def objective_function(thetas):
    fitness = np.sum(np.square(solution - thetas))
    return fitness

# Cr = crossover rate
# F = mutation rate
# NP = n population
def differential_evolution(thetas_limit, Cr=0.5, F=0.5, NP=10, max_gen=100, cr_type=''):
    n_params = len(thetas_limit)
    # Generate random target vectors
    target_vectors = np.random.rand(NP, n_params)
    target_vectors = np.interp(target_vectors, (0,1), (-1,1))
    # Variable donor vectors
    donor_vector = np.zeros(n_params)
    # Variable trial vectors
    trial_vector = np.zeros(n_params)
    best_fitness = np.inf
    list_best_fitness = []
    for gen in range(max_gen):
        print("Generation :", gen)
        for pop in range(NP):
            # print("Target vectors :", target_vectors[pop])
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = np.random.choice(index_choice, 3)
            donor_vector = target_vectors[a] - F * (target_vectors[b]-target_vectors[c])
            # print("Donor vectors :", donor_vector)
            n = np.random.randint(n_params)
            L = np.random.randint(1, n_params)
            n_end = n+L

            cross_points = np.random.rand(n_params) < Cr
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            # print("Trial vector", trial_vector)
            target_fitness = objective_function(target_vectors[pop])
            trial_fitness = objective_function(trial_vector)
            # print("Target fitness :", target_fitness)
            # print("Trial fitness :", trial_fitness)
            if trial_fitness < target_fitness:
                target_vectors[pop] = trial_vector.copy()
                best_fitness = trial_fitness
            else:
                best_fitness = target_fitness
        print("Best fitness :", best_fitness)
        list_best_fitness.append(best_fitness)
    return list_best_fitness

def main():
    limits = [(-1,1)] * dim_theta
    print(limits)
    result = differential_evolution(limits)
    fig, ax = plt.subplots()
    ax.plot(result)
    plt.show()
    
if __name__ == "__main__":
    main()
