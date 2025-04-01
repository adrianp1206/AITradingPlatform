import random
import numpy as np
from train_boost import train_xgboost

# GA Parameters
population_size = 10
generations = 50
mutation_rate = 0.1

# Initialize Population
def initialize_population(all_features):
    return [
        random.sample(all_features, k=random.randint(5, len(all_features)))
        for _ in range(population_size)
    ]

def fitness_function(feature_subset, ticker, start_date, end_date, n_splits=5, params=None):
    """
    Fitness function for genetic algorithm to evaluate a feature subset.

    Args:
    feature_subset: list, Selected features to evaluate.
    ticker: str, The stock ticker symbol.
    start_date: str, Start date for training data.
    end_date: str, End date for training data.
    n_splits: int, Number of splits for TimeSeriesSplit.
    params: dict, Hyperparameters for XGBoost.

    Returns:
    accuracy: float, Average accuracy score for the feature subset.
    """
    metrics, _ = train_xgboost(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_subset=feature_subset,
        n_splits=n_splits,
        params=params,
        save_model=False  # Do not save during GA
    )
    return metrics['accuracy']

# Selection
def selection(population, fitness_scores):
    paired_population = list(zip(population, fitness_scores))
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    return [individual for individual, _ in sorted_population[:population_size // 2]]

# Crossover
def crossover(parent1, parent2):
    return list(set(parent1[:len(parent1)//2] + parent2[len(parent2)//2:]))

# Mutation
def mutate(individual, all_features):
    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(individual) > 1:
            individual.remove(random.choice(individual))
        else:
            feature_to_add = random.choice([f for f in all_features if f not in individual])
            individual.append(feature_to_add)
    return individual

# Genetic Algorithm
def genetic_algorithm(ticker, start_date, end_date, all_features, n_splits=5, params=None):
    """
    Run the genetic algorithm to identify the best feature subset for a given stock ticker.

    Args:
    ticker: str, The stock ticker symbol.
    start_date: str, Start date for training data.
    end_date: str, End date for training data.
    all_features: list, All potential features to use in the GA.
    n_splits: int, Number of splits for TimeSeriesSplit.
    params: dict, Hyperparameters for XGBoost.

    Returns:
    best_features: list, The best feature subset identified by the GA.
    """
    population = initialize_population(all_features)
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        fitness_scores = [
            fitness_function(individual, ticker, start_date, end_date, n_splits, params)
            for individual in population
        ]
        
        selected_individuals = selection(population, fitness_scores)
        next_generation = []

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = mutate(crossover(parent1, parent2), all_features)
            if child not in next_generation:
                next_generation.append(child)

        population = next_generation

    # Final evaluation
    fitness_scores = [
        fitness_function(individual, ticker, start_date, end_date, n_splits, params)
        for individual in population
    ]
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual
