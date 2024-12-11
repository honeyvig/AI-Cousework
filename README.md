# AI-Cousework
Please see attached zip file containing the starter code for this part of the assignment and a PDF containing all the deliverables.

Question 1
Please upload your report for PART B Experimenting with genetic algorithms and evolved
creatures.

The document should be in PDF format.

The word count should be less than 2000 words.

Question 2
Please upload ONE zip file containing subfolders with Code files for ALL the exercises here.

Remember to include all the requested files!

Question 3
Please upload video demonstration of all exercises in an MP4 format.
======================
It seems you're working on an assignment related to genetic algorithms and evolved creatures, specifically focused on creating a simulation where creatures evolve to climb a mountain. You have a starter codebase and an environment provided for the task. The goal is to adapt the genetic algorithm to enable creatures to climb the mountain effectively.

Here’s a breakdown of how to approach this task and a sample Python code that can help you get started. Since the provided environment likely involves evolving agents (creatures) based on fitness criteria, the task seems to revolve around adapting the fitness function to measure how well a creature can climb the mountain.
1. Understanding the Mountain Climbing Task

The task involves evolving creatures to climb a mountain without flying. The creatures will need to have some form of movement or action based on their genetic code. The fitness function needs to reward creatures that are able to climb as high as possible up the mountain.
2. Environment Code

You’ll need to integrate the mountain environment with your genetic algorithm. This typically involves:

    Loading the environment: This might be a 2D or 3D space representing the mountain.
    Evaluating the creature's fitness: The fitness will likely involve evaluating how high up the mountain a creature can get.
    Selection, crossover, mutation: These steps ensure the genetic algorithm evolves creatures over generations.

The cw-envt.py file likely contains the logic for the mountain environment, and prepare_shapes.py generates the mountain. Let’s focus on the fitness function and how to modify it to suit the mountain climbing task.
3. Genetic Algorithm Fitness Function

The fitness function needs to be adapted for the mountain-climbing task. Typically, this will involve:

    Initial creature setup: Each creature should be initialized with some set of genetic traits.
    Movement simulation: You will need to simulate the movement of each creature in the environment.
    Fitness Evaluation: The fitness function will evaluate how high the creature has climbed, without flying.

Here’s an example of how you might structure the Python code for the genetic algorithm with an adapted fitness function for mountain climbing.
Example Python Code for Genetic Algorithm with Mountain Climbing Task

import numpy as np
import random

# Define the creature class
class Creature:
    def __init__(self, genes):
        self.genes = genes  # This could represent movement abilities (e.g., speed, angle of movement)
        self.position = [0, 0]  # Initial position at the base of the mountain
        self.fitness = 0

    def move(self):
        # Simulate the movement of the creature based on its genes
        # Example: genes could control movement in 2D (speed, angle)
        self.position[0] += self.genes[0]  # Moving in x-direction (horizontal movement)
        self.position[1] += self.genes[1]  # Moving in y-direction (vertical movement)

    def evaluate_fitness(self, mountain):
        # The fitness function evaluates how high the creature gets on the mountain
        # We assume the mountain is represented as a function f(x), where y = f(x)
        # The goal is to maximize the vertical position (y-coordinate) without going off-path
        x, y = self.position
        if x < 0 or x >= mountain.width:
            self.fitness = 0  # If the creature goes out of bounds
        else:
            y_mountain = mountain.get_height_at(x)
            self.fitness = max(0, y_mountain - abs(y - y_mountain))  # Reward proximity to the mountain height

class Mountain:
    def __init__(self, width, shape_function):
        self.width = width
        self.shape_function = shape_function

    def get_height_at(self, x):
        # Generate the height of the mountain at a particular x coordinate
        return self.shape_function(x)

def generate_initial_population(population_size, gene_count):
    return [Creature([random.uniform(-1, 1) for _ in range(gene_count)]) for _ in range(population_size)]

def evolve_population(population, mountain, mutation_rate=0.1):
    # Selection: Pick the top 50% based on fitness
    population.sort(key=lambda creature: creature.fitness, reverse=True)
    survivors = population[:len(population) // 2]

    # Crossover: Mate pairs of survivors
    children = []
    while len(children) < len(population) - len(survivors):
        parent1, parent2 = random.sample(survivors, 2)
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        children.append(Creature(child_genes))

    # Mutation: Randomly mutate genes of some creatures
    for creature in children:
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(creature.genes) - 1)
            creature.genes[mutation_point] += random.uniform(-0.5, 0.5)

    # Combine survivors and children to form new population
    new_population = survivors + children
    return new_population

def main():
    # Define the mountain (e.g., a simple sine function as the shape)
    def mountain_shape(x):
        return np.sin(x / 50) * 10 + 10  # Example mountain shape (sine wave)

    mountain = Mountain(width=100, shape_function=mountain_shape)

    # Initialize the population of creatures
    population_size = 50
    population = generate_initial_population(population_size, gene_count=2)

    generations = 100
    for generation in range(generations):
        for creature in population:
            creature.move()
            creature.evaluate_fitness(mountain)
        
        # Evolve the population
        population = evolve_population(population, mountain)

        # Output the best fitness in each generation
        best_creature = max(population, key=lambda c: c.fitness)
        print(f"Generation {generation}, Best Fitness: {best_creature.fitness}")

if __name__ == "__main__":
    main()

Explanation of Code:

    Creature Class:
        The Creature class represents a creature in the environment. Each creature has a set of genes (for movement) and a position. The fitness function evaluates how well the creature can climb the mountain.

    Mountain Class:
        The Mountain class represents the mountain, which is defined by a function (mountain_shape). For this example, it’s a simple sine wave, but you can replace it with a more complex function or a dataset of mountain elevations.

    Fitness Function:
        The fitness function calculates how close the creature is to the actual height of the mountain at a given x-coordinate. If the creature moves too far from the optimal path, it gets a lower fitness score.

    Genetic Algorithm Operations:
        Initialization: Creatures are initialized with random genes.
        Selection: The top 50% of creatures based on their fitness are selected for mating.
        Crossover: Genes from two parents are combined to create offspring.
        Mutation: Some offspring genes are mutated randomly.

    Main Loop:
        The creatures evolve over multiple generations. The best fitness score is printed for each generation.

4. What You Need to Do:

    Integrate with Provided Code: Incorporate this logic into the provided code for the environment (cw-envt.py) and the mountain-shape generation (prepare_shapes.py).
    Run Experiments: Tune parameters like mutation rate, population size, and the fitness evaluation to evolve creatures that climb the mountain efficiently.
    Document Results: Record observations, including performance in different generations, challenges, and possible improvements to the evolutionary process.

Deliverables:

    Report (PDF): Describe the experiment, your approach, challenges, and outcomes.
    Code (ZIP file): All the code files and subfolders.
    Video Demonstration (MP4): A screen recording showing the experiments, including the running of the simulation and the genetic algorithm evolution.
