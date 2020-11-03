import random

random.seed(0)

NUM_OF_CELLS = 32           # number of cells initialized in first generation
ELITISM = 0.02              # percent of new individuals made by elitism
FRESH = 0.1                 # percent of new individuals made for new generation
CROSSOVER = 0.6             # percent of new individuals made by crossover
TOURNAMENT = 0.5            # percent of individuals in tournament


NUM_OF_GENERATIONS = 100
NUM_OF_INDIVIDUALS = 400

map_lines = 0
map_columns = 0
all_tresures = {}
start_line = 0
start_column = 0


def init(file_path):
    global map_lines
    global map_columns
    global all_tresures
    global start_line
    global start_column
    file = open(file_path, "r")
    data = {}
    for line in file:
        arr = line.split(":")
        if arr[0].startswith("treasure"):
            data[arr[0]] = arr[1].split(",")
        else:
            data[arr[0]] = int(arr[1])

    map_lines = data["map_lines"]
    map_columns = data["map_columns"]
    start_line = data["start_line"]
    start_column = data["start_column"]

    for i in range(data["num_of_treasures"]):
        treasure = f"treasure{i}"
        all_tresures[i] = [int(data[treasure][0]), int(data[treasure][1][:1])]

    file.close()


def first_generation():
    global NUM_OF_INDIVIDUALS
    global NUM_OF_CELLS

    individuals = {}
    for i in range(NUM_OF_INDIVIDUALS):
        individuals[i] = {}
        individuals[i]["fitness"] = 0
        individuals[i]["path"] = []
        individuals[i]["memory_cells"] = [0 for _ in range(64)]
        individuals[i]["treasures"] = list()
        for j in range(NUM_OF_CELLS):
            individuals[i]["memory_cells"][j] = random.randint(0, 255)
    return individuals


def write(move):
    if move == "00":
        return "H"
    elif move == "01":
        return "D"
    elif move == "10":
        return "P"
    elif move == "11":
        return "L"
    return None


def virtual_machine(individual):
    program_counter = 0
    index = 0
    instructions = [individual["memory_cells"][i] for i in range(64)]
    while program_counter <= 500 and 0 <= index < 64:
        operation = format(instructions[index], '#010b')[2:4]
        address = int(format(instructions[index], '#010b')[4:], 2)

        if operation == "00":  # increment
            instructions[address] += 1
            if instructions[address] == 256:
                instructions[address] = 0

        elif operation == "01":  # decrement
            instructions[address] -= 1
            if instructions[address] == -1:
                instructions[address] = 255

        elif operation == "10":  # jump
            index = address

        elif operation == "11":  # write uses last 2 bits
            move = format(instructions[index], '#010b')[8:]
            individual["path"].append(write(move))

        program_counter += 1
        index += 1


def found_treasures(individual):
    global all_tresures
    global start_line
    global start_column
    global map_lines
    global map_columns

    curr_line = start_line
    curr_column = start_column
    counter = 0
    for move in individual["path"]:
        if move == "H":
            curr_line -= 1
        elif move == "D":
            curr_line += 1
        elif move == "P":
            curr_column += 1
        elif move == "L":
            curr_column -= 1

        if not 0 <= curr_line < map_lines or not 0 <= curr_column < map_columns:  # check if I'm in the map
            individual["path"] = individual["path"][:counter]
            return

        curr_pos = [curr_line, curr_column]
        if curr_pos in all_tresures.values() and curr_pos not in individual["treasures"]:  # check for treasure
            individual["treasures"].append(curr_pos)

        if len(individual["treasures"]) == len(all_tresures):
            # print("All treasures were found")
            return
        counter += 1


def set_fitness(individual):
    if len(individual["path"]) == 0:
        return

    individual["fitness"] = len(individual["treasures"]) + 1 - len(individual["path"]) / 1000


def fresh_individual():
    individual = {"fitness": 0, "path": [], "memory_cells": [0 for _ in range(64)], "treasures": list()}
    for i in range(NUM_OF_CELLS):
        individual["memory_cells"][i] = random.randint(0, 255)
    return individual


def crossover(mom, dad):
    child = []
    copy_length = random.randint(0, 63)
    start_index = random.randint(0, 63 - copy_length)

    for i in range(64):
        if start_index <= i < start_index + copy_length:
            child.append(mom[i])
        else:
            child.append(dad[i])

    individual = {"fitness": 0, "path": [], "memory_cells": child, "treasures": list()}
    return individual


def tournament(generation, new_generation):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    end_index = int((ELITISM + FRESH + CROSSOVER) * NUM_OF_INDIVIDUALS)
    num_tournament = int(TOURNAMENT * NUM_OF_INDIVIDUALS)

    for i in range(start_index, end_index):
        sorted_individuals = sorted(random.choices(generation, k=num_tournament), reverse=True, key=lambda x: x["fitness"])
        new_generation[i] = crossover(sorted_individuals[0]["memory_cells"], sorted_individuals[1]["memory_cells"])

    return new_generation


def mutate_little(individual):
    for i in range(2):
        index = random.randint(0, 63)
        individual["memory_cells"][index] = random.randint(0, 255)
    return individual


def mutate_random(individual):
    mut_count = random.randint(5, 15)
    for i in range(mut_count):
        index = random.randint(0, 63)
        individual["memory_cells"][index] = random.randint(0, 255)
    return individual


def mutate_switch(individual):
    index_1 = random.randint(0, 60)
    index_2 = random.randint(0, 60)

    while abs(index_1 - index_2) < 3:
        index_1 = random.randint(0, 60)
        index_2 = random.randint(0, 60)

    minimum = min(index_1, index_2)
    copy_1 = individual["memory_cells"][minimum: minimum + 3]
    maximum = max(index_1, index_2)
    copy_2 = individual["memory_cells"][maximum: maximum + 3]

    for i in range(minimum, maximum + 4):
        if i <= minimum + 3:
            individual["memory_cells"][i] = copy_2.pop()
        elif i >= maximum:
            individual["memory_cells"][i] = copy_1.pop()

    return individual


def mutation_selection(sorted_gen, new_generation):
    index = int((1 - (ELITISM + FRESH + CROSSOVER)) * NUM_OF_INDIVIDUALS)
    best_individuals = sorted_gen[:index]
    index = int((ELITISM + FRESH + CROSSOVER) * NUM_OF_INDIVIDUALS)
    # dorobit mutacia najlepsich je pridane ako posledna caast novej generacie
    for i in range(index, NUM_OF_INDIVIDUALS):
        individual = random.choice(best_individuals.items())
        new_generation[i] = mutate_switch(individual)


def create_generation(generation):
    new_generation = {}
    sorted_gen = [i[1] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
    # elitism
    elite_end = int(ELITISM * NUM_OF_INDIVIDUALS)
    for i in range(0, elite_end):
        new_generation[i] = sorted_gen[i]
    # fresh individuals
    start_fresh = int(ELITISM * NUM_OF_INDIVIDUALS)
    end_fresh = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    for i in range(start_fresh, end_fresh):
        new_generation[i] = fresh_individual()
    # dorobit tournament, crossover iba najlepsich 50 napr
    tournament(generation, new_generation)
    # dorobit mutaciu najlepsich
    start_mutation = int((ELITISM + CROSSOVER + FRESH) * NUM_OF_INDIVIDUALS)
    for _ in range(NUM_OF_INDIVIDUALS - start_mutation):
        index = random.randint(start_mutation, NUM_OF_INDIVIDUALS - 1)
        new_generation[index] = mutate_random(new_generation[index])

    return new_generation


def info_generation(generation):
    sorted_gen = [i[1] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
    # if sorted_gen[0]["fitness"] >= 5:
    #     return True
    #
    avg = round(sum([i["fitness"] for i in generation.values()]) / NUM_OF_INDIVIDUALS, 3)
    # print(avg)
    print(f"Generation avg fitness: {avg}")
    print(f"""Best individual info:
    fitness:        {sorted_gen[0]["fitness"]}
    path:           {sorted_gen[0]["path"]}
    path length:    {len(sorted_gen[0]["path"])}""")

    return False


def start():
    global NUM_OF_GENERATIONS
    global NUM_OF_INDIVIDUALS
    file_path = "init.txt"

    init(file_path)
    generation = first_generation()
    for i in range(NUM_OF_GENERATIONS):
        for curr_ind in range(NUM_OF_INDIVIDUALS):
            virtual_machine(generation[curr_ind])
            found_treasures(generation[curr_ind])
            set_fitness(generation[curr_ind])

        print(f"{i}. generation")
        info_generation(generation)
        # if i % 10 == 0:
        #     print(f"{i}. generation")
        #     done = info_generation(generation)
        #     if done:
        #         print("Finito. Found solution.")
        #         return

        generation = create_generation(generation)


start()


# sorted_gen = [i[0] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]


# # libraries
# import matplotlib.pyplot as plt
# import numpy as np
#
# # create data
# values = np.cumsum(np.random.randn(1000, 1))
#
# # use the plot function
# plt.plot(values)
# plt.show()
