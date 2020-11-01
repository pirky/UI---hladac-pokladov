import random

random.seed(0)

NUM_OF_CELLS = 32
map_lines = 0
map_columns = 0
all_tresures = {}
start_line = 0
start_column = 0


def init():
    global map_lines
    global map_columns
    global all_tresures
    global start_line
    global start_column
    map_lines = 7
    map_columns = 7
    all_tresures[0] = [1, 4]
    all_tresures[1] = [2, 2]
    all_tresures[2] = [3, 6]
    all_tresures[3] = [4, 1]
    all_tresures[4] = [5, 4]
    start_line = 6
    start_column = 3


def first_generation(num_of_individuals):
    individuals = {}
    for i in range(num_of_individuals):
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
            print("All treasures were found")
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
    copy_length = random.randint(0, 64)
    start_index = random.randint(0, 64 - copy_length)

    for i in range(64):
        if start_index <= i < start_index + copy_length:
            child.append(mom[i])
        else:
            child.append(dad[i])

    individual = {"fitness": 0, "path": [], "memory_cells": child, "treasures": list()}
    return individual


def tournament(generation, new_generation):
    for i in range(2, 89):
        sorted_individuals = sorted(random.choices(generation, k=20), reverse=True, key=lambda x: x["fitness"])
        new_generation[i] = crossover(sorted_individuals[0]["memory_cells"], sorted_individuals[1]["memory_cells"])

    return new_generation


def create_generation(generation):
    new_generation = {}
    sorted_gen = [i[1] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
    for i in range(2):  # best 2 are going to next generation without change
        new_generation[i] = sorted_gen[i]

    tournament(generation, new_generation)

    for i in range(89, 100):
        new_generation[i] = fresh_individual()

    # este mutovanie dorobit
    return new_generation


def info_generation(generation):
    sorted_gen = [i[1] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
    if sorted_gen[0]["fitness"] >= 5:
        return True
    print(f"""Best individual info:
    fitness:        {sorted_gen[0]["fitness"]}
    path:           {sorted_gen[0]["path"]}
    path length:    {len(sorted_gen[0]["path"])}""")
    return False


def start():
    num_of_generations = 1
    num_of_individuals = 100
    init()
    generation = first_generation(num_of_individuals)
    for i in range(num_of_generations):
        print(f"{i}. generation")
        for curr_ind in range(num_of_individuals):
            print(f"{curr_ind}. individual")
            virtual_machine(generation[curr_ind])
            found_treasures(generation[curr_ind])
            set_fitness(generation[curr_ind])

        done = info_generation(generation)
        if done:
            print("Finito. Found solution.")
            return
        generation = create_generation(generation)


start()


# sorted_gen = [i[0] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
