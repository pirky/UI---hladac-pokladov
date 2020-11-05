import random
import numpy as np

random.seed(0)

NUM_OF_CELLS = 32           # number of cells initialized in first generation
ELITISM = 0.02              # percent of new individuals made by elitism
FRESH = 0.25                # percent of new individuals made for new generation
TOURNAMENT = 0.5            # percent of individuals in tournament
MUTATION = 0.5              # probability of mutation

NUM_OF_GENERATIONS = 400
NUM_OF_INDIVIDUALS = 100

map_lines = 0
map_columns = 0
all_treasures = {}
start_line = 0
start_column = 0
best_individual = {"fitness": 0, "path": []}
averages = 0


def init(file_path):
    global map_lines
    global map_columns
    global all_treasures
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
        all_treasures[i] = [int(data[treasure][0]), int(data[treasure][1][:1])]

    file.close()


def first_generation():
    global NUM_OF_INDIVIDUALS
    global NUM_OF_CELLS

    individuals = {}
    for i in range(NUM_OF_INDIVIDUALS):
        individuals[i] = {}
        individuals[i]["fitness"] = 0
        individuals[i]["path"] = []
        individuals[i]["memory_cells"] = np.zeros(64, dtype=np.uint8)
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
            instructions[address] = instructions[address].astype(np.uint8)

        elif operation == "01":  # decrement
            instructions[address] -= 1
            instructions[address] = instructions[address].astype(np.uint8)

        elif operation == "10":  # jump
            index = address

        elif operation == "11":  # write uses last 2 bits
            move = format(instructions[index], '#010b')[8:]
            individual["path"].append(write(move))

        program_counter += 1
        index += 1


def found_treasures(individual):
    global all_treasures
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

        if not 0 <= curr_line < map_lines or not 0 <= curr_column < map_columns:            # check if I'm in the map
            individual["path"] = individual["path"][:counter]
            return

        curr_pos = [curr_line, curr_column]
        if curr_pos in all_treasures.values() and curr_pos not in individual["treasures"]:  # check for treasure
            individual["treasures"].append(curr_pos)

        if len(individual["treasures"]) == len(all_treasures):
            return
        counter += 1


def set_fitness(individual):
    if len(individual["path"]) == 0:
        return
    elif len(individual["treasures"]) == 0:
        individual["fitness"] = len(individual["path"]) / 1000

    individual["fitness"] = len(individual["treasures"]) + 1 - len(individual["path"]) / 1000


def fresh_individual():
    individual = {"fitness": 0, "path": [], "memory_cells": np.zeros(64, dtype=np.uint8), "treasures": list()}
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


def roulette(sorted_gen, new_generation):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    weights = [i["fitness"] for i in sorted_gen]

    for i in range(start_index, NUM_OF_INDIVIDUALS):
        parents = random.choices(sorted_gen, weights=weights, k=2)
        new_generation[i] = crossover(parents[0]["memory_cells"], parents[1]["memory_cells"])


def tournament(sorted_gen, new_generation):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    num_tournament = int(TOURNAMENT * NUM_OF_INDIVIDUALS)

    for i in range(start_index, NUM_OF_INDIVIDUALS):
        sorted_individuals = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x["fitness"])
        new_generation[i] = crossover(sorted_individuals[0]["memory_cells"], sorted_individuals[1]["memory_cells"])


def mutate_little(individual):
    for i in range(2):
        index = random.randint(0, 63)
        individual["memory_cells"][index] = np.uint8(random.randint(0, 255))

    return {"fitness": 0, "path": [], "memory_cells": individual["memory_cells"], "treasures": list()}


def mutate_random(individual):
    mut_count = random.randint(5, 15)
    for i in range(mut_count):
        index = random.randint(0, 63)
        individual["memory_cells"][index] = np.uint8(random.randint(0, 255))

    return {"fitness": 0, "path": [], "memory_cells": individual["memory_cells"], "treasures": list()}


def mutate_switch(individual):
    blok_length = 6
    index_1 = random.randint(0, 63 - blok_length)
    index_2 = random.randint(0, 63 - blok_length)

    while abs(index_1 - index_2) < blok_length:
        index_1 = random.randint(0, 63 - blok_length)
        index_2 = random.randint(0, 63 - blok_length)

    minimum = min(index_1, index_2)
    copy_1 = individual["memory_cells"][minimum: minimum + blok_length]
    maximum = max(index_1, index_2)
    copy_2 = individual["memory_cells"][maximum: maximum + blok_length]

    for i in range(minimum, maximum + blok_length):
        if i < minimum + blok_length:
            individual["memory_cells"][i] = copy_2.pop()
        elif i >= maximum:
            individual["memory_cells"][i] = copy_1.pop()

    return {"fitness": 0, "path": [], "memory_cells": individual["memory_cells"], "treasures": list()}


def mutation(new_generation):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    for i in range(start_index, len(new_generation)):
        if random.random() < MUTATION:
            new_generation[i] = mutate_random(new_generation[i])


def create_generation(sorted_gen):
    new_generation = {}

    elite_end = int(ELITISM * NUM_OF_INDIVIDUALS)       # elitism
    for i in range(0, elite_end):
        new_generation[i] = {"fitness": 0, "path": [], "memory_cells": sorted_gen[i]["memory_cells"], "treasures": list()}

    start_fresh = int(ELITISM * NUM_OF_INDIVIDUALS)     # fresh individuals
    end_fresh = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    for i in range(start_fresh, end_fresh):
        new_generation[i] = fresh_individual()

    roulette(sorted_gen, new_generation)                # crossover

    mutation(new_generation)                            # mutation

    return new_generation


def path_print(path):
    string_path = ""
    for char in path:
        string_path += char + ","
    return string_path[:len(string_path) - 1]


def info_generation(sorted_gen):
    global best_individual

    avg = round(sum([i["fitness"] for i in sorted_gen]) / NUM_OF_INDIVIDUALS, 3)
    print(f"avg:\t{avg}", "\tbest:\t{}".format(sorted_gen[0]["fitness"]))
    # path = path_print(sorted_gen[0]["path"])
    # print(f"Generation avg fitness: {avg}")
    # print(f"""Best individual info:
    # fitness:        {sorted_gen[0]["fitness"]}
    # path:           {path}
    # path length:    {len(sorted_gen[0]["path"])}""")

    if best_individual["fitness"] < sorted_gen[0]["fitness"]:
        best_individual["fitness"] = sorted_gen[0]["fitness"]
        best_individual["path"] = [char for char in sorted_gen[0]["path"]]
        if sorted_gen[0]["fitness"] >= 5:
            print("\nFound new best global solution.")
            return True

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

        sorted_gen = [i[1] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
        print(f"{i + 1}. generation")
        done = info_generation(sorted_gen)
        if done:
            commands = ["1", "2"]
            print("Press \"1\" if you want to keep looking for better solution.\nPress \"2\" if you want to end.")
            command = input("Type your option: ")
            while command not in commands:
                print("Try it again.")
                command = input("Type your option: ")
            if command == "2":
                return

        generation = create_generation(sorted_gen)


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
