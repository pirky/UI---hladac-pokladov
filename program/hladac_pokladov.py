import random
import numpy as np
import matplotlib.pyplot as plt

# random.seed(0)

NUM_OF_CELLS = 32  # number of cells initialized in first generation
ELITISM = 0.01  # percent of new individuals made by elitism
FRESH = 0.05  # percent of new individuals made for new generation
MUTATION = 0.05  # probability of mutation
TOURNAMENT = 0.2  # number of individuals in tournament

NUM_OF_GENERATIONS = 400
NUM_OF_INDIVIDUALS = 100

map_lines = 0
map_columns = 0
all_treasures = {}
start_line = 0
start_column = 0
best_individual = {"fitness": 0, "path": []}
averages = []
bests = []


# initialize program with data from txt file
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


# create first generation of individuals, one individual will have "NUM_OF_CELLS" filled with random values
def first_generation():
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


# returns direction of move
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


# virtual machine to execute program of one individual
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


# go through path and collect treasures
def found_treasures(individual):
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

        if curr_line < 0 or curr_line >= map_lines or curr_column < 0 or curr_column >= map_columns:  # check if I'm in the map
            individual["path"] = individual["path"][:counter]
            return

        curr_pos = [curr_line, curr_column]
        if curr_pos in all_treasures.values() and curr_pos not in individual["treasures"]:  # check for treasure
            individual["treasures"].append(curr_pos)

        if len(individual["treasures"]) == len(all_treasures):  # all treasures found, searching ends
            individual["path"] = individual["path"][:counter + 1]
            return
        counter += 1


# calculate fitness
def set_fitness(individual):
    if len(individual["path"]) == 0:
        return
    elif len(individual["treasures"]) == 0:
        individual["fitness"] = len(individual["path"]) / 1000

    individual["fitness"] = len(individual["treasures"]) + 1 - len(individual["path"]) / 1000


# create brand new individual
def fresh_individual():
    individual = {"fitness": 0, "path": [], "memory_cells": np.zeros(64, dtype=np.uint8), "treasures": list()}
    for i in range(NUM_OF_CELLS):
        individual["memory_cells"][i] = random.randint(0, 255)
    return individual


# crossover parents and return new individual
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


# roulette selection for selecting parents for crossover
def roulette(sorted_gen, new_generation):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    weights = [i["fitness"] for i in sorted_gen]

    for i in range(start_index, NUM_OF_INDIVIDUALS):
        mom = random.choices(sorted_gen, weights=weights)[0]
        dad = random.choices(sorted_gen, weights=weights)[0]

        while np.array_equal(dad["memory_cells"], mom["memory_cells"]):
            dad = random.choices(sorted_gen, weights=weights)[0]
        new_generation[i] = crossover(mom["memory_cells"], dad["memory_cells"])


# tournament selection for selecting parents for crossover
def tournament(sorted_gen, new_generation):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    num_tournament = int(TOURNAMENT * NUM_OF_INDIVIDUALS)
    for i in range(start_index, NUM_OF_INDIVIDUALS):
        mom = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x["fitness"])[0]
        dad = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x["fitness"])[0]

        while np.array_equal(dad["memory_cells"], mom["memory_cells"]):
            dad = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x["fitness"])[0]
        new_generation[i] = crossover(mom["memory_cells"], dad["memory_cells"])


# mutate individual's cells with random number
def mutate_random(individual):
    for i in range(64):
        if random.random() < MUTATION:
            individual["memory_cells"][i] = np.uint8(random.randint(0, 255))

    return {"fitness": 0, "path": [], "memory_cells": individual["memory_cells"], "treasures": list()}


# mutate individual's cells with inverting one random bit
def mutate_invert(individual):
    for i in range(64):
        if random.random() < MUTATION:
            shift = random.randint(0, 7)
            individual["memory_cells"][i] = np.uint8(individual["memory_cells"][i] ^ (1 << shift))

    return {"fitness": 0, "path": [], "memory_cells": individual["memory_cells"], "treasures": list()}


# mutate generation with particular mutation function
def mutation(new_generation, choices):
    start_index = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    for i in range(start_index, NUM_OF_INDIVIDUALS):
        if choices["mutation"] == "1":
            new_generation[i] = mutate_random(new_generation[i])
        else:
            new_generation[i] = mutate_invert(new_generation[i])


# create new generation using elitism, creating new individuals and rest are selected with particular selection method
def create_generation(sorted_gen, choices):
    new_generation = {}

    elite_end = int(ELITISM * NUM_OF_INDIVIDUALS)  # elitism
    for i in range(0, elite_end):
        new_generation[i] = {"fitness": 0, "path": [], "memory_cells": sorted_gen[i]["memory_cells"],
                             "treasures": list()}

    start_fresh = int(ELITISM * NUM_OF_INDIVIDUALS)  # fresh individuals
    end_fresh = int((ELITISM + FRESH) * NUM_OF_INDIVIDUALS)
    for i in range(start_fresh, end_fresh):
        new_generation[i] = fresh_individual()

    if choices["selection_type"] == "1":  # crossover
        tournament(sorted_gen, new_generation)
    else:
        roulette(sorted_gen, new_generation)

    mutation(new_generation, choices)  # mutation

    return new_generation


# return path as string
def path_print(path):
    string_path = ""
    for char in path:
        string_path += char + ","
    return string_path[:len(string_path) - 1]


# print info about generation -> average fitness and info about best individual
def info_generation(sorted_gen):
    global best_individual
    global averages
    global bests

    avg = round(sum([i["fitness"] for i in sorted_gen]) / NUM_OF_INDIVIDUALS, 3)
    averages.append(avg)
    bests.append(sorted_gen[0]["fitness"])
    path = path_print(sorted_gen[0]["path"])
    print(f"Generation avg fitness: {avg}")
    print(f"""Best individual info:
    fitness:        {sorted_gen[0]["fitness"]}
    path:           {path}
    path length:    {len(sorted_gen[0]["path"])}""")

    if best_individual["fitness"] < sorted_gen[0]["fitness"]:
        best_individual["fitness"] = sorted_gen[0]["fitness"]
        best_individual["path"] = [char for char in sorted_gen[0]["path"]]
        if sorted_gen[0]["fitness"] >= 5:
            print("\nFound new best global solution.")
            return True

    return False


# print final summarization with graph
def final_print():
    avg = round(sum(averages) / len(averages), 3)
    path = path_print(best_individual["path"])
    print(f"\nAverage fitness: {avg}")
    print(f"""Best individual info:
    fitness:        {best_individual["fitness"]}
    path:           {path}
    path length:    {len(best_individual["path"])}""")

    fig, ax = plt.subplots()
    ax.plot(averages, label="Average fitness")
    ax.plot(bests, label="Best fitness")
    ax.set(xlabel='Generations', ylabel='Fitness',
           title=f"INDIVIDUALS: {NUM_OF_INDIVIDUALS} GENERATIONS: {NUM_OF_GENERATIONS}\nELITISM: {ELITISM} FRESH: "
                 f"{FRESH} MUTATION: {MUTATION}")
    ax.grid()
    plt.legend()
    plt.show()


# starting function with
def start():
    global NUM_OF_GENERATIONS
    global NUM_OF_INDIVIDUALS
    global FRESH
    global ELITISM
    global MUTATION
    commands = ["1", "2"]
    choices = {"selection_type": "1", "mutation": "2"}

    print("""    ------------------------
        Treasures Hunter
       ------------------
        """)

    print("Type path to file with initialization information.")
    while True:
        try:
            file_path = input("File path: ")
            open(file_path, "r")
        except FileNotFoundError:
            print("File not found. Try it again.")
        else:
            break

    NUM_OF_INDIVIDUALS = int(input("Number of individuals: "))
    NUM_OF_GENERATIONS = int(input("Number of generations: "))

    print("\nFor \"tournament\" selection type \"1\"\nFor \"roulette\" selection type \"2\"")
    choices["selection_type"] = input("Your choice: ")
    while choices["selection_type"] != "1" and choices["selection_type"] != "2":
        print("Try it again.")
        print("For \"tournament\" selection type \"1\"\nFor \"roulette\" selection type \"2\"")
        choices["selection_type"] = input("Your choice: ")

    print("\nFor \"random number\" mutation type \"1\"\nFor \"invert\" mutation type \"2\"")
    choices["mutation"] = input("Your choice: ")
    while choices["mutation"] != "1" and choices["mutation"] != "2":
        print("Try it again.")
        print("For \"random number\" mutation type \"1\"\nFor \"invert\" mutation type \"2\"")
        choices["mutation"] = input("Your choice: ")

    print()
    MUTATION = float(input("Percent of probability to mutate: ")) / 100
    while MUTATION > 1:
        print("Try it again.")
        MUTATION = int(input("Percent of probability to mutate: ")) / 100

    ELITISM = float(input("Percent of elitism: ")) / 100
    FRESH = float(input("Percent of fresh individuals: ")) / 100
    while ELITISM + FRESH > 1:
        print("Wrong numbers. Try it again.")
        ELITISM = float(input("Percent of elitism: ")) / 100
        FRESH = float(input("Percent of fresh individuals: ")) / 100

    init(file_path)
    generation = first_generation()

    counter = 1
    while True:
        for i in range(NUM_OF_GENERATIONS):
            for curr_ind in range(NUM_OF_INDIVIDUALS):
                virtual_machine(generation[curr_ind])
                found_treasures(generation[curr_ind])
                set_fitness(generation[curr_ind])

            sorted_gen = [i[1] for i in sorted(generation.items(), reverse=True, key=lambda x: x[1]["fitness"])]
            print(f"{counter}. generation")

            done = info_generation(sorted_gen)
            if done:

                print("Press \"1\" if you want to keep looking for better solution.\nPress \"2\" if you want to end.")
                command = input("Type your option: ")
                while command not in commands:
                    print("Try it again.")
                    command = input("Type your option: ")
                if command == "2":
                    final_print()
                    return

            generation = create_generation(sorted_gen, choices)
            counter += 1

        print("Press \"1\" if you want to continue with generating generations\nPress \"2\" if you want to end.")
        command = input("Type your option: ")
        while command not in commands:
            print("Try it again.")
            command = input("Type your option: ")
        if command == "2":
            final_print()
            return


# function for doing n-tests
def testing():
    global map_lines
    global map_columns
    global all_treasures
    global start_line
    global start_column
    global best_individual
    global averages
    global bests
    number_of_tests = 5
    all_averages = []
    all_best = []
    global_avg = [0 for _ in range(NUM_OF_GENERATIONS)]
    global_best = [0 for _ in range(NUM_OF_GENERATIONS)]

    for _ in range(number_of_tests):
        map_lines = 0
        map_columns = 0
        all_treasures = {}
        start_line = 0
        start_column = 0
        best_individual = {"fitness": 0, "path": []}
        averages = []
        bests = []
        start()
        all_averages.append(averages)
        all_best.append(bests)

    fig, ax = plt.subplots()
    for i in range(number_of_tests):
        ax.plot(all_averages[i])

    ax.set(xlabel='Generations', ylabel='Fitness',
           title=f"ALL AVERAGES\nINDIVIDUALS: {NUM_OF_INDIVIDUALS} GENERATIONS: {NUM_OF_GENERATIONS}\nELITISM: "
                 f"{ELITISM} FRESH: {FRESH} MUTATION: {MUTATION}")
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    for i in range(number_of_tests):
        ax.plot(all_best[i])

    ax.set(xlabel='Generations', ylabel='Fitness',
           title=f"ALL BESTS\nINDIVIDUALS: {NUM_OF_INDIVIDUALS} GENERATIONS: {NUM_OF_GENERATIONS}\nELITISM: "
                 f"{ELITISM} FRESH: {FRESH} MUTATION: {MUTATION}")
    ax.grid()
    plt.show()

    for avg in all_averages:
        for i in range(len(avg)):
            global_avg[i] += avg[i] / number_of_tests

    for best in all_best:
        for i in range(len(best)):
            global_best[i] += best[i] / number_of_tests

    fig, ax = plt.subplots()
    ax.plot(global_avg, label="Average fitness")
    ax.plot(global_best, label="Best fitness")
    ax.set(xlabel='Generations', ylabel='Fitness',
           title=f"GLOBAL AVERAGE\nINDIVIDUALS: {NUM_OF_INDIVIDUALS} GENERATIONS: {NUM_OF_GENERATIONS}\nELITISM: "
                 f"{ELITISM} FRESH: {FRESH} MUTATION: {MUTATION}")
    ax.grid()
    plt.legend()
    plt.show()


start()
