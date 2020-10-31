import random


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
        individuals[i]["treasures"] = 0
        for j in range(25):                              # prvych 25 byteov bude random
            individuals[i]["memory_cells"][j] = random.randint(0, 255)
    return individuals


def write(move):
    if move == "00":
        move = "H"
    if move == "01":
        move = "D"
    if move == "10":
        move = "P"
    if move == "11":
        move = "L"
    return move


def virtual_machine(individual):
    program_counter = 0
    index = 0
    instructions = [individual["memory_cells"][i] for i in range(64)]
    while program_counter <= 500 and 0 <= index < 64:
        operation = format(instructions[index], '#010b')[2:4]
        address = int(format(instructions[index], '#010b')[4:], 2)

        if operation == "00":       # increment
            instructions[address] += 1
            if instructions[address] == 256:
                instructions[address] = 0

        elif operation == "01":     # decrement
            instructions[address] -= 1
            if instructions[address] == -1:
                instructions[address] = 255

        elif operation == "10":     # jump
            index = address

        elif operation == "11":     # write uses last 2 bits
            move = format(instructions[index], '#010b')[8:]
            individual["path"].append(write(move))

        program_counter += 1
        index += 1


def start():
    num_of_generations = 1
    num_of_individuals = 50
    init()
    generation = first_generation(num_of_individuals)
    for i in range(num_of_generations):
        for curr_ind in range(num_of_individuals):
            print(f"{curr_ind}. individual")
            virtual_machine(generation[curr_ind])


start()
