#############################################################################################################
############################## DO NOT TOUCH ANYTHING UNTIL I TELL YOU TO DO SO ##############################
#############################################################################################################

# This is the skeleton program NatAlgDiscrete.py around which you should build your implementation.
#
# On input GCGraphA.txt, say, the output is a witness set that is in the file WitnessA_<timestamp>.txt where
# <timestamp> is a timestamp so that you do not overwrite previously produced witnesses. You can always
# rename these files. The witness file is placed in a folder called "abcd12" (or whatever your username is)
# which is assumed to exist.
#
# It is assumed that all graph files are in a folder called GraphFiles that lies in the same folder as
# NatAlgDiscrete.py. So, there is a folder that looks like {NatAlgDiscrete.py, GraphFiles, abcd12}.

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "ccbd24"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "AB"

#################################################################
#### ENTER THE CODE FOR THE GRAPH PROBLEM YOU ARE OPTIMIZING ####
#################################################################

problem_code = "GC"

#############################################################
#### ENTER THE DIGIT OF THE INPUT GRAPH FILE (A, B OR C) ####
#############################################################

graph_digit = "A"

#############################################################################################################
############################## DO NOT TOUCH ANYTHING UNTIL I TELL YOU TO DO SO ##############################
#############################################################################################################

import time
import os
import random
import math

def get_a_timestamp_for_an_output_file():
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp

def read_the_graph_file(problem_code, graph_digit):
    vertices_tag = "number of vertices = "
    len_vertices_tag = len(vertices_tag)
    edges_tag = "number of edges = "
    len_edges_tag = len(edges_tag)
    if problem_code == "GC":
        colours_tag = "number of colours to use = "
        len_colours_tag = len(colours_tag)
    if problem_code == "GP":
        sets_in_partition_tag = "number of partition sets = "
        len_sets_in_partition_tag = len(sets_in_partition_tag)
    input_file = "GraphFiles/" + problem_code + "Graph" + graph_digit + ".txt"
    
    f = open(input_file, 'r')
    whole_line = f.readline()
    vertices = whole_line[len_vertices_tag:len(whole_line) - 1]
    v = int(vertices)
    whole_line = f.readline()
    edges = whole_line[len_edges_tag:len(whole_line) - 1]
    if problem_code == "GC":
        whole_line = f.readline()
        colours = whole_line[len_colours_tag:len(whole_line) - 1]
        colours = int(colours)
    if problem_code == "GP":
        whole_line = f.readline()
        sets_in_partition = whole_line[len_sets_in_partition_tag:len(whole_line) - 1]
        sets_in_partition = int(sets_in_partition)
    matrix = []
    for i in range(0, v - 1):
        whole_line = f.readline()
        splitline = whole_line.split(',')
        splitline.pop(v - 1 - i)
        splitline.insert(0, 0)
        matrix.append(splitline[:])
    matrix.append([0])
    for i in range(0, v):
        for j in range(0, i):
            matrix[j][i] = int(matrix[j][i])
            matrix[i].insert(j, matrix[j][i])
    f.close()

    edges = []
    for i in range(0, v):
        for j in range(i + 1, v):
            if matrix[i][j] == 1:
                edges.append([i, j])

    if problem_code == "GC":
        return v, edges, matrix, colours
    elif problem_code == "GP":
        return v, edges, matrix, sets_in_partition
    else:
        return v, edges, matrix
 
if problem_code == "GC":
    v, edges, matrix, colours = read_the_graph_file(problem_code, graph_digit)
elif problem_code == "GP":
    v, edges, matrix, sets_in_partition = read_the_graph_file(problem_code, graph_digit)
else:
    v, edges, matrix = read_the_graph_file(problem_code, graph_digit)

#######################################
#### READ THE FOLLOWING CAREFULLY! ####
#######################################

# For the problem GC, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph
#   - 'edges' = a list of the edges of the graph (just in case you need them)
#   - 'matrix' = the full adjacency matrix of the graph
#   - 'colours' = the maximum number of colours to be used when colouring

# For the problem CL, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph
#   - 'edges' = a list of the edges of the graph (just in case you need them)
#   - 'matrix' = the full adjacency matrix of the graph

# For the problem GP, the graph data has now been read into the following reserved variables:
#   - 'v' = the number of vertices of the graph
#   - 'edges' = a list of the edges of the graph (just in case you need them)
#   - 'matrix' = the full adjacency matrix of the graph
#   - 'sets_in_partition' = the number of sets in any partition

# These are reserved variables and need to be treated as such, i.e., use these names for these
# concepts and don't re-use the names.

# For the problem GC, you will produce a colouring in the form of a list of n integers called
# 'colouring' where the entries range from 1 to 'colours'. Note! 0 is disallowed as a colour!
# You will also produce an integer in the variable 'conflicts' which denotes how many edges
# are such that the two incident vertices are identically coloured (of course, your aim is to
# minimize the value of 'conflicts').

# For the problem CL, you will produce a clique in the form of a list of n integers called
# 'clique' where the entries are either 0 or 1. If 'clique[i]' = 1 then this denotes that the
# vertex i is in the clique.
# You will also produce an integer in the variable 'clique_size' which denotes how many vertices
# are in your clique (of course, your aim is to maximize the value of 'clique_size').

# For the problem GP, you will produce a partition in the form of a list of n integers called
# 'partition' where the entries are in {1, 2, ..., 'sets_in_partition'}. Note! 0 is not the
# name of a partition set! If 'partition[i]' = j then this denotes that the vertex i is in the
# partition set j.
# You will also produce an integer in the variable 'conflicts' which denotes how many edges are
# incident with vertices in different partition sets (of course, your aim is to minimize the
# value of 'conflicts').

# In consequence, the following additional variables are reserved:
#   - 'colouring'
#   - 'conflicts'
#   - 'clique'
#   - 'clique_size'
#   - 'partition'

# The various algorithms all have additional parameters (see the lectures). These parameters
# are detailed below and are referred to using the following reserved variables.
#
# AB (Artificial bee colony)
#   - 'n' = dimension of the optimization problem
#   - 'num_cyc' = number of cycles to iterate
#   - 'N' = number of employed bees / food sources
#   - 'M' = number of onlooker bees
#   - 'lambbda' = limit threshold

if alg_code == 'AB':
    n = int(v/10)
    num_cyc = 5000
    N = 200
    M = 500
    lambbda = 500

# FF (Firefly)
#   - 'n' = dimension of the optimization problem
#   - 'num_cyc' = number of cycles to iterate
#   - 'N' = number of fireflies
#   - 'lambbda' = light absorption coefficient
#   - 'alpha' = scaling parameter

if alg_code == 'FF':
    n = None
    num_cyc = None
    N = None
    lambbda = None
    alpha = None

# CS (Cuckoo search)
#   - 'n' = dimension of optimization problem
#   - 'num_cyc' = number of cycles to iterate
#   - 'N' = number of nests
#   - 'p' = fraction of local flights to undertake
#   - 'q' = fraction of nests to abandon
#   - 'alpha' = scaling factor for Levy flights
#   - 'beta' = parameter for Mantegna's algorithm

if alg_code == 'CS':
    n = None
    num_cyc = None
    N = None
    p = None
    q = None
    alpha = None
    beta = None

# WO (Whale optimization)
#   - 'n' = dimension of optimization problem
#   - 'num_cyc' = number of cycles to iterate
#   - 'N' = number of whales
#   - 'b' = spiral constant

if alg_code == 'WO':
    n = None
    num_cyc = None
    N = None
    b = None

# BA (Bat)
#   - 'n' = dimension of optimization problem
#   - 'num_cyc' = number of cycles to iterate
#   - 'N' = number of fireflies
#   - 'sigma' = scaling factor
#   - 'f_min' = minimum frequency
#   - 'f_max' = maximum frequency

if alg_code == 'BA':
    n = None
    num_cyc = None
    N = None
    sigma = None
    f_min = None
    f_max = None

# These are reserved variables and need to be treated as such, i.e., use these names for these
# parameters and don't re-use the names! Note that I have initialized them as I write them in the
# output file even if you don't actually use them. If you use the parameters then don't touch anything
# above but initialize them for yourself below. Also, you may introduce additional parameters if you
# wish (below) but they won't get written to the output file.

start_time = time.time()

#############################################################################################################
########################################### ENTER YOUR CODE BELOW ###########################################
#############################################################################################################


########################################### Functions to Generate Populations of Sequences for vertices###########################################

# This is generating one random sequence of vertex numbers 
def gen_sequence() -> list:
    '''
    This is to generate a single sequence of vertices
    '''
    list_of_vertices = range(v)
    return  random.sample(list_of_vertices, v)


# This calls the function above to generate a population of random sequences of vertices
def gen_pop():
    '''
    This generates a populaiton of randomly shuffled vertices 
    which would then be coloured 
    ''' 
    population = [gen_sequence() for i in range(N)]
    return population

########################################### Function to Generate and adjacency dictionary for the vertices ###########################################

def dic_edges() -> dict:
    '''
    This function returns a dictionary with key being the vertex number and the values being a list of adjacent vertices
    This is made so its more efficient during coloring and because otherwise the edges would need to be iterated through
    to generate a sample for the population each time

    Eg : 
    adjacency_dic[vertex_number] = [Adjacent_vertex1, ...., 3]
    '''
    adjacency_dictionary = {}
    for i in range(v):
        adjacency_dictionary[i] = [j for j in range(v) if matrix[i][j] == 1]
    return adjacency_dictionary


########################################### Functions to Generate coloring ###########################################


def generate_color_sequence_half_greedy(one_sequence : list, adjacency_dictionary : dict):
    """
    This is to generate a coloring for one sequence or one food souece
    * `one_sequence` -> one food source for eg [0,1,2,3,...,v-1]
    * `adjacency_dictionary` -> to recognise which vertices the vertices in the sequence share edges with
    """
    color_dictionary_for_sequence = {}
    possible_colors_avaliable = set(range(1, colours + 1))
    node_available_colors = {i : possible_colors_avaliable.copy() for i in range(v)}
    for node in one_sequence:
        if len(node_available_colors[node]) == 0:
            # At some point one of these nodes in the sequences have gottten all their neighbours colored, dont have possibility of assigning color
            color_dictionary_for_sequence[node] = random.randint(a=1, b=colours)
        else:
            color_dictionary_for_sequence[node] = min(node_available_colors[node])
            for adjacent_vertex in adjacency_dictionary[node]:
                if min(node_available_colors[node]) in node_available_colors[adjacent_vertex]:
                    node_available_colors[adjacent_vertex].remove(min(node_available_colors[node]))
                    if len(node_available_colors[adjacent_vertex]) == 0:
                        color_dictionary_for_sequence[adjacent_vertex] = random.randint(a=1, b=colours)
    return color_dictionary_for_sequence


def coloring_half_greddy(sequences_of_nodes, adjacency_dictionary):
    """
    This calls the function above but for the whole population of sequence
    * `sequence_of_nodes` -> just the population of food sources
    * `adjacency_dictionary` -> to recognise which vertices the vertices in the sequence share edges with

    Returns a List[dict] with keys being the vertex number and values being the colorings 
    """
    population_of_colored_nodes = []
    for i in range(N):
        population_of_colored_nodes.append(generate_color_sequence_half_greedy(sequences_of_nodes[i],adjacency_dictionary))

    return population_of_colored_nodes


################################## Experiment Two using different fitness  this below is the one reccomended in the lectures ###########################################
# Doesnt work very well with this problem

# Calculate the number of unique colors used 
def unique_colors(color_seq : dict) -> int:
    '''
    Return the number of unique colors in a coloring
    '''
    return len(set(color_seq.values()))


# Here calculate fitness using the lecture suggested 
def calc_fitness_lecture(num_bad_edges : int, colouring_seq : dict) -> float:

    # the formula for fitness although doesnt perform well as number of bad edges overarched by the |v|^3 especially on larger graphs
    fitness = math.pow(v,3) + ((num_bad_edges * v) + unique_colors(colouring_seq))
    return fitness


# Calculate the fitness for the whole population
def calc_fitness_pop_lecture(num_bad_edges : list, colorings ) -> list:
    '''
    Calculate the lecture suggested fitness for the whole population 
    '''
    # Creating array of fitness
    fitness = [0] * N

    # Calculating for the whole population
    for i in range(N):
        fitness[i] = calc_fitness_lecture(num_bad_edges=num_bad_edges[i], colouring_seq=colorings[i])
    return fitness


########################################### Calculating number of conflicts and fitness ###########################################

def bad_edge_and_conflicts(colourings_seq : dict, bad_edges : bool = False, conflicts : bool = False):
    """
    This calculates the number of conflicts that arise from the colourings of the different sequences and set of badly coloured edges
    * `bad_edges` : boolean so you can choose if you wanna calculate the number of bad edges or number of conflicts
    * `conflicts` : boolean so you can calculate either num_bad_edges or conflicts or even both if both are true
    * `If both booleans are True` -> wil genertae a tuple
    """
    if conflicts:
        length = len(colourings_seq)
        num_conflicts = 0
        for i in range(0, length):
            for j in range(i + 1, length):
                if matrix[i][j] == 1 and colourings_seq[i] == colourings_seq[j]:
                    num_conflicts = num_conflicts + 1
        return num_conflicts

    if bad_edges:
        num_bad_edges = 0
        for i in range(0, v):
            for j in range(i + 1, v):
                if matrix[i][j] == 1 and colourings_seq[i] == colourings_seq[j]:
                    num_bad_edges = num_bad_edges + 1
        return num_bad_edges

    if bad_edges and conflicts :
        return num_bad_edges, num_conflicts


########################################### Near Neighbour Algorithms for bee colony ###########################################

# First version this is all named the same in the report highlighted in blue 
def near_neighbourv1(sequence1, sequence2, vertex_number):
    '''
    * First Near neighbour version 
    * Chose 2 k, jdifferent sequences -> pop[k], pop[j]
    * Choose a vertex number
    * Only switch if p
    * Switch the vertex color of k to vertex color of k
    '''
    # Copying these sequences to prevent updating to the original population 
    sequence1_temp = sequence1.copy()
    sequence2_temp = sequence2.copy()
    # Returning the dictionary with the changed colour of the randomly chosen vertex
    sequence1_temp[vertex_number] = sequence2_temp[vertex_number]
    return sequence1_temp


# Second version this is all named the same in the report highlighted in blue 
# Very important this near_neighbour makes significant changes to the sequence so good if you are in a local minimum and cant escape it
def near_neighbourv2(colouring_dictionary_seq):
    '''
    Here we choose a source and then alter, by first arranging in canotical form, then choose one vertex 
    for each color and then randomly change the color of the nodes among differnet colors(classes)
    Major change so get you out of the local min as changes the colour of colours number of vertices 
    '''
    # Copying the dictionary, this is just a coloring dictionary for one sequence in the population
    coloring_dictionary_copy = colouring_dictionary_seq.copy()

    # Converting in the coloquial form 
    # Goes from vertex_number : color [1: 2,...] to # color : vertex_numbers for eg {1 : [1, 2, 3], ... }
    coloquial_format = {n:[keys for keys in colouring_dictionary_seq.keys() if colouring_dictionary_seq[keys] == n] for n in set(colouring_dictionary_seq.values())}


    # For each color in the coloquial form or in this class 
    for color in coloquial_format.keys():

        # choosing a random vertex
        random_vertex = random.sample(coloquial_format[color],1)[0]

        # Making sure the new color is not the same as the old color 
        new_color = random.randint(1, colours)
        while new_color == color:
            new_color = random.randint(1, colours)
        
        # I update the original coloring dictionary by changing this vertex number with the new color
        coloring_dictionary_copy[random_vertex] = new_color
    return coloring_dictionary_copy


# Third version but this is useless as this does change the sequence however the sequence is only relevant when generating the greedy colorings
# Not good for near neighbours as would represent the same solution 
def near_neighbourv3(sequence1 : dict):
    '''
    Multiple consideations of 
    '''
    potenital_vertices = random.sample(range(0,v), 2)
    vertex1 = potenital_vertices[0]
    vertex2 = potenital_vertices[1]
    sequence1_temp = sequence1.copy()
    sequence1_temp[vertex1],sequence1_temp[vertex2]  = sequence1_temp[vertex2],sequence1_temp[vertex1]
    return sequence1_temp


########################################### Bee colony Algorithm ###########################################


def bee():

    # First step generate the population of different sequences of vertex numbers
    population = gen_pop()

    # Generate a dictionary with vertex_number : [] and list of vertices that it shares edges with
    adjacent_vertices = dic_edges()

    # This is necessary to generate the hybrid colourings for the whole population of sequences
    colored_population = coloring_half_greddy(sequences_of_nodes=population, adjacency_dictionary=adjacent_vertices)

    # Calculate an arrray of size N with the fitness associated with each food source necessary for the roulette wheel
    # The fitness used here is 1/number_of_conflicts, not usin the one in lectures as in that the number of vertices dwarfs
    # conflicts and not great at minimiisng the number of conflicts
    fitness_pop = [(1/ bad_edge_and_conflicts(colored_population[i], conflicts=True)) for i in range(N)]

######################################### UnComment below this to test out with fitness from lectures###########################################
# This has to be done wherever the fitness is being updated in the bee colony code
# Present near wherever fitness is updated 

    # number_of_bad_edges = [bad_edge_and_conflicts(colourings_seq=colored_population[i], bad_edges=True) for i in range(N)]
    # fitness_pop = 1 / calc_fitness_pop_lecture(num_bad_edges=number_of_bad_edges, colorings=colored_population)

######################################### UnComment above this to test out with fitness from lectures###########################################


    # Keeping track of the minimum number of conflicts in our population
    min_conflicts = int(1/max(fitness_pop))

    # Keeping track of the best associated coloring to that minimum number of conflicts
    best_global_colouring = colored_population[fitness_pop.index(max(fitness_pop))]

    # This is initialising the number of epochs
    t = 0

    # Here this is a variable associated for each food source so we know when to abandon a food source
    limit = [0] * N

    # This is for convergence of solutions if the near neighbours arent producing better solutions than the greedy it gets incremented
    # This decided which near neighbour algorithm we are using 
    threshold_for_convergence = 0


    while t <= num_cyc:
        
        # This is just to eliminate the code before 60 seconds to ensure it is ran in time
        func_time = time.time()
        time_elapsed = round(func_time - start_time, 1)

        # Just doing this becsue graph C takes a lot longer to run so need to decrease this time otherwise doesnt reach here on time 
        # to eliminate at 60 secods
        if graph_digit == 'C':
            max_time = 25
        else:
            max_time = 55
        if time_elapsed >= max_time:
            
            # Returning the required best coloring and best number of conflicts for the colorings, had to do this sorted thing to match datastructures
            return min_conflicts, list(dict(sorted(best_global_colouring.items())).values())

        # Going for the epochs now 
        for i in range(N + M):

            # Employee bees here 
            if i <= N-1:
                k = i

            # Onloooker bees
            else:
                # Roulette wheel to decide which solution to explore for the onlooker bees
                k = random.choices(list(range(N)), weights=fitness_pop)[0]
            
            # Generating another random food source along with the original one
            random_sequence = random.randint(0, N-1)

            # Ensuring these 2 food sources are not the same 
            while random_sequence == k:
                random_sequence = random.randint(0, N-1)

            # Choosing a random vertex which we use to change 
            random_vertex = random.randint(0, v-1)
            

########################################## Near Neighbour Algorithms Explaination ###########################################

# One of the experimentations to just use without this experimentation 
# 1) 2 differnet versions of the near neighbour being used
# If threshold of convergence is less than 400 that means for now noemal near neighbour has not generated 400 worse solutions than the 
# global minimum, so just keep continuing the normal version for now

# If the threshold foes over 400 it means that no better solution than the original greedy could be gotten from choosing 2 random 
# food sources, so if > 400 
# Choose the best solution the global minimum coloring and then choose another random food source
# Perform the near neighbour on the best and another random sequence
# If that results in a better solution than the global maximum reset threshold
# If that still doesnt work then increment threshold and if it is greater than 600
# Then we emplot version 2 this would chnage the colours of the same number of vertices as number of colours allowed to be used
# AS greater than 600 is an indication of a local minima this would help move very far away from that local minimum and explore
# a differnet region of the solution space hopefully


########################################## Comment below this to test out without near neighbour experiment###########################################

            # To chnage a lot and esape the local minimum
            if threshold_for_convergence > 600:

                # Choosing the best possible solution or the global minima
                k = fitness_pop.index(max(fitness_pop))

                # Perform version 2 still has a random aspect as you choose random colors and random vertices
                new_potential_colouring = near_neighbourv2(colored_population[k])

                # Calculate the fitness of old fitness for comaprison later
                old_potential_fitness = 1 / min_conflicts
                
                # Reset the threshold here because more than likely you escaped local minima and dont want to do this again and again
                # As then you would never converge to a good point
                threshold_for_convergence = 0
            
            # Version one choose 
            elif threshold_for_convergence > 400:

                # Choosing the best possible solution or the global minima 
                k = fitness_pop.index(max(fitness_pop))

                # Then use the current global minima choosing another random food source
                # Potentially come up with a better solution 
                new_potential_colouring = near_neighbourv1(colored_population[k], colored_population[random_sequence], random_vertex)

                # Calculate the fitness of old fitness for comaprison later
                old_potential_fitness = 1 / min_conflicts

            # Normal near neighbour when threshold is not reached
            else:
                # The new potential coloring you get from choosing 2 random food sources
                new_potential_colouring = near_neighbourv1(colored_population[k], colored_population[random_sequence], random_vertex)

                # Calculate the fitness of old fitness for comaprison later
                old_potential_fitness = 1 / bad_edge_and_conflicts(colored_population[k], conflicts=True)

                # This is the code for the different fitness function
                # old_potential_fitness = 1 / (calc_fitness_lecture(bad_edge_and_conflicts(colored_population[k], bad_edges=True), colored_population[k]))


            # Calculate the fitness of this newly generated sequence 
            new_potential_fitness = 1 / bad_edge_and_conflicts(new_potential_colouring, conflicts=True)

########################################## Comment above this to test out without near neighbour experiment###########################################


########################################## Uncomment below this to test out without near neighbour experiment###########################################

            # If want to use without experimentation of near neighbours please uncomment this below and comment the whole if else 
            # statements above it would be clear that without this no impovements are made
            # The new potential coloring you get from choosing 2 random food sources
            # new_potential_colouring = near_neighbourv1(colored_population[k], colored_population[random_sequence], random_vertex)

            # # Calculate the fitness of old fitness for comaprison later
            # old_potential_fitness = 1 / bad_edge_and_conflicts(colored_population[k], conflicts=True)

            # # Calculate the fitness of this newly generated sequence 
            # new_potential_fitness = 1 / bad_edge_and_conflicts(new_potential_colouring, conflicts=True)

########################################## Uncomment above this to test out without near neighbour experiment###########################################

            # Compare if this newly generated is better than the old oen
            if new_potential_fitness > old_potential_fitness:

                # If the generated solution is better than the chosen sequence then you update the 
                # population of colorings
                colored_population[k] = new_potential_colouring

                # Also reset the limit for that food source so its not being abandoned
                limit[k] = 0

                # Now to see if this newly generated sequence is better than the global minima generted 
                if new_potential_fitness > max(fitness_pop):

                    # print('Global minima being updted, so greedy solution being imroved by bee')

                    # Update the number of minimum conflicts with the new minima
                    min_conflicts = int(1/new_potential_fitness)

                    # Update the global colorings with the new ones disocvered now
                    best_global_colouring = colored_population[k]

                    # Reset threshold as we found a better global solution 
                    threshold_for_convergence = 0
                
                # Now update the fitness list for the population
                fitness_pop[k] = new_potential_fitness
            
            # if no better soltuion generated
            else:

                # Increase the limit for that food source by 1
                limit[k] +=1

                # Also increase the threshold for that convergence
                threshold_for_convergence +=1
        
        # Checking if any soltutions need to be abandoned and the scout bees need to generate new random solutions
        for i in range(N):

            # This thrreshold determines abandoning of any solutions 
            if limit[i] > lambbda:

                # If reached generate a new food source reminder not the full population
                population[i] = gen_sequence()

                # Generate the coloring for this new sequence
                colored_population[i] = generate_color_sequence_half_greedy(population[i], adjacent_vertices)

                # Now update the fitness list as well to take this into account 
                fitness_pop[i] = 1 / bad_edge_and_conflicts(colored_population[i], conflicts=True)


                # This is for calculating fitness using lecture fitness uncomment if needed
                # fitness_pop[i] = 1 / (calc_fitness_lecture(bad_edge_and_conflicts(colored_population[i], bad_edges=True), colored_population[i]))
                

                # But as this is a greedy solution check if this new solution generated is potentially a global minimum
                min_conflicts = int(1/max(fitness_pop))

                # Update the gloabl best colouring if this new solution generated is actually the new best soluiton
                best_global_colouring = colored_population[fitness_pop.index(max(fitness_pop))]

                # Then reset the limit for this particular food source
                limit[i] = 0
        
        # One epoch done 
        t +=1

    # Return the best solutions here 
    return min_conflicts, list(dict(sorted(best_global_colouring.items())).values())

conflicts, colouring = bee()

#############################################################################################################
################################## DO NOT TOUCH ANYTHING BELOW THIS COMMENT #################################
#############################################################################################################

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

# You should now have computed the list 'colouring' and integer 'conflicts', if you are solving GC;
# the list 'clique' and the integer 'clique_size', if you are solving CL; or the list 'partition' and the
# integer 'conflicts', if you are solving GP.

timestamp = get_a_timestamp_for_an_output_file()
witness_set = username + "/Witness" + graph_digit + "_" + timestamp + ".txt"

f = open(witness_set, "w")

f.write("username = {0}\n".format(username))
f.write("graph = {0}Graph{1}.txt with (|V|,|E|) = ({2},{3})\n".format(problem_code, graph_digit, v, len(edges)))
if problem_code == "GC":
    f.write("colours-to-use = {0}\n".format(colours))
if problem_code == "GP":
    f.write("number of partition sets = {0}\n".format(sets_in_partition))
f.write("algorithm = {0}\n".format(alg_code))
if alg_code == "AB":
    f.write("associated parameters [n, num_cyc, N, M, lambbda] = ")
    f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n,num_cyc,N,M,lambbda))
elif alg_code == "FF":
    f.write("associated parameters [n, num_cyc, N, lambbda, alpha] = ")
    f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n,num_cyc,N,lambbda,alpha))
elif alg_code == "CS":
    f.write("associated parameters [n, num_cyc, N, p, q, alpha, beta] = ")
    f.write("[{0}, {1}, {2}, {3}, {4}, {5}, {6}]\n".format(n,num_cyc,N,p,q,alpha,beta))
elif alg_code == "WO":
    f.write("associated parameters [n, num_cyc, N, b] = ")
    f.write("[{0}, {1}, {2}, {3}]\n".format(n,num_cyc,N,b))
elif alg_code == "BA":
    f.write("associated parameters [n, num_cyc, sigma, f_max, f_min] = ")
    f.write("[{0}, {1}, {2}, {3}, {4}]\n".format(n,num_cyc,sigma,f_max,f_min))
if problem_code == "GC" or problem_code == "GP":
    f.write("conflicts = {0}\n".format(conflicts))
else:
    f.write("clique size = {0}\n".format(clique_size))
f.write("elapsed time = {0}\n".format(elapsed_time))

if problem_code == "GC":
    error = []
    length = len(colouring)
    if length != v:
        error.append("*** error: 'colouring' has length " + str(length) + " but should have length " + str(v) + "\n")
    bad_colouring = False
    for i in range(0, length):
        if colouring[i] < 1 or colouring[i] > colours:
            bad_colouring = True
            break
    if bad_colouring == True:
        error.append("*** error: 'colouring' uses illegal colours \n")
    true_conflicts = 0
    for i in range(0, length):
        for j in range(i + 1, length):
            if matrix[i][j] == 1 and colouring[i] == colouring[j]:
                true_conflicts = true_conflicts + 1
    if conflicts != true_conflicts:
        error.append("*** error: you claim " + str(conflicts) + " but there are actually " + str(true_conflicts) + " conflicts\n")
    if error != []:
        print("I am saving your colouring into a witness file but there are errors:")
        for item in error:
            print(item)
    for i in range(0, length):
        f.write("{0},".format(colouring[i]))
        if (i + 1) % 40 == 0:
            f.write("\n")
    if length % 40 != 0:
        f.write("\n")
elif problem_code == "GP":
    error = []
    length = len(partition)
    if length != v:
        error.append("*** error: 'partition' has length " + str(length) + " but should have length " + str(v) + "\n")
    bad_partition = False
    for i in range(0, length):
        if partition[i] < 1 or partition[i] > sets_in_partition:
            bad_partition = True
            break
    if bad_partition == True:
        error.append("*** error: 'partition' uses illegal set numbers \n")
    true_conflicts = 0
    for i in range(0, length):
        for j in range(i + 1, length):
            if matrix[i][j] == 1 and partition[i] != partition[j]:
                true_conflicts = true_conflicts + 1
    if conflicts != true_conflicts:
        error.append("*** error: you claim " + str(conflicts) + " but there are actually " + str(true_conflicts) + " conflicts\n")
    if error != []:
        print("I am saving your partition into a witness file but there are errors:")
        for item in error:
            print(item)
    for i in range(0, length):
        f.write("{0},".format(partition[i]))
        if (i + 1) % 40 == 0:
            f.write("\n")
    if length % 40 != 0:
        f.write("\n")
else:
    error = []
    length = len(clique)
    if length != v:
        error.append("*** error: 'clique' has length " + str(length) + " but should have length " + str(v) + "\n")
    bad_clique = False
    for i in range(0, length):
        if clique[i] != 0 and clique[i] != 1:
            bad_clique = True
            break
    if bad_clique == True:
        error.append("*** error: 'clique' is not a list of 0s and 1s\n")
    true_size = 0
    for i in range(0, length):
        if clique[i] == 1:
            true_size = true_size + 1
    if clique_size != true_size:
        error.append("*** error: you claim a clique of size " + str(clique_size) + " but it actually has size " + str(true_size) + "\n")
    if error != []:
        print("I am saving your clique into a witness file but there are errors:")
        for item in error:
            print(item)
    for i in range(0, length):
        f.write("{0},".format(clique[i]))
        if (i + 1) % 40 == 0:
            f.write("\n")
    if length % 40 != 0:
        f.write("\n")

f.close()
    
print("witness file saved")


















    
