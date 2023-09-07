#############################################################################################################
############################## DO NOT TOUCH ANYTHING UNTIL I TELL YOU TO DO SO ##############################
#############################################################################################################

# This is the skeleton program NatAlgReal.py around which you should build your implementation.
# Read all comments below carefully.

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "ccbd24"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "AB"

#############################################################################################################
############################## DO NOT TOUCH ANYTHING UNTIL I TELL YOU TO DO SO ##############################
#############################################################################################################

import time
import random
import math
#
# The function f is 3-dimensional and you are attempting to MINIMIZE it.
# To compute the value f(a, b, c), call the function 'compute_f(a, b, c)'.
# The variables 'f', 'a', 'b' and 'c' are reserved.
# On termination your algorithm should be such that:
#   - the reserved variable 'min_f' holds the minimum value that you have computed for the function f 
#   - the reserved variable 'minimum' is a list of length 3 holding the minimum point that you have found.
#

def compute_f(a, b, c):
    f = a**2/4000 + b**2/4000 + c**2/4000 - (math.sin(math.pi/2 + a) * math.sin(math.pi/2 + b/math.sqrt(2)) \
                                          * math.sin(math.pi/2 + c/math.sqrt(3))) + 1
    return f

#
# The ranges for the values for a, b and c are [-500, 500]. The lists below hold the minimum and maximum
# values for each a, b and c, respectively, and you should use these list variables in your code.
#

min_range = [-500, -500, -500]
max_range = [500, 500, 500]

start_time = time.time()

#############################################################################################################
########################################### ENTER YOUR CODE BELOW ###########################################
#############################################################################################################


########################################### Hyperparamters for the bee colony algorithm ###########################################
# N = 100
# M = 200
# num_cyc = 50000
# lamb = 1000

N = 40
M = 40
num_cyc = 50000
lamb = 80

########################################### Functions to Generate Populations of diffeent points ###########################################


# Generating one 3 dimension point in space
def gen_food_source() -> list:
    '''
    Generating one point of 3 dimensions 
    '''
    return [random.uniform(a = -500, b = 500),random.uniform(a = -500, b = 500),random.uniform(a = -500, b = 500)]



def gen_pop() -> list:
    '''
    Generate population  of food sources which act as potential solutions 
    as there are 3 dimensions a list of 3 is appended to overall multidimensional list for the population
    '''
    population = [gen_food_source() for i in range(N)]
    return population


########################################### Near Neighbour Algorithms for bee colony ###########################################


# 1st Improvement added a schduler here to do a stochastic learning rate 
def near_neighbour(source1 : int, source2 : int, epoch : int, experientaiton : bool = False) -> list:
    """
    Performing the near neighbour algorithm
    * `experimentation` : a boolean argument to tets the experimentation where the change variable is acted like a schduler
    This schduler prevents to fall in the local minimum and explore a larger solution space around the convergeed solution
    """

    # choose a random dimension
    dim = random.randint(0,2)

    # Choosing how far you want to explore from the current food source
    change = random.uniform(-1,1) # some sort of schedule cosien or omething 

    # 1st Change here if experimentation true here 
    # epoch-based learning rate adaptation schedule
    # USed in SGD could be useful here as would let us converge 
    if experientaiton:
        if epoch ==0 :
            epoch += 1

        # Calculating the decay
        decay = abs(change) / epoch

        # Then calculating the new change
        change = change * 1/(1 + decay * epoch)
#
    # Calculating what the new point is going to be now 
    new_point = source1[dim] + change * (source2[dim]- source1[dim])

    # Ensuring to copy lists so no chnages in original
    new_list = source1.copy()

    # Now for the clipping so not going out of range
    if new_point < 0:

        # if this new point is less than 0 so it should be max(new_point, -500)
        new_point = max(new_point,min_range[dim])

    else: 

        # if this new point is greater than 0 so it should be min(new_point, 500)
        new_point = min(new_point,max_range[dim])

    # Now just return the updated list witth the new point
    new_list[dim] = new_point

    return new_list


########################################### Functions for calculating fitness of the food sources ###########################################


# If these fitnesses not calculated right the best food sources wont be selected to be improvedc

# Did this so that you can just input the tuple rather than calling compute_f and index each time
def calc_fitness(source : list) -> float:
    return compute_f(source[0],source[1],source[2])


# This is the fitness funciton in lectures, however this has a flaw as the fitness decreases so does the total fitness 
# But the total_fitness dwarfs this change in comaprison and hence results in a worse fitness funciton to minimise
# Also as fitness decreases so does total_fitness and this would not be the bestr
def calc_fitness_population_lectures(population : list) -> list:
    '''
    This is to output a list of fitnessses for the population of different points generated 
    '''
    # The array for probabilities 
    probabilities = []

    # Calculaing the total fitness for the population
    total_fitness = 0

    # Iterating through the population
    for i in range(len(population)):
        # The local fitness calculated here so doesnt have to be recalculted again in this loop
        local_fitness = calc_fitness(population[i])

        # Add this to the total fitness
        total_fitness += local_fitness

        # Also add this local fitness to a list of probabilities which would then be divieded by the 
        probabilities.append(local_fitness)

    return  [x / total_fitness for x in probabilities]


# 2nd Improvement
# Calcuating the fitness of each food source in the population and then outputting a list
# This is a better fitness function in my opinion
def calc_fitness_population(population : list) -> list:
    '''
    This is to output a list of fitnessses for the population of different points generated 
    '''
    # The array for probabilities 
    probabilities = []

    # Iterating through the samples in the population
    for i in range(len(population)):

        # Computing the fitness hoever sometimes it might get to zero
        if compute_f(population[i][0],population[i][1],population[i][2]) == 0 :

            # If the fitness is 0 then the minimum of funciton reached so make the fitness infinite 
            # AS we have minimised the function and maximised the fitness here
            each_source = math.inf
        
        # Otherwise
        else:
            
            # Just compute the fitness 
            each_source = 1 / compute_f(population[i][0],population[i][1],population[i][2])
        
        # Append it to the list which is then used by the roultter wheel
        probabilities.append(each_source)

    return probabilities


########################################### Near Neighbour Algorithms for bee colony ###########################################

def bee():

    # Generating the population of 3-d points in space within the range [-500, 500]
    population = gen_pop()

    # Initlising the epoch
    t = 0

    # Making a list of N food sources so we know which one to abandon
    limit = [0] * N

    # Keep track of the global best solution
    global_best = math.inf

    # Keep track of the associated point with that global best solution
    minimum = []

    # Keeping a track of array for the fitness for each sample in the population
    fitness_population = [0] * N

    # Starting the epoch
    while t <= num_cyc:

        # This here so exit code before 60 seconds 
        func_time = time.time()

        # Just to terminate before 60 seconds
        time_elapsed = round(func_time - start_time, 1)
        if time_elapsed >= 55:
            return minimum,float(global_best)

        # Firstly employee bees
        for i in range(0,N+M):
            
            # Employee bees
            if i <= N - 1:
                k = i

            # Onlooker bees
            else : 
                
                # Here to test the lecture fitness fucntion replace calc_fitness_population to calc_fitness_population_lectures
                # Calculate the fitness population as needed to choose the solution to improve
                fitness_population = calc_fitness_population(population=population)

                # Roulette wheel to choose the food source for the onlooker bees
                k = random.choices(list(range(N)), weights=fitness_population)[0]

            
            # The old source or food source which is then going ot be explored around
            old_source = population[k] # old one

            # Choosing a random food source
            j = random.randint(a=0,b=N-1)
            
            # Making sure this food source is not the same one 
            while k == j:
                j = random.randint(a=0,b=N-1)

            # Now initlise the new source
            new_source = population[j]

            # Call the near neighbour algorithm here
            # 2nd Change to revert back to lecture near_neighbour just change experimentation boolean to False
            # To test the experimentation turn to True
            new_source_tring = near_neighbour(source1=old_source, source2=new_source, epoch = t, experientaiton=False) # old one 

            # Checking if the fitness of this new one is better than the old source
            if calc_fitness(new_source_tring) < calc_fitness(old_source):

                # If generated a better solution reset the limit for this sequence
                limit[k] = 0

                # Now update the k index in this population with the better points
                population[k][:] = new_source_tring

                # Now calculate the local_fitness to compare with the global solution
                local_fitness = calc_fitness(new_source_tring)

                # Sometimes we get the best poossible solution so not to divide by 0
                if local_fitness == 0:
                    getting_to_zero = time.time()

                    # Just to terminate before 60 seconds
                    time_elapsed_to_get_zero = round(getting_to_zero - start_time, 1)
                    # This measn funciton has been minimised to least and fitness to maximum
                    fitness_population[k] = math.inf

                # If not reached 0 no problem just process as normal
                else:
                    fitness_population[k] = 1 / local_fitness

                # Now make sure compare this local fitness with the global best solution to #
                if local_fitness < global_best:

                    # If entered that measn you are improivng the best solution
                    global_best = local_fitness

                    # Change the best possible points as well
                    minimum = population[k].copy()
            
            # If no better solution generated
            else:

                # Increase the limit for this particular food source so it might be abandoned
                limit[k] += 1

        # Now go through this limit array and make sure none of the food sources are exceddin this limit 
        for i in range(N):

            # If exceeded this limit then abandon this food source
            if limit[i] > lamb:

                # Generate a new sequence for this 
                population[i][:] = gen_food_source()

                # Reset the limit then
                limit[i] = 0
        
        # Add one epoch
        t += 1
    
    # Now return the best possible solution that were available
    return minimum,float(global_best)

minimum, min_f = bee()


#############################################################################################################
################################## DO NOT TOUCH ANYTHING BELOW THIS COMMENT #################################
#############################################################################################################

# You should now have computed your minimum value for the function f in the variable 'min_f' and the reserved
# variable 'minimum' should hold a list containing the values a, b and c for which function f is such that
# f(a, b, c) achieves its minimum; that is, your minimum point.

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)
    
error_flag = False
if type(min_f) != float and type(min_f) != int:
    print("\n*** error: you don't have a real-valued variable 'min_f'")
    error_flag = True
if type(minimum) != list:
    print("\n*** error: you don't have a tuple 'minimum' giving the minimum point")
    error_flag = True
elif len(minimum) != 3:
    print("\n*** error: you don't have a 3-tuple 'minimum' giving the minimum point; you have a {0}-tuple".format(len(minimum)))
    error_flag = True
else:
    var_names = ['a', 'b', 'c']
    for i in range(0, 3):
        if type(minimum[i]) != float and type(minimum[i]) != int:
            print("\n*** error: the value for {0} in your minimum point (a, b, c) is not numeric".format(var_names[i]))
            error_flag = True

if error_flag == False:
    print("\nYou have found a minimum value of {0} and a minimum point of [{1}, {2}, {3}].".format(min_f, minimum[0], minimum[1], minimum[2]))
    print("Your elapsed time was {0} seconds.".format(elapsed_time))


    

















    
