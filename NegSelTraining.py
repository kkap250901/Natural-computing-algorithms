#############################################################################################################
############################## DO NOT TOUCH ANYTHING UNTIL I TELL YOU TO DO SO ##############################
#############################################################################################################

# This is the skeleton program NegSelTraining.py around which you should build your implementation.
#
# The training set should be in a file self_training.txt.
#
# The output is a detector set that is in the file detector_<timestamp>.txt where <timestamp> is a timestamp
# so that you do not overwrite previously produced detector sets. You can always rename these files.
#
# It is assumed that NegSelTraining.py and self_training.txt are in the same folder.

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "ccbd24"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "VD"

#####################################################################################################################
#### ENTER THE THRESHOLD: IF YOU ARE IMPLEMENTING VDETECTOR THEN SET THE THRESHOLD AS YOUR CHOICE OF SELF-RADIUS ####
#####################################################################################################################

threshold = 0.0081

######################################################
#### ENTER THE INTENDED SIZE OF YOUR DETECTOR SET ####
######################################################

num_detectors = 200

#############################################################################################################
############################## DO NOT TOUCH ANYTHING UNTIL I TELL YOU TO DO SO ##############################
#############################################################################################################

import time
import os.path
import random
import math
import sys
import numpy

def get_a_timestamp_for_an_output_file():
    local_time = time.asctime(time.localtime(time.time()))
    timestamp = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
    timestamp = timestamp.replace(" ", "0") 
    return timestamp

def read_points(f, point_length, num_points, file):
    list_of_points = []
    count = 0
    error = ""
    the_line = f.readline()
    while the_line != "":
        points = the_line.split("[")
        points.pop(0)
        how_many = len(points)
        for i in range(0, how_many):
            if points[i][len(points[i]) - 1] == ",":
                points[i] = points[i][0:len(points[i]) - 2]
            elif points[i][len(points[i]) - 1] == "\n":
                points[i] = points[i][0:len(points[i]) - 3]
            else:
                points[i] = points[i][0:len(points[i]) - 1]
            split_point = points[i].split(",")
            if len(split_point) != point_length:
                error = "\n*** error: a point number has the wrong length\n"
                return list_of_points, error
            numeric_point = []
            for j in range(0, point_length):
                numeric_point.append(float(split_point[j]))
            list_of_points.append(numeric_point[:])
            count = count + 1
        the_line = f.readline()
    if count != num_points:
        error = "\n*** error: there should be " + str(num_points) + " points in " + file + " but there are " + str(count) + "\n"
    return list_of_points, error
 
self_training = "self_training.txt"

if not os.path.exists(self_training):
    print("\n*** error: " + self_training + " does not exist\n")
    sys.exit()

f = open(self_training, "r")

self_or_non_self = f.readline()
if self_or_non_self != "Self\n":
    print("\n*** error: the file " + self_training + " is not denoted as a Self-file\n")
    f.close()
    sys.exit()
dim = f.readline()
length_of_dim = len(dim)
dim = dim[len("dimension = "):length_of_dim - 1]
n = int(dim)
num_points = f.readline()
length_of_num_points = len(num_points)
num_points = num_points[len("number of points = "):length_of_num_points - 1]
Self_num_points = int(num_points)

list_of_points, error = read_points(f, n, Self_num_points, self_training)
Self = list_of_points[:]

f.close()

if error != "":
    print(error)
    sys.exit()

# The training data has now been read into the following reserved variables:
#   - 'n' = the dimension of the points in the training set
#   - 'Self_num_points' = the number of points in the training set
#   - 'Self' = the list of points in the training set
# These are reserved variables and their names should not be changed.

# You also have the reserved variables 'user_name', 'alg_code', 'threshold' and 'num_detectors'.
# Remember: if 'alg_code' = 'VD' then 'threshold' denotes your chosen self-radius.

# You need to initialize any other parameters (if you are implementing 'Real-valued Negative Selection'
# or 'VDetector') yourself in your code below.

# The list of detectors needs to be stored in the variable 'detectors'. This is a reserved variable
# and is initialized below. You need to ensure that your computed detector set is stored in 'detectors'
# as a list of points, i.e., as a list of lists-of-reals-of-length-'n' for NS and RV and 'n' + 1 for VD
# (remember: a detector for VD is a point plus its individual radius - see Lecture 4).

detectors = []

start_time = time.time()

#############################################################################################################
########################################### ENTER YOUR CODE BELOW ###########################################
#############################################################################################################

# Asssigning some reserved variables to new ones as the variable names are more readable
c_0 = 0.999

c_1 = 0.99999

# Renaming the threshold as self_radius to maintain consistency with lectures 
self_radius = threshold

# Again renaming the n as dimensions for better name of functions
dimensions = n


def gen_random_indvitual() -> numpy.array:
    """
    This funciton is to generate the detector set 
    """
    return numpy.array([random.uniform(0,1),random.uniform(0,1)])


#!todo possibly use different type of distance measurements 
def distance(a : numpy.array, b : numpy.array) -> float:
    """
    This is to get the eucledian distance between 2 points
    """
    temp_arr = a - b
    return numpy.sqrt(numpy.dot(temp_arr.T,temp_arr))


def vdetector() -> list:
    '''
    This is the V detector function from the lectures
    
    '''
    t_1 = 0
    while len(detectors) < num_detectors:
        # Return the detector set when the time elapsed is greater than 9 seconds 
        func_time = time.time()
        time_elapsed = round(func_time - start_time, 1)
        if time_elapsed >= 9:
            return detectors

        t_0 = 0
        phase_one_flag = 'failed'
        while phase_one_flag == 'failed':
            x = gen_random_indvitual()
            r = numpy.inf
            phase_one_flag = 'successful'
            
            for detector in detectors:
                if distance(x, numpy.array(detector[:dimensions])) <= detector[dimensions]:
                    t_0 += 1
                    if t_0 >= 1 / (1 - c_0):
                        return detectors
                    else: 
                        phase_one_flag = 'failed'
                        break
                    
        if phase_one_flag == 'successful':
            for self in Self:
                dis_x_d = distance(x, numpy.array(self))
                if  dis_x_d - self_radius < r:
                    r = dis_x_d - self_radius


            if r > self_radius: 
                detector_x = list(x)
                new_radius = float(r)
                detector_x.append(new_radius)
                detectors.append(detector_x)
            else:
                t_1 += 1

            if t_1 >= 1/(1-c_1):
                return detectors

    return detectors

vdetector()
#############################################################################################################
################################## DO NOT TOUCH ANYTHING BELOW THIS COMMENT #################################
#############################################################################################################

now_time = time.time()
training_time = round(now_time - start_time, 1)

timestamp = get_a_timestamp_for_an_output_file()
detector_set = "detector_" + timestamp + ".txt"

f = open(detector_set, "w")

f.write("username = {0}\n".format(username))
f.write("detector set\n")
f.write("algorithm = {0}\n".format(alg_code))
f.write("dimension = {0}\n".format(n))
if alg_code != "VD":
    f.write("threshold = {0}\n".format(threshold))
else:
    f.write("self-radius = {0}\n".format(threshold))
num_detectors = len(detectors)
f.write("number of points = {0}\n".format(num_detectors))
f.write("training time = {0}\n".format(training_time))
detector_length = n
if alg_code == "VD":
    detector_length = n + 1
for i in range(0, num_detectors):
    f.write("[")
    for j in range(0, detector_length):
        if j != detector_length - 1:
            f.write("{0},".format(detectors[i][j]))
        else:
            f.write("{0}]".format(detectors[i][j]))
            if i == num_detectors - 1:
                f.write("\n")
            else:
                f.write(",\n")
f.close()

















    
