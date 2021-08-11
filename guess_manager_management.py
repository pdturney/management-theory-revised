#
# Guess Manager Management
#
# Peter Turney, August 10, 2021
#
# From the 20 runs, extract all of the pickled seeds with
# two, three, or four parts. Try to guess which part is
# the manager by running many one-on-one competitions
# between two parts. The part that wins the most competitions
# is the best guess for the manager. Score is based on green
# versus orange.
#
import golly as g
import model_classes as mclass
import model_functions as mfunc
import model_parameters as mparam
import numpy as np
import scipy.stats as st
import copy
import time
import pickle
import os
import re
import sys
#
# Parameter values for the experiments.
#
num_files = 20                 # number of folders of runs of Model-S
possible_num_parts = [2, 3, 4] # possible numbers of parts in the seed
#
# Location of fusion_storage.bin files -- the input pickles.
#
fusion_dir = "C:/Users/peter/Peter's Projects" + \
             "/management-theory-revised/Experiments"
# list of pickle files
fusion_files = []
# loop through the fusion files and record the file paths
# -- we assume the folders have the form "run1", "run2", ...
for i in range(num_files):
  fusion_files.append(fusion_dir + "/run" + str(i + 1) + \
    "/fusion_storage.bin")
#
# Loop through the pickles, loading them into fusion_list.
# Each fusion file will contain several pickles.
#
fusion_list = mfunc.read_fusion_pickles(fusion_files)
#
# Output file path.
#
results_file = fusion_dir + "/guess_manager_management.txt"
results_handle = open(results_file, "w")
#
# Process fusion_list in batches, where each batch has the
# same number of parts (see possible_num_parts above).
#
for current_num_parts in possible_num_parts:
  # note the number of parts in the current group of seeds
  results_handle.write("\n\n" + str(current_num_parts) + " parts in seed\n\n")
  # some variables for calculating conditional probabilities
  # - if there is exactly one manager, what is the probability that
  #   the manager has the most Immigration Game fitness, compared to the
  #   fitness of the workers? -- p(manager max fitness | exactly one manager)
  count_one_manager = 0
  count_one_manager_max_fitness = 0
  total_sample_size = 0
  # iterate through fusion_list skipping over cases that don't have
  # exactly part_num parts
  for seed in fusion_list:
    # get a map of the regions in the seed
    seed_map = mfunc.region_map(seed)
    num_regions = np.amax(seed_map)
    # make sure seed has current_num_parts parts
    if (num_regions != current_num_parts):
      continue
    # update sample size for current_num_parts parts
    total_sample_size += 1
    # extract the parts from the seed, converting each part
    # to a Game of Life pattern, composed entirely of zeros
    # and ones; the new part will be reduced in size to match
    # the size of the chosen region
    part_list = []
    for target_region in range(1, current_num_parts + 1):
      target_part = mfunc.extract_parts(seed, seed_map, target_region)
      target_part.num_living = target_part.count_ones()
      part_list.append(target_part)
    # measure the fitness of each part by one-on-one competitions
    # in the Immigration Game
    fitness_list = []
    num_trials = 500
    for i in range(current_num_parts):
      part1 = part_list[i]
      scores = []
      for j in range(current_num_parts):
        part2 = part_list[j]
        if (i == j):
          continue
        else:
          [score1, score2] = mfunc.score_management(g, part1, part2, \
            mparam.width_factor, mparam.height_factor, \
            mparam.time_factor, num_trials)
        scores.append(score1)
      average_fitness = sum(scores) / len(scores)
      fitness_list.append(average_fitness)
    # output fitness numbers from the Immigration Game
    fitness_list_string = ", ".join(map(str, fitness_list))
    results_handle.write("fitness of parts: " + fitness_list_string + "\n")
    # calculate the tensor using the Management Game
    seed_num    = 0 # only one seed
    seed_list   = [seed] # only one seed
    step_size   = 1000 # one giant step
    max_seeds   = 10 # more than we'll need here
    num_steps   = 1001 # number of time steps, from 0 to 1000
    num_colours = 5 # white, red, blue, orange, green
    num_parts   = current_num_parts # 2, 3, or 4 parts
    step_num    = 1000 # this is the number of the final step
    [tensor, num_seeds] = mfunc.growth_tensor(g, seed_list, step_size, \
      max_seeds, num_steps, num_colours, num_parts)
    # classify each part as a manager or worker
    manager_list = []
    for part_num in range(num_parts):
      # extract colours
      red    = tensor[seed_num, step_num, 1, part_num]
      blue   = tensor[seed_num, step_num, 2, part_num]
      orange = tensor[seed_num, step_num, 3, part_num]
      green  = tensor[seed_num, step_num, 4, part_num]
      # we focus on the current part (part_num) only
      # -- the current part is always red, by convention
      red_manager = int(orange > green) # true or false -> 1 or 0
      manager_list.append(red_manager)
      #
    # output management status from the Management Game  
    manager_list_string = ", ".join(map(str, manager_list))
    results_handle.write("manager status: " + manager_list_string + "\n\n")
    # some variables for calculating conditional probabilities
    # - if there is exactly one manager, what is the probability that
    #   the manager has the most Immigration Game fitness, compared to the
    #   fitness of the workers? -- p(manager max fitness | exactly one manager)
    if (sum(manager_list) == 1): # if exactly one manager
      count_one_manager += 1
      manager_fitness = max(np.multiply(manager_list, fitness_list))
      sorted_fitness = sorted(fitness_list, reverse=True)
      # if the highest fitness matches the manager's fitness and the second
      # highest fitness does not match the manager's fitness ...
      if ((manager_fitness == sorted_fitness[0]) and \
          (manager_fitness != sorted_fitness[1])):
        count_one_manager_max_fitness += 1
    #
  #
  # print out conditional probabilities
  # - p(manager max fitness | exactly one manager)
  prob_manager_max_fitness_given_one_manager = \
    count_one_manager_max_fitness / count_one_manager
  #
  results_handle.write(
    "p(manager max fitness | exactly one manager) = " + \
    str(prob_manager_max_fitness_given_one_manager) + "\n" + \
    " = " + str(count_one_manager_max_fitness) + " / " + \
    str(count_one_manager) + "\n\n" + \
    "p(one specific part | " + str(current_num_parts) + \
    " parts to choose from) = " + str(1 / current_num_parts) + \
    "\n\ntotal sample size = " + str(total_sample_size) + "\n\n")
  #
#
results_handle.close()
#
#