#
# Fusion Pickles Probability 3 Parts
#
# Peter Turney, July 14, 2021
#
# From the 20 runs, extract all of the pickled three-part seeds
# that are stored in the 20 "fusion_storage.bin" pickle files.
# Read the pickles and run each pickle, recording the results in
# a numpy tensor:
#
# tensor = num_seeds x num_steps x num_colours x num_parts
#
# num_seeds   = to be determined
# num_steps   = 1001
# num_colours = 5 (white, red, orange, blue, green)
# num_parts   = 3
#
# After this tensor has been filled with values, generate
# a table of the form:
#
# <prob N M> = <probability for N managers and M workers>
#
# row in table = <step number> <prob 3 0> <prob 2 1> <prob 1 2> <prob 0 3>
#
import golly as g
import model_classes as mclass
import model_functions as mfunc
import model_parameters as mparam
import numpy as np
import copy
import time
import pickle
import os
import re
import sys
#
# Parameter values for making the graphs.
#
max_seeds   = 2000 # probably won't need more seeds than this
num_steps   = 1001 # number of time steps in the game
num_colours = 5    # 5 colours [white, red, blue, orange, green]
num_parts   = 3    # number of parts
num_files   = 20   # number of fusion pickle files
step_size   = 20   # number of time steps between each plot point
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
seed_list = mfunc.read_fusion_pickles(fusion_files)
#
# Given a list of seeds, fill a tensor with counts of the growth of colours
# generated by running the Management Game.
#
[tensor, num_seeds] = mfunc.growth_tensor(g, seed_list, step_size,
                      max_seeds, num_steps, num_colours, num_parts)
#
# now the tensor is full, so let's make the graph for 3 parts
#
graph_file = fusion_dir + "/fusion_pickles_probability_3.txt"
graph_handle = open(graph_file, "w")
graph_handle.write("\n\nNOTE: {} Seeds -- {} Parts per seed\n\n".format(
  num_seeds, num_parts))
header = ["step num", \
  "3 managers and 0 workers", \
  "2 managers and 1 worker", \
  "1 manager and 2 workers", \
  "0 managers and 3 workers"]
graph_handle.write("\t".join(header) + "\n")
#
for step_num in range(0, num_steps, step_size):
  # initialize counts
  count_3m0w = 0 # 3 managers, 0 workers
  count_2m1w = 0 # 2 managers, 1 worker
  count_1m2w = 0 # 1 manager,  2 workers
  count_0m3w = 0 # 0 managers, 3 workers
  # iterate over seed_num
  for seed_num in range(num_seeds):
    # iterate over parts
    manager_count = 0
    for part_num in range(num_parts):
      # extract colours
      red    = tensor[seed_num, step_num, 1, part_num]
      blue   = tensor[seed_num, step_num, 2, part_num]
      orange = tensor[seed_num, step_num, 3, part_num]
      green  = tensor[seed_num, step_num, 4, part_num]
      # we focus on the current part (part_num) only
      # -- the current part is always red, by convention
      red_manager = (orange > green) # true or false
      manager_count += red_manager # will increment by 0 or 1
    # increment counts
    if (manager_count == 3):
      count_3m0w += 1
    elif (manager_count == 2):
      count_2m1w += 1
    elif (manager_count == 1):
      count_1m2w += 1
    else:
      count_0m3w += 1
  #
  assert count_3m0w + count_2m1w + count_1m2w + count_0m3w == num_seeds
  #
  probability_3m0w = count_3m0w / num_seeds
  probability_2m1w = count_2m1w / num_seeds
  probability_1m2w = count_1m2w / num_seeds
  probability_0m3w = count_0m3w / num_seeds
  #
  graph_handle.write("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(step_num,
    probability_3m0w, probability_2m1w, probability_1m2w, probability_0m3w))
  #
#
graph_handle.close()
#
#