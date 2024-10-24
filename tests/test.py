import random, math, itertools, cProfile, time, ctypes, os
from utils import remove_pairs, insert_into_sorted
from collections import Counter
from fastList import FastList
import numpy as np

# user settings

allowed_divisors = [2, 3] # must be sorted
conveyor_speeds = [60, 120, 270, 480, 780, 1200] # must be sorted

allowed_divisors_r = allowed_divisors[::-1]
min_sum_count, max_sum_count = allowed_divisors[0], allowed_divisors_r[0]
conveyor_speeds_r = conveyor_speeds[::-1]
conveyor_speed_limit = conveyor_speeds_r[0]

