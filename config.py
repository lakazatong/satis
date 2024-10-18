import re

allowed_divisors = [2, 3] # must be sorted
conveyor_speeds = [60, 120, 270, 480, 780, 1200] # must be sorted

logging = False
log_filename = "logs.txt"

short_repr = True
include_depth_informations = False

solutions_filename = lambda i: f"solution{i}"
solution_regex = re.compile(r'solution\d+\.png') # ext is always png