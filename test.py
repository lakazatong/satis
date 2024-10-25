import sys, os, pathlib
dirpath = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(dirpath)
sys.path.append(os.path.join(dirpath, 'src'))
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

import cProfile, random, time
from bisect import insort
from fastList import FastList
from config import config

value = 45
conveyor_speed = next(c for c in config.conveyor_speeds if c > value)
print(conveyor_speed)

# from tests.test_distance import test_distance

# cProfile.run("test_distance()")
# test_distance()