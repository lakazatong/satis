import sys, os, pathlib
dirpath = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(dirpath)
sys.path.append(os.path.join(dirpath, 'src'))
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

import cProfile

from tests.test_distance import test_distance

cProfile.run("test_distance()")
# test_distance()