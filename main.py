import sys, os, pathlib
dirpath = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(dirpath)
sys.path.append(os.path.join(dirpath, 'src'))
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

import time, cProfile

from src.satisSolver import SatisSolver
from src.CLI import CLI

# An example class that could be used as backend for the CLI class
class Test:
	def __init__(self):
		pass

	def load(self, user_input):
		return True

	def stop(self):
		# called when SIGINT was catched
		self.running = False
		pass

	def run(self):
		# called if self.load returned True
		print("backend running...")
		for _ in range(30):
			if not self.running: break
			time.sleep(0.1)
		self.running = False

	def close(self):
		# called before the CLI closes
		pass

if __name__ == "__main__":
	cli = CLI("Satisfactory Solver", SatisSolver)
	# cli = CLI("Test", Test)
	# cProfile.run('cli.run()')
	cli.run()
