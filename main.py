import sys, os, pathlib
dirpath = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(dirpath, "src")
if dirpath not in sys.path: sys.path.insert(0, dirpath)
if src_path not in sys.path: sys.path.insert(0, src_path)
if sys.platform == "win32":
	path = pathlib.Path(r"C:\Program Files\Graphviz\bin")
	if path.is_dir() and str(path) not in os.environ["PATH"]:
		os.environ["PATH"] += f";{path}"

# import cProfile

from src.solver import SatisSolver
from src.utils.cli import CLI

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
		import time
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
