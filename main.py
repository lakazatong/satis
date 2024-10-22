import threading, signal, time, traceback, cProfile

from satisSolver import SatisSolver
from config import config
from node import Node

class CLI:
	def __init__(self, name, backend_class):
		self.name = name
		self.backend = backend_class()
		self.backend.running = False
		self.user_input = None
		self.running = threading.Event()
		signal.signal(signal.SIGINT, self.exit)

	def main(self):
		def catching_run():
			try:
				self.backend.run()
			except:
				self.backend.running = False
				print(traceback.format_exc(), end="")
		
		if self.backend.load(self.user_input):
			backend_thread = threading.Thread(target=catching_run, daemon=True)
			self.backend.running = True
			backend_thread.start()
			
			# keep this thread alive to catch ctrl + c
			try:
				while self.backend.running: time.sleep(0.25)
			except KeyboardInterrupt:
				pass
			
			backend_thread.join()

	def exit(self, signum, frame):
		if self.backend.running:
			self.backend.stop()
		else:
			self.running.clear()

	def run(self):
		self.running.set()
		while self.running.is_set():
			try:
				self.user_input = input(f"\n{self.name}> ")
			except EOFError:
				# SIGINT received while in input
				break
			if self.user_input in ["exit", "quit", "q"]:
				break

			self.main()
			self.user_input = None
	
		self.backend.close()
		self.running.clear()

class Test:
	def __init__(self):
		pass

	def load(self, user_input):
		return True

	def stop(self):
		# a callback that is called when SIGINT was catched
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

def test():
	# from simsManager import SimsManager 
	# simsManager = SimsManager()
	# r = simsManager.get_extract_sims((5, 650))
	# print(r)
	# node655 = Node(655)

	# node650 = Node(650)
	# node5 = Node(5)

	# node325_1 = Node(325)
	# node325_2 = Node(325)

	# node330 = Node(330)

	# node120 = Node(120)
	# node205 = Node(205)
	
	# node450 = Node(450)

	# node150_1 = Node(150)
	# node150_2 = Node(150)
	# node150_3 = Node(150)

	# node655.children = [node650, node5]
	# node650.children = [node325_1, node325_2]
	# node5.children = [node330]
	# node325_1.children = [node330]
	# node325_2.children = [node205, node120]
	# node330.children = [node450]
	# node120.children = [node450]
	# node450.children = [node150_1, node150_2, node150_3]

	# tmp = str(node655)
	# tmp = re.sub("parents=\\[\\]", "parents=[.*]", tmp)
	# for c in "()[]":
	# 	tmp = tmp.replace(c, "\\" + c)
	# tmp = re.sub("short_node_id=(.*?),", "short_node_id=.*,", tmp)
	# pattern = re.sub("value=(.*?),", "value=.*,", tmp)
	
	# content = None
	# with open('logs.txt', 'r') as file:
	# 	content = file.read()

	# results = re.findall(pattern, content)

	# print(results)
	# cProfile.run('solve([475, 85, 100], [45, 55, 100])')
	# cProfile.run('solve([5, 650], [150, 150, 150, 205])')
	# cProfile.run('solve([40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50], [420, 420])')
	pass

if __name__ == "__main__":
	# test()
	# exit(0)
	cli = CLI("Satisfactory Solver", SatisSolver)
	# cli = CLI("Test", Test)
	# cProfile.run('cli.run()')
	cli.run()
