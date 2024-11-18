import os, threading, signal, time, traceback
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

class UniqueInMemoryHistory(InMemoryHistory):
	def append_string(self, string: str) -> None:
		string = string.replace('⧸', '/')
		if string in self._storage: self._storage.remove(string)
		if string in self._loaded_strings: self._loaded_strings.remove(string)
		self.store_string(string)
		self._loaded_strings.insert(0, string)

class CLI:
	def __init__(self, name, backend_class):
		self.name = name
		self.backend = backend_class()
		self.backend.running = False
		self.user_input = None
		self.running = threading.Event()
		self.input_history = UniqueInMemoryHistory()
		for cmd in self.load_recent_directories():
			self.input_history.append_string(cmd)
		signal.signal(signal.SIGINT, self.exit)

	def load_recent_directories(self):
		directories = [d for d in os.listdir('.') if os.path.isdir(d) and 'to' in d]
		directories.sort(key=lambda x: os.path.getmtime(x))
		return directories

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
				# self.user_input = input(f"\n{self.name}> ")
				self.user_input = prompt(f"\n{self.name}> ", history=self.input_history)
			except KeyboardInterrupt:
				# SIGINT received while in input
				break
			except EOFError:
				# idk
				break
			if self.user_input in ["exit", "quit", "q"]:
				break
			self.main()
			self.user_input = None
	
		self.backend.close()
		self.running.clear()