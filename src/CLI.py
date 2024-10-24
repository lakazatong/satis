import threading, signal, time, traceback

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