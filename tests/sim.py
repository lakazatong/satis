class Generator:
	def __init__(self, speed):
		self.speed = speed
		self.next = None

	def step(self):
		self.next.step(self.speed)

class Eater:
	def __init__(self, speed):
		self.speed = speed
		self.total = 0
		self.count = 0

	def step(self, speed):
		self.total += speed
		self.count += 1

class Splitter:
	def __init__(self, speed):
		self.speed = speed
		self.outputs = []
		self.index = 0
		self.leftovers = 0

	def step(self, speed):
		speed += self.leftovers
		self.leftovers = 0
		left = len(self.outputs)
		while left > 0:
			output = self.outputs[self.index]
			self.index = (self.index + 1) % len(self.outputs)
			if output.speed > speed:
				output.step(speed)
				break
			speed -= output.speed
			output.step(output.speed)
			left -= 1
		self.leftovers += speed

def main():
	g = Generator(600)
	s = Splitter(780)
	e1 = Eater(270)
	e2 = Eater(270)

	g.next = s
	s.outputs = [e1, e2, s]

	while True:
		g.step()
		print()
		print(e1.total / e1.count)
		print(e2.total / e2.count)

if __name__ == '__main__':
	main()