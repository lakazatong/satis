class Binary:
	def __init__(self, n):
		self.n = n
		self._arr = [0] * n
		self.bit_count = 0

	def increment(self):
		# returns if it's 0 after the increment
		for i in range(self.n):
			self._arr[i] = not self._arr[i]
			if self._arr[i]:
				self.bit_count += 1
				return True
			self.bit_count -= 1
		return False

	def __iadd__(self, other):
		for _ in range(other - 1): self.increment()
		return self.increment()

	def __getitem__(self, index):
		return self._arr[index]

	def __setitem__(self, index, value):
		old_bit = self._arr[index]
		self._arr[index] = value
		self.bit_count += (value - old_bit) 

	def __iter__(self):
		return iter(self._arr)

	def __str__(self):
		return str(self._arr)