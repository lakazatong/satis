# handles like a regular list, rejects duplicates, has a contains method for fast lookup
class FastList(list):
	def __init__(self, *args):
		self._set = set()
		self.extend(args)

	def append(self, item):
		if self.contains(item): return
		super().append(item)
		self._set.add(item)

	def remove(self, item):
		super().remove(item)
		self._set.remove(item)

	def contains(self, item):
		return item in self._set

	def extend(self, iterable):
		for item in iterable: self.append(item)

	def clear(self):
		super().clear()
		self._set.clear()

	def pop(self, index=-1):
		item = super().pop(index)
		self._set.remove(item)
		return item