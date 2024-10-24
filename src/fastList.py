
# handles like a regular list, but has a contains method for fast lookup
class FastList(list):
	def __init__(self, *args):
		super().__init__(args)
		self._set = set(self)

	def append(self, item):
		super().append(item)
		self._set.add(item)

	def remove(self, item):
		super().remove(item)
		self._set.remove(item)

	def contains(self, item):
		return item in self._set

	def extend(self, iterable):
		super().extend(iterable)
		self._set.update(iterable)

	def clear(self):
		super().clear()
		self._set.clear()

	def pop(self, index=-1):
		item = super().pop(index)
		self._set.remove(item)
		return item