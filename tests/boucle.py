# (5 + v(n-1))/6 = v(n)
def get_next(prev):
	return (prev + 5) / 6

cur = 5
for i in range(100):
	cur = get_next(cur)
	print(i, round(cur, 2))