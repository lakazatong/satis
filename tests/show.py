from fractions import Fraction

a, b, c, d, e, f, g, h = 0, 0, 0, 0, 0, 0, 0, 0
while float(d) != 2.0 or float(e) != 2.0 or float(f) != 2.0 or float(g) != 2.0 or float(h) != 2.0:
	a = 10 + Fraction(b,3)
	b = Fraction(a,2)
	c = Fraction(a,2)
	d = Fraction(b,3)
	e = Fraction(b,3)
	f = g = h = Fraction(c,3)

print(float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h))