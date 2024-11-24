from fractions import Fraction

a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0
while float(c) != 36.0 or float(e) != 26.0:
	a = 62 + f
	b = Fraction(a,2)
	d = Fraction(b,2)
	g = Fraction(d,3)
	f = Fraction(d,3) + 2*Fraction(g,3)

	c = Fraction(a,2)
	e = Fraction(b,2) + Fraction(d,3) + Fraction(g,3)

print(float(b), float(d), float(g), float(f))