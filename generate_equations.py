import random
import sympy as sp
import json
import os

random.seed(1)

x, y, z, t, a, b, c, d, n, m, i, j, k = sp.symbols('x y z t a b c d n m i j k')
alpha, beta, gamma, delta, theta, phi, psi, omega = sp.symbols('alpha beta gamma delta theta phi psi omega')
f, g, h, F, G, H = sp.symbols('f g h F G H', cls=sp.Function)

# Helper to generate random integer
def ri(lo=1, hi=12):
    return random.randint(lo, hi)

def gen_polynomial():
    deg = ri(2, 5)
    coeffs = [ri(-10, 10) for _ in range(deg + 1)]
    coeffs[-1] = max(abs(coeffs[-1]), 1)
    expr = sum(c * x**i for i, c in enumerate(coeffs))
    rhs = ri(-20, 20)
    return sp.Eq(expr, rhs)

def gen_quadratic():
    r1, r2 = ri(-8, 8), ri(-8, 8)
    a_val = ri(1, 5)
    return sp.Eq(a_val * (x - r1) * (x - r2), 0)

def gen_factorial():
    n_val = ri(3, 8)
    return sp.Eq(sp.factorial(n_val), ri(6, 40320))

def gen_sum():
    n_sym = sp.Symbol('n')
    k_sym = sp.Symbol('k')
    start = ri(1, 3)
    end = ri(5, 10)
    expressions = [k_sym * x, k_sym**2, k_sym + x, 2**k_sym * x]
    expr = random.choice(expressions)
    return sp.Sum(expr, (k_sym, start, end))

def gen_product():
    k_sym = sp.Symbol('k')
    expr = (k_sym + 1) * x
    return sp.Product(expr, (k_sym, 1, ri(4, 8)))

def gen_log():
    base = ri(2, 5)
    arg = random.choice([x**2, sp.sqrt(x), x * y, x + y])
    return sp.Eq(sp.log(arg, base), ri(-3, 5))

def gen_exp():
    return sp.Eq(sp.exp(x), ri(1, 100))

def gen_root():
    n_val = random.choice([2, 3, 4])
    expr = sp.root(x * ri(1, 10), n_val)
    return sp.Eq(expr, ri(-10, 10))

def gen_binomial():
    n_val, k_val = ri(5, 12), ri(1, 5)
    return sp.Eq(sp.binomial(n_val, k_val), ri(1, 792))

def gen_rational():
    num = ri(-10, 10) * x + ri(-5, 5)
    den = x + ri(1, 5)
    return sp.Eq(num / den, ri(-10, 10))

def gen_power():
    return sp.Eq((x + ri(1, 5))**ri(2, 4), ri(1, 1000))

def gen_abs():
    return sp.Eq(sp.Abs(x - ri(-5, 5)), ri(1, 10))

def gen_piecewise_expr():
    return sp.Piecewise((x**2, sp.Ge(x, 0)), (-x, True))

def gen_complex():
    return sp.Eq((1 + sp.I * x)**2, ri(-5, 5) + sp.I * ri(-10, 10))

def gen_inequality():
    return sp.Ge(x**2 - ri(1, 5) * x + ri(-10, 10), 0)

def gen_trig():
    funcs = [sp.sin, sp.cos, sp.tan, sp.asin, sp.acos]
    arg = random.choice([x, x/2, 2*x, x + y])
    func = random.choice(funcs)
    return sp.Eq(func(arg), random.choice([0, 1, -1, sp.Rational(1, 2), sp.sqrt(2)/2]))

def gen_hyperbolic():
    return sp.Eq(sp.sinh(x), ri(-5, 5))

def gen_derivative():
    exprs = [x**3, x**4 * sp.sin(x), sp.exp(x**2), sp.log(x)**2, x * sp.sin(x)]
    expr = random.choice(exprs)
    return sp.diff(expr, x)

def gen_integral():
    exprs = [x**3, sp.sin(x), sp.exp(x), x * sp.log(x), sp.cos(x)**2]
    expr = random.choice(exprs)
    return sp.Integral(expr, x)

def gen_definite_integral():
    exprs = [x**2, sp.sin(x), sp.exp(-x)]
    expr = random.choice(exprs)
    a_val, b_val = ri(0, 3), ri(4, 10)
    return sp.Integral(expr, (x, a_val, b_val))

def gen_limit():
    exprs = [(1 + x/n)**n, sp.sin(x)/x, (sp.sqrt(x+1) - sp.sqrt(x)), (1/x)**(1/x)]
    expr = random.choice(exprs)
    return sp.Limit(expr, n if '(1 + x/n)' in str(expr) else x, sp.oo if 'sqrt' in str(expr) else 0)

def gen_series():
    return sp.Sum(x**n / sp.factorial(n), (n, 0, sp.oo))

def gen_matrix_det():
    A = sp.Matrix([[ri(-3, 3) for _ in range(2)] for _ in range(2)])
    return sp.Eq(sp.det(A), ri(-10, 10))

def gen_cramer():
    return sp.Eq(x, ri(1, 10))  # placeholder

def gen_matrix_solve():
    A = sp.Matrix([[2, 1], [1, 3]])
    b = sp.Matrix([ri(1, 10), ri(1, 10)])
    return sp.Eq(A * sp.Matrix([x, y]), b)

def gen_matrix_mult():
    A = sp.Matrix([[ri(-3, 3) for _ in range(3)] for _ in range(3)])
    B = sp.Matrix([[ri(-3, 3) for _ in range(3)] for _ in range(3)])
    return A * B

def gen_matrix_power():
    A = sp.Matrix([[ri(-2, 2) for _ in range(2)] for _ in range(2)])
    return A**ri(2, 3)

def gen_matrix_inverse():
    while True:
        A = sp.Matrix([[ri(-3, 3) for _ in range(3)] for _ in range(3)])
        if A.det() != 0:
            break
    return sp.Inverse(A)

def gen_matrix_eigen():
    A = sp.Matrix([[ri(-2, 2) for _ in range(2)] for _ in range(2)])
    return A.eigenvals()

def gen_matrix_transpose():
    A = sp.Matrix([[ri(-3, 3) for _ in range(2)] for _ in range(3)])
    return A.T

def gen_matrix_rank():
    A = sp.Matrix([[ri(-2, 2) for _ in range(3)] for _ in range(2)])
    return sp.Eq(A.rank(), ri(1, 2))

def gen_matrix_system():
    A = sp.Matrix([[ri(-3, 3) for _ in range(3)] for _ in range(3)])
    b = sp.Matrix([ri(-5, 5) for _ in range(3)])
    return sp.Eq(A * sp.Matrix([x, y, z]), b)

def gen_matrix_augmented():
    A = sp.Matrix([[ri(-3, 3) for _ in range(4)] for _ in range(2)])
    return A

def gen_matrix_sylvester():
    A = sp.Matrix([[ri(-2, 2) for _ in range(2)] for _ in range(2)])
    B = sp.Matrix([[ri(-2, 2) for _ in range(2)] for _ in range(2)])
    X = sp.Matrix([[sp.Symbol(f'x{i}{j}') for j in range(2)] for i in range(2)])
    return sp.Eq(A * X + X * B, sp.zeros(2, 2))

def gen_matrix_block():
    O = sp.zeros(2, 2)
    I = sp.eye(2)
    return sp.Matrix.hstack(I, O, I)

def gen_matrix_orthogonal():
    M1 = sp.Matrix([[sp.cos(x), -sp.sin(x)], [sp.sin(x), sp.cos(x)]])
    M2 = sp.Matrix([[sp.cos(x), sp.sin(x)], [-sp.sin(x), sp.cos(x)]])
    return sp.Eq(M1 * M2, sp.eye(2))

def gen_matrix_orthogonality():
    v1 = sp.Matrix([1, 1, 0])
    v2 = sp.Matrix([1, -1, 0])
    return sp.Eq(v1.dot(v2), 0)

def gen_chain_rule():
    outer = random.choice([sp.sin, sp.cos, sp.exp, sp.log])
    inner = x**2 + x + ri(1, 3)
    return sp.diff(outer(inner), x)

def gen_product_rule():
    f1 = x**ri(1, 3)
    f2 = sp.sin(x) if random.random() > 0.5 else sp.exp(x)
    return sp.diff(f1 * f2, x)

def gen_quotient_rule():
    num = x**ri(1, 2) + ri(1, 5)
    den = x + ri(1, 3)
    return sp.diff(num / den, x)

def gen_partial():
    return sp.diff(x**2 * y + x * y**2, x)

def gen_taylor():
    exprs = [sp.sin, sp.cos, sp.exp]
    expr = random.choice(exprs)
    return expr(x).series(x, n=ri(4, 8))

def gen_implicit():
    return sp.Eq(x**2 + y**2, ri(4, 25)**2)

def gen_ode_simple():
    return sp.Eq(sp.diff(y(x), x), x * y(x))

def gen_second_derivative():
    exprs = [x**4, x**3 * sp.sin(x), sp.exp(x) * sp.sin(x)]
    expr = random.choice(exprs)
    return sp.diff(expr, x, 2)

def gen_integration_by_parts():
    return sp.Integral(x * sp.exp(x), x)

def gen_vector_derivative():
    return sp.Matrix([x**2, sp.sin(x), sp.exp(x)]).diff(x)

def gen_gradient():
    return sp.Matrix([sp.diff(x**2 * y + x * y**2, var) for var in [x, y]])

def gen_jacobian():
    return sp.Matrix([[sp.diff(expr, var) for var in [x, y]] for expr in [x**2 * y, x * y**2]])

def gen_divergence():
    vec = sp.Matrix([x**2, y**2, z**2])
    return sp.diff(vec[0], x) + sp.diff(vec[1], y) + sp.diff(vec[2], z)

def gen_line_integral():
    return sp.Integral(sp.sin(x**2) * sp.exp(y), x)

def gen_double_integral():
    return sp.Integral(sp.Integral(x * y, (x, 0, 1)), (y, 0, 2))

def gen_word_problem():
    templates = [
        {'text': 'A car accelerates from rest with a constant acceleration of {a} m/s² for {t} seconds. It then continues at constant velocity for another {t2} seconds. What is the total distance traveled during this entire time period?',
         'equation': lambda: sp.Eq(sp.Symbol('d_total'), sp.Symbol('v') * sp.Symbol('t2') + sp.Rational(1, 2) * sp.Symbol('a') * sp.Symbol('t')**2)},

        {'text': 'You invest ${principal} in a savings account that offers an annual interest rate of {rate}%, compounded quarterly. After {years} years, what is the total amount in the account? Use the compound interest formula to find the final balance.',
         'equation': lambda: sp.Eq(sp.Symbol('A'), sp.Symbol('principal') * (1 + sp.Symbol('rate')/400)**(4 * sp.Symbol('years')))},

        {'text': 'A rectangular garden has a length of {l} meters and a width of {w} meters. If you want to build a fence around the entire garden, how much fencing material do you need? Also, what is the total area of the garden in square meters?',
         'equation': lambda: sp.Eq(sp.Symbol('P'), 2 * (sp.Symbol('l') + sp.Symbol('w')))},

        {'text': 'The height of a projectile is given by h(t) = -{c1}t² + {c2}t + {c3}, where t is time in seconds and h is height in meters. At what time t does the projectile reach its maximum height? Also find the maximum height.',
         'equation': lambda: sp.Eq(sp.Symbol('t_max'), sp.Symbol('c2') / (2 * sp.Symbol('c1')))},

        {'text': 'A farmer has {fence} meters of fencing material and wants to enclose a rectangular field against a long wall (so only three sides need fencing). What dimensions will maximize the enclosed area? Find the optimal length and width of the field.',
         'equation': lambda: sp.Eq(sp.Symbol('A'), sp.Symbol('l') * (sp.Symbol('fence')/2 - sp.Symbol('l')))},

        {'text': 'A spherical balloon is being inflated with air at a rate of {rate} cm³/s. When the radius of the balloon is {r} cm, how fast is the radius increasing? Use the volume formula V = (4/3)πr³ to find dr/dt at this instant.',
         'equation': lambda: sp.Eq(sp.Symbol('dr/dt'), sp.Symbol('rate') / (4 * sp.pi * sp.Symbol('r')**2))},

        {'text': 'The velocity of a particle moving along a straight line is given by v(t) = {v1}t² + {v2}t + {v3} m/s. If the particle starts at position x = 0 when t = 0, what is its position at time t = {t_end} seconds?',
         'equation': lambda: sp.Integral(sp.Symbol('v1') * t**2 + sp.Symbol('v2') * t + sp.Symbol('v3'), (t, 0, sp.Symbol('t_end')))},

        {'text': 'Solve the following system of three linear equations with three unknowns x, y, and z:\n{x} + 2{y} + {z} = {r1}\n3{x} - {y} + {z} = {r2}\n2{x} + 3{y} - 2{z} = {r3}\nExpress your answer as a vector [x, y, z].',
         'equation': lambda: sp.Matrix([[1, 2, 1], [3, -1, 1], [2, 3, -2]])},

        {'text': 'A ball of mass {m} kg is dropped from a height of {h} meters. Ignoring air resistance, what is the velocity of the ball just before it hits the ground? Use conservation of energy: mgh = (1/2)mv² to solve for v.',
         'equation': lambda: sp.Eq(sp.Symbol('v'), sp.sqrt(2 * sp.Symbol('g') * sp.Symbol('h')))},

        {'text': 'Given that y = sin({f1}x² + {f2}x + {f3}), find the derivative dy/dx using the chain rule. Then evaluate the derivative at x = 1 to find the instantaneous rate of change at that point.',
         'equation': lambda: sp.diff(sp.sin(sp.Symbol('f1') * x**2 + sp.Symbol('f2') * x + sp.Symbol('f3')), x)},

        {'text': 'A cylinder has a radius of {r} cm and a height of {h} cm. If the radius is increased by 10% and the height is decreased by 10%, what is the percentage change in volume? Use V = πr²h to compute both volumes.',
         'equation': lambda: sp.Eq(sp.Symbol('V'), sp.pi * sp.Symbol('r')**2 * sp.Symbol('h'))},

        {'text': 'Consider the curve defined by the equation x² + {c1}xy + {c2}y² = {c3}. Find dy/dx using implicit differentiation. Evaluate the derivative at the point (1, 1) to find the slope of the tangent line.',
         'equation': lambda: sp.Eq(sp.Symbol('dy/dx'), - (2*sp.Symbol('x') + sp.Symbol('c1')*sp.Symbol('y')) / (sp.Symbol('c1')*sp.Symbol('x') + 2*sp.Symbol('c2')*sp.Symbol('y')))},

        {'text': 'A fair six-sided die is rolled {n} times. What is the probability of getting exactly {k} sixes? Use the binomial probability formula P(X=k) = C({n},k) × (1/6)^k × (5/6)^({n}-k) to compute the answer.',
         'equation': lambda: sp.binomial(sp.Symbol('n'), sp.Symbol('k')) * (sp.Rational(1, 6))**sp.Symbol('k') * (sp.Rational(5, 6))**(sp.Symbol('n') - sp.Symbol('k'))},

        {'text': 'A wave is described by the equation y(x,t) = {A}sin({k}x - {w}t), where y is displacement in meters, x is position in meters, t is time in seconds, and A, k, ω are constants. What is the wave speed v = ω/k?',
         'equation': lambda: sp.Eq(sp.Symbol('v'), sp.Symbol('w') / sp.Symbol('k'))},

        {'text': 'Find the Taylor series expansion of f(x) = e^x centered at x = 0 up to the term containing x^{n}. Then use this series to approximate e^1 (i.e., Euler\'s number) by summing the first {terms} terms.',
         'equation': lambda: sp.Sum(x**sp.Symbol('n') / sp.factorial(sp.Symbol('n')), (sp.Symbol('n'), 0, sp.Symbol('terms')))},

        {'text': 'Given the 2×2 matrix A = [[{a1}, {a2}], [{a3}, {a4}]], find the eigenvalues λ by solving the characteristic equation det(A - λI) = 0. Also find the corresponding eigenvectors for each eigenvalue.',
         'equation': lambda: sp.Matrix([[sp.Symbol('a1'), sp.Symbol('a2')], [sp.Symbol('a3'), sp.Symbol('a4')]]).eigenvals()},

        {'text': 'A projectile is launched from ground level with an initial velocity of {v0} m/s at an angle of {angle}° above the horizontal. The equations of motion are x(t) = v₀cos(θ)t and y(t) = v₀sin(θ)t - (1/2)gt². Find the time of flight (when y = 0 again) and the horizontal range.',
         'equation': lambda: sp.Eq(sp.Symbol('T_flight'), 2 * sp.Symbol('v0') * sp.sin(sp.Symbol('angle') * sp.pi/180) / sp.Symbol('g'))},

        {'text': 'Evaluate the definite integral ∫₀¹ xⁿ e^x dx using integration by parts. Show that this equals {c1} - n∫₀¹ xⁿ⁻¹ e^x dx, and use this recurrence relation to compute the value for n = {n_val}.',
         'equation': lambda: sp.Eq(sp.Integral(sp.Symbol('x')**sp.Symbol('n') * sp.exp(sp.Symbol('x')), (sp.Symbol('x'), 0, 1)), sp.Symbol('c1') - sp.Symbol('n') * sp.Integral(sp.Symbol('x')**(sp.Symbol('n')-1) * sp.exp(sp.Symbol('x')), (sp.Symbol('x'), 0, 1)))},

        {'text': 'Two similar triangles have corresponding side lengths in the ratio 2:3. If the area of the smaller triangle is {area1} square units, what is the area of the larger triangle? Explain why the ratio of areas equals the square of the ratio of side lengths.',
         'equation': lambda: sp.Eq(sp.Symbol('area2'), sp.Symbol('area1') * (sp.Rational(3, 2))**2)},

        {'text': 'A box with an open top is to be constructed from a rectangular piece of cardboard measuring {l} cm by {w} cm. Squares of side length x cm are cut from each corner, and the sides are folded up. Find the value of x that maximizes the volume of the box.',
         'equation': lambda: sp.Eq(sp.Symbol('V'), (sp.Symbol('l') - 2*sp.Symbol('x')) * (sp.Symbol('w') - 2*sp.Symbol('x')) * sp.Symbol('x'))},

        {'text': 'A circuit has two resistors R₁ = {r1} Ω and R₂ = {r2} Ω connected in parallel. This parallel combination is then connected in series with a third resistor R₃ = {r3} Ω. Find the total equivalent resistance of the circuit.',
         'equation': lambda: sp.Eq(sp.Symbol('R_eq'), sp.Symbol('r3') + 1 / (1/sp.Symbol('r1') + 1/sp.Symbol('r2')))},

        {'text': 'Given the function z = f(x,y) = {f1}x² + {f2}xy + {f3}y² + {f4}x + {f5}y, find the partial derivatives ∂z/∂x and ∂z/∂y. Then evaluate both partial derivatives at the point (x, y) = (1, 1).',
         'equation': lambda: sp.diff(sp.Symbol('f1')*x**2 + sp.Symbol('f2')*x*y + sp.Symbol('f3')*y**2 + sp.Symbol('f4')*x + sp.Symbol('f5')*y, x)},

        {'text': 'Given matrices A (2×2) and B (2×2) where A = [[{a11}, {a12}], [{a21}, {a22}]] and B = [[{b11}, {b12}], [{b21}, {b22}]], compute the product AB. Then verify whether AB = BA (commutativity check).',
         'equation': lambda: sp.Matrix([[sp.Symbol('a11'), sp.Symbol('a12')], [sp.Symbol('a21'), sp.Symbol('a22')]]) * sp.Matrix([[sp.Symbol('b11'), sp.Symbol('b12')], [sp.Symbol('b21'), sp.Symbol('b22')]])},

        {'text': 'Evaluate the limit lim(x→0) sin({k}x) / x using L\'Hôpital\'s rule. Since both numerator and denominator approach 0, differentiate both with respect to x and evaluate the resulting limit.',
         'equation': lambda: sp.limit(sp.sin(sp.Symbol('k')*x) / x, x, 0)},

        {'text': 'A projectile is launched with initial speed {v0} m/s at an angle θ above the horizontal. The horizontal and vertical components of velocity are v_x = v₀cos(θ) and v_y = v₀sin(θ) - gt. Find the maximum height reached and the time to reach it (when v_y = 0).',
         'equation': lambda: sp.Eq(sp.Symbol('H_max'), (sp.Symbol('v0') * sp.sin(sp.Symbol('theta')))**2 / (2 * sp.Symbol('g')))},

        {'text': 'Given the vector field F(x,y,z) = {f1}xi + {f2}yj + {f3}zk, compute the divergence ∇·F and the curl ∇×F. The divergence measures how much the field spreads out, while the curl measures rotation.',
         'equation': lambda: sp.Symbol('f1') + sp.Symbol('f2') + sp.Symbol('f3')},

        {'text': 'You deposit ${principal} in an account with a simple interest rate of {rate}% per year. What is the total amount after {years} years? Use the simple interest formula A = P(1 + rt) to find the final balance.',
         'equation': lambda: sp.Eq(sp.Symbol('A'), sp.Symbol('principal') * (1 + sp.Symbol('rate')/100 * sp.Symbol('years')))},

        {'text': 'A train travels at a constant speed of {v} km/h for {t} hours. How far does the train travel in total? Use the formula distance = speed × time.',
         'equation': lambda: sp.Eq(sp.Symbol('d'), sp.Symbol('v') * sp.Symbol('t'))},

        {'text': 'Worker A can complete a task in {t} hours alone, while Worker B can complete the same task in {t2} hours alone. How long would it take them to complete the task working together? Use the combined work rate formula 1/t_total = 1/t_A + 1/t_B.',
         'equation': lambda: sp.Eq(sp.Symbol('t_total'), 1 / (1/sp.Symbol('t_A') + 1/sp.Symbol('t_B')))},

        {'text': 'A triangle has a base of {l} cm and a height of {h} cm. What is the area of the triangle? Use the formula A = (1/2) × base × height.',
         'equation': lambda: sp.Eq(sp.Symbol('A'), sp.Rational(1, 2) * sp.Symbol('base') * sp.Symbol('height'))},

        {'text': 'A circle has a radius of {r} cm. Find both the area and the circumference of the circle. Use A = πr² and C = 2πr.',
         'equation': lambda: sp.Eq(sp.Symbol('A'), sp.pi * sp.Symbol('r')**2)},

        {'text': 'An arithmetic sequence has a first term of {a1} and a common difference of {d}. What is the {n}th term of this sequence? Use the formula a_n = a_1 + (n-1)d.',
         'equation': lambda: sp.Eq(sp.Symbol('a_n'), sp.Symbol('a1') + (sp.Symbol('n') - 1) * sp.Symbol('d'))},

        {'text': 'A geometric series has a first term of {a1} and a common ratio of {gr}. What is the sum of the first {n} terms? Use the formula S_n = a_1(1-r^n)/(1-r).',
         'equation': lambda: sp.Eq(sp.Symbol('S_n'), sp.Symbol('a1') * (1 - sp.Symbol('r')**sp.Symbol('n')) / (1 - sp.Symbol('r')))},

        {'text': 'A committee of {k} people is to be chosen from a group of {n} people. How many different committees are possible? Use the combination formula C(n,k) = n!/(k!(n-k)!).',
         'equation': lambda: sp.Eq(sp.Symbol('C'), sp.binomial(sp.Symbol('n'), sp.Symbol('k')))},

        {'text': 'A {l}-meter ladder leans against a wall. The bottom of the ladder slides away from the wall at a rate of {v} m/s. How fast is the top of the ladder sliding down when the bottom is {r} meters from the wall? Use the Pythagorean relation x² + y² = L².',
         'equation': lambda: sp.Eq(sp.Symbol('dy/dt'), -sp.Symbol('x')/sp.Symbol('y') * sp.Symbol('dx/dt'))},

        {'text': 'A cylindrical can must have a volume of {v} cubic cm. Find the radius that minimizes the total surface area of the can. Use V = πr²h and A = 2πr² + 2πrh.',
         'equation': lambda: sp.Eq(sp.Symbol('r'), (sp.Symbol('V') / (2*sp.pi))**sp.Rational(1, 3))},

        {'text': "A net force of {f} N is applied to an object with a mass of {m} kg. What is the resulting acceleration of the object? Use Newton's second law F = ma.",
         'equation': lambda: sp.Eq(sp.Symbol('a'), sp.Symbol('F') / sp.Symbol('m'))},

        {'text': 'A radioactive sample starts with {n0} grams and has a half-life of {half} years. How much of the sample remains after {years} years? Use the decay formula N(t) = N_0 × (1/2)^(t/half-life).',
         'equation': lambda: sp.Eq(sp.Symbol('N'), sp.Symbol('N0') * (sp.Rational(1, 2))**(sp.Symbol('t')/sp.Symbol('half')))},

        {'text': 'Two vectors are given as u = ({u1}, {u2}) and v = ({vx}, {vy}). Find the angle θ between the two vectors. Use the formula cos(θ) = (u·v)/(|u||v|).',
         'equation': lambda: sp.Eq(sp.cos(sp.Symbol('theta')),
                                    (sp.Symbol('u1')*sp.Symbol('v1') + sp.Symbol('u2')*sp.Symbol('v2')) /
                                    (sp.sqrt(sp.Symbol('u1')**2 + sp.Symbol('u2')**2) * sp.sqrt(sp.Symbol('v1')**2 + sp.Symbol('v2')**2)))},
    ]

    template = random.choice(templates)
    text = template['text']

    # Replace placeholders
    replacements = {
        '{v}': str(ri(1, 20)), '{v0}': str(ri(1, 20)), '{t}': str(ri(1, 30)),
        '{t2}': str(ri(1, 20)), '{r}': str(ri(1, 10)), '{principal}': str(ri(100, 10000)),
        '{rate}': str(round(random.uniform(1, 10), 1)), '{years}': str(ri(1, 30)),
        '{l}': str(ri(5, 20)), '{w}': str(ri(2, 10)),
        '{c1}': str(ri(1, 5)), '{c2}': str(ri(1, 10)), '{c3}': str(ri(1, 100)),
        '{fence}': str(ri(50, 200)), '{k}': str(ri(1, 10)), '{n}': str(ri(5, 20)),
        '{k_val}': str(ri(1, 5)), '{m}': str(ri(1, 50)), '{h}': str(ri(1, 100)),
        '{g}': str(9.8), '{theta}': str(ri(15, 75)), '{angle}': str(ri(15, 75)),
        '{f1}': str(ri(1, 10)), '{f2}': str(ri(1, 10)), '{f3}': str(ri(1, 10)),
        '{f4}': str(ri(1, 10)), '{f5}': str(ri(1, 10)),
        '{terms}': str(ri(3, 10)), '{n_val}': str(ri(1, 5)),
        '{a1}': str(ri(1, 5)), '{a2}': str(ri(1, 5)), '{a3}': str(ri(1, 5)), '{a4}': str(ri(1, 5)),
        '{a11}': str(ri(1, 5)), '{a12}': str(ri(1, 5)), '{a21}': str(ri(1, 5)), '{a22}': str(ri(1, 5)),
        '{b11}': str(ri(1, 5)), '{b12}': str(ri(1, 5)), '{b21}': str(ri(1, 5)), '{b22}': str(ri(1, 5)),
        '{area1}': str(ri(10, 100)), '{r1}': str(ri(1, 10)), '{r2}': str(ri(1, 10)),
        '{r3}': str(ri(1, 10)), '{v1}': str(ri(1, 10)), '{v2}': str(ri(1, 10)),
        '{v3}': str(ri(1, 10)), '{t_end}': str(ri(1, 10)), '{A}': str(round(random.uniform(1, 5), 1)),
        '{w}': str(ri(1, 10)), '{r1}': str(ri(1, 10)), '{r2}': str(ri(1, 10)),
        '{r3}': str(ri(1, 10)), '{r1}': str(ri(1, 10)), '{r2}': str(ri(1, 10)),
        '{r3}': str(ri(1, 10)), '{r1}': str(ri(1, 10)), '{r2}': str(ri(1, 10)),
        '{r3}': str(ri(1, 10)),
        '{d}': str(ri(1, 10)), '{gr}': str(ri(2, 4)),
        '{f}': str(ri(1, 100)), '{n0}': str(ri(10, 1000)), '{half}': str(ri(1, 50)),
        '{u1}': str(ri(-10, 10)), '{u2}': str(ri(-10, 10)),
        '{vx}': str(ri(-10, 10)), '{vy}': str(ri(-10, 10)),
    }

    for key, val in replacements.items():
        text = text.replace(key, val)

    # Final pass for any remaining placeholders
    remaining = ['{k}', '{n}', '{k_val}', '{m}', '{h}', '{theta}', '{angle}',
                 '{f1}', '{f2}', '{f3}', '{f4}', '{f5}', '{terms}', '{n_val}',
                 '{a1}', '{a2}', '{a3}', '{a4}', '{a11}', '{a12}', '{a21}', '{a22}',
                 '{b11}', '{b12}', '{b21}', '{b22}', '{area1}', '{r1}', '{r2}', '{r3}',
                 '{v1}', '{v2}', '{v3}', '{t_end}', '{A}', '{v0}', '{rate}', '{years}',
                 '{principal}', '{fence}', '{c1}', '{c2}', '{c3}']
    for key in remaining:
        if key in text:
            text = text.replace(key, str(ri(1, 100)))

    try:
        eq = template['equation']()
        if isinstance(eq, dict):
            latex = ', \\ '.join(f"{sp.latex(k)} = {sp.latex(v)}" for k, v in eq.items())
        else:
            latex = sp.latex(eq)
        latex = latex.replace('\n', ' ').replace('\r', '')
        return {'text': text, 'latex': latex, 'category': 'word_problem'}
    except Exception as e:
        print(f"Error generating word problem: {e}")
        return None

print("Generating algebra expressions...")
algebra_items = []
used_latex = set()

for _ in range(50):
    func = random.choice([gen_polynomial, gen_quadratic, gen_factorial, gen_sum, gen_product,
                          gen_log, gen_exp, gen_root, gen_binomial, gen_rational,
                          gen_power, gen_abs, gen_piecewise_expr, gen_complex,
                          gen_inequality, gen_trig, gen_hyperbolic, gen_derivative,
                          gen_integral, gen_definite_integral, gen_limit, gen_series,
                          gen_matrix_det, gen_cramer, gen_matrix_solve, gen_matrix_orthogonality,
                          gen_matrix_rank, gen_matrix_block, gen_matrix_transpose, gen_matrix_power,
                          gen_matrix_inverse, gen_matrix_eigen, gen_matrix_augmented])
    try:
        result = func()
        if result is None:
            continue
        latex = sp.latex(result) if hasattr(result, 'latex') else sp.latex(sp.Eq(result, 0))
        latex = latex.replace('\n', ' ').replace('\r', '')
        # Filter out broken results that are just boolean values
        if latex.strip() in ('False', 'True', '\\text{False}', '\\text{True}'):
            continue
        if latex in used_latex:
            continue
        used_latex.add(latex)
        algebra_items.append({'latex': latex, 'category': 'equation', 'subcat': 'algebra'})
    except Exception as e:
        continue
    if len(algebra_items) >= 40:
        break

print(f"  Generated {len(algebra_items)} algebra expressions")

print("Generating calculus expressions...")
calculus_items = []
for _ in range(30):
    func = random.choice([gen_chain_rule, gen_product_rule, gen_quotient_rule, gen_partial,
                          gen_taylor, gen_implicit, gen_ode_simple, gen_second_derivative,
                          gen_integration_by_parts, gen_vector_derivative, gen_gradient,
                          gen_jacobian, gen_divergence, gen_line_integral, gen_double_integral])
    try:
        result = func()
        latex = sp.latex(result) if hasattr(result, 'latex') else sp.latex(sp.Eq(result, 0))
        latex = latex.replace('\n', ' ').replace('\r', '')
        # Filter out broken results that are just boolean values
        if latex.strip() in ('False', 'True', '\\text{False}', '\\text{True}'):
            continue
        calculus_items.append({'latex': latex, 'category': 'equation', 'subcat': 'calculus'})
    except Exception as e:
        continue
    if len(calculus_items) >= 25:
        break

print(f"  Generated {len(calculus_items)} calculus expressions")

print("Generating matrix expressions...")
matrix_items = []
for _ in range(200):
    func = random.choice([gen_matrix_mult, gen_matrix_power, gen_matrix_inverse, gen_matrix_eigen,
                          gen_matrix_transpose, gen_matrix_rank, gen_matrix_system, gen_matrix_augmented,
                          gen_matrix_sylvester, gen_matrix_block, gen_matrix_orthogonal, gen_matrix_orthogonality,
                          gen_matrix_det, gen_cramer, gen_matrix_solve])
    try:
        result = func()
        if result is None:
            continue

        latex = sp.latex(result)
        latex = latex.replace('\n', ' ').replace('\r', '')
        # Filter out broken results that are just boolean values
        if latex.strip() in ('False', 'True', '\\text{False}', '\\text{True}'):
            continue
        if latex in used_latex:
            continue
        used_latex.add(latex)
        matrix_items.append({'latex': latex, 'category': 'equation', 'subcat': 'matrix'})
    except Exception as e:
        continue
    if len(matrix_items) >= 15:
        break

print(f"  Generated {len(matrix_items)} matrix expressions")

print("Generating word problems...")
word_problems = []
for _ in range(80):
    result = gen_word_problem()
    if result:
        word_problems.append(result)
    if len(word_problems) >= 53:
        break

print(f"  Generated {len(word_problems)} word problems")

# Combine and save
all_equations = algebra_items + calculus_items + matrix_items
print(f"\nTotal: {len(all_equations)} equations + {len(word_problems)} word problems = {len(all_equations) + len(word_problems)} items")

os.makedirs('data', exist_ok=True)

with open('data/raw_items.json', 'w') as f:
    json.dump(all_equations, f, indent=2)

with open('data/raw_word_problems.json', 'w') as f:
    json.dump(word_problems, f, indent=2)

print("Saved raw items to data/raw_items.json and data/raw_word_problems.json")