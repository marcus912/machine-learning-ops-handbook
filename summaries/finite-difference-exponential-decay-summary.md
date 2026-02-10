# Finite Difference Computing with Exponential Decay Models - Comprehensive Summary

**Author:** Hans Petter Langtangen
**Publisher:** Springer (Lecture Notes in Computational Science and Engineering), 2016
**Pages:** 210
**Focus:** Teaching scientific computing fundamentals through the simple ODE u' = -au, bridging mathematics, algorithms, and Python programming

---

## Core Definition

> "This book teaches the basic components in the scientific computing pipeline: modeling, differential equations, numerical algorithms, programming, plotting, and software testing. The pedagogical idea is to treat these topics in the context of a very simple mathematical model, the differential equation for exponential decay, u'(t) = -au(t)."

---

## Chapter 1: Algorithms and Implementations

### The Basic Model

The exponential decay ODE:
```
u'(t) = -au(t), u(0) = I
```
Where:
- u(t): Unknown function of time
- a: Positive decay constant
- I: Initial condition

**Exact Solution:** u(t) = I·e^(-at)

### Finite Difference Schemes

| Scheme | Formula | Accuracy | Stability |
|--------|---------|----------|-----------|
| **Forward Euler** | u^(n+1) = u^n - a·Δt·u^n | O(Δt) | Conditional: aΔt < 2 |
| **Backward Euler** | u^(n+1) = u^n - a·Δt·u^(n+1) | O(Δt) | Unconditional |
| **Crank-Nicolson** | u^(n+1) = u^n - a·Δt·(u^n + u^(n+1))/2 | O(Δt²) | Unconditional |

### The θ-Rule (Unified Scheme)

```
u^(n+1) = u^n - aΔt[θ·u^(n+1) + (1-θ)·u^n]
```

| θ Value | Scheme |
|---------|--------|
| θ = 0 | Forward Euler |
| θ = 1 | Backward Euler |
| θ = 0.5 | Crank-Nicolson |

### Python Implementation

```python
def solver(I, a, T, dt, theta):
    """Solve u'=-a*u, u(0)=I, for t in (0,T]."""
    dt = float(dt)
    Nt = int(round(T/dt))
    T = Nt*dt
    u = np.zeros(Nt+1)
    t = np.linspace(0, T, Nt+1)

    u[0] = I
    for n in range(0, Nt):
        u[n+1] = (1 - (1-theta)*a*dt)/(1 + theta*dt*a)*u[n]
    return u, t
```

### Verification Techniques

1. **Reproducing Constant Solutions**: u = C should satisfy discrete equations
2. **Reproducing Linear Solutions**: u = Ct + D should be exact for certain schemes
3. **Convergence Rate Testing**: Error should decrease as O(Δt^p) for p-order scheme
4. **Manufactured Solutions**: Create exact solution, derive source term

---

## Chapter 2: Analysis

### Stability Analysis

**Amplification Factor:**
```
A = u^(n+1)/u^n
```

For the θ-rule:
```
A = (1 - (1-θ)aΔt)/(1 + θaΔt)
```

**Stability Criterion:** |A| ≤ 1

| Scheme | Stability Condition |
|--------|-------------------|
| Forward Euler | aΔt ≤ 2 |
| Backward Euler | Always stable |
| Crank-Nicolson | Always stable |

### Accuracy Analysis

**Truncation Error:** Error when exact solution substituted into discrete equation

| Scheme | Truncation Error |
|--------|-----------------|
| Forward Euler | O(Δt) |
| Backward Euler | O(Δt) |
| Crank-Nicolson | O(Δt²) |

### Key Concepts

- **Consistency**: Truncation error → 0 as Δt → 0
- **Stability**: Errors don't grow unboundedly
- **Convergence**: Numerical solution → exact solution as Δt → 0

**Lax Equivalence Theorem:** Consistency + Stability = Convergence

### Error Types in Modeling

| Error Type | Description |
|------------|-------------|
| **Model Errors** | Simplifications in mathematical model |
| **Data Errors** | Uncertainty in parameters |
| **Discretization Errors** | From finite difference approximation |
| **Rounding Errors** | Finite precision arithmetic |

---

## Chapter 3: Generalizations

### General ODE Form

```
u'(t) = f(u, t), u(0) = I
```

### Advanced Schemes

**Runge-Kutta Methods:**

| Order | Method |
|-------|--------|
| 2nd | Heun's method, Midpoint method |
| 4th | Classical RK4 |

**RK4 Formula:**
```
k1 = Δt·f(u^n, t^n)
k2 = Δt·f(u^n + k1/2, t^n + Δt/2)
k3 = Δt·f(u^n + k2/2, t^n + Δt/2)
k4 = Δt·f(u^n + k3, t^n + Δt)
u^(n+1) = u^n + (k1 + 2k2 + 2k3 + k4)/6
```

**Adams-Bashforth Methods:**
- 2nd order: Uses u^n and u^(n-1)
- 3rd order: Uses u^n, u^(n-1), u^(n-2)

**Leapfrog Scheme:**
```
u^(n+1) = u^(n-1) + 2Δt·f(u^n, t^n)
```

### Odespy Library

Python interface to ODE solvers:
```python
import odespy
solver = odespy.RK4(f)
solver.set_initial_condition(I)
u, t = solver.solve(time_points)
```

---

## Chapter 4: Physical Models (Applications)

### Population Dynamics

**Exponential Growth/Decay:**
```
N'(t) = (b - d)N, N(0) = N0
```
- b: Birth rate
- d: Death rate

**Logistic Growth:**
```
u' = ρ(1 - u/M)u
```
- ρ: Initial growth rate
- M: Carrying capacity

### Compound Interest

**Continuous Compounding:**
```
u' = (r/100)u, u(0) = u0
```
Solution: u(t) = u0·e^(rt/100)

### Newton's Law of Cooling

```
dT/dt = -k(T - Ts)
```
- T: Body temperature
- Ts: Surrounding temperature
- k: Heat transfer coefficient

### Radioactive Decay

**Deterministic Model:**
```
u' = -au, u(0) = 1
```
Half-life: t_1/2 = ln(2)/a

**Stochastic Model:**
- Individual atoms decay randomly
- Mean behavior follows deterministic ODE

### Chemical Kinetics

**First-Order Reaction (A → B):**
```
d[A]/dt = -k[A]
```

**Second-Order Reaction (A + B → C):**
```
d[A]/dt = -k[A][B]
```

### Disease Spreading (SIR Model)

```
S' = -βSI
I' = βSI - γI
R' = γI
```
- S: Susceptible
- I: Infected
- R: Recovered

### Vertical Motion in Fluid

```
m·dv/dt = -mg + F_d + F_b
```
- F_d: Drag force
- F_b: Buoyancy force

### Scaling and Dimensionless Numbers

Transform u'= -au + b to dimensionless form:
```
dū/dt̄ = -ū, ū(0) = 1 - β
```
where β = b/(Ia)

---

## Chapter 5: Scientific Software Engineering

### Code Organization

**Module Structure:**
```
decay_mod/
├── __init__.py
├── solver.py
├── experiments.py
└── visualization.py
```

### User Interfaces

1. **Command-Line Arguments:**
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=float, default=0.1)
```

2. **Web GUI (Flask/Parampool):**
```python
from parampool.generator.flask import generate
generate(compute_function, pool, 'decay_gui')
```

### Testing Framework

**Doctest:**
```python
def solver(I, a, T, dt, theta):
    """
    >>> u, t = solver(I=1, a=2, T=4, dt=0.5, theta=1)
    >>> print(u)
    [1.    0.5   0.25  ...]
    """
```

**pytest:**
```python
def test_solver_linear_solution():
    """Test that linear u(t) is reproduced exactly."""
    u, t = solver(I=0.2, a=1.5, T=4, dt=0.5, theta=0.5)
    u_exact = lambda t: 0.2*t + 0.1
    diff = np.abs(u_exact(t) - u).max()
    assert diff < 1E-14
```

### Classes for Problem/Solver

```python
class Problem:
    def __init__(self, I=1, a=1, T=10):
        self.I, self.a, self.T = I, a, T

class Solver:
    def __init__(self, problem, dt=0.1, theta=0.5):
        self.problem = problem
        self.dt, self.theta = dt, theta

    def solve(self):
        # Implementation
```

### Automating Experiments

```python
def run_experiments(dt_values):
    for dt in dt_values:
        u, t = solver(I, a, T, dt, theta)
        error = compute_error(u, t)
        # Log and plot results
```

---

## Key Tools & Technologies

| Category | Tools |
|----------|-------|
| **Core Python** | NumPy, SciPy |
| **ODE Solvers** | Odespy, scipy.integrate |
| **Visualization** | Matplotlib |
| **Testing** | pytest, nose, doctest |
| **Symbolic Math** | SymPy |
| **Documentation** | Sphinx, DocOnce |

---

## Best Practices Summary

1. **Start Simple**: Master u' = -au before tackling PDEs
2. **Verify First**: Test with known solutions before production runs
3. **Understand Stability**: Know when schemes fail and why
4. **Use Convergence Rates**: Verify expected O(Δt^p) behavior
5. **Automate Testing**: Unit tests catch regressions
6. **Scale Equations**: Reduce parameters via dimensionless formulation
7. **Document Code**: Docstrings and doctest examples
8. **Modularize**: Separate solver, problem, and visualization

---

## Target Audience

- Graduate students in computational science
- Engineers learning numerical methods
- Scientists needing ODE simulation skills
- Programmers transitioning to scientific computing
- Bio/geo-scientists entering computational modeling
- Students preparing for PDE courses

---

## Key Takeaways for ML Engineers

1. **Finite Differences Foundation**: Same ideas apply to neural network discretization (ResNets as Euler discretization)
2. **Stability Analysis**: Understanding when algorithms fail applies to training dynamics
3. **Verification Methodology**: Testing numerical code rigorously transfers to ML testing
4. **Convergence Rates**: Important for understanding optimizer behavior
5. **Scaling**: Normalization in ML mirrors equation scaling
6. **Software Engineering**: Testing, modularity, documentation best practices

---

## Unique Value

This book bridges the gap between mathematical analysis and working code. Every formula is implemented in Python, every concept is verified computationally. The "simplify, understand, then generalize" philosophy provides a solid foundation for more complex numerical methods, including those used in scientific machine learning and physics-informed neural networks.
