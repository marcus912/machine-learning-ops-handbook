# Basics of Linear Algebra for Machine Learning - Comprehensive Summary

**Author:** Jason Brownlee
**Publisher:** Machine Learning Mastery, 2018
**Pages:** 212
**Focus:** Practical linear algebra fundamentals for ML practitioners using NumPy, emphasizing the "mathematical language of data"

---

## Core Definition

> "Linear algebra is a pillar of machine learning. You cannot develop a deep understanding and application of machine learning without it. This book was designed to teach you linear algebra in a systematic way as an ML practitioner."

---

## Part I: Foundations

### Chapter 1-3: Introduction to NumPy and Arrays

**N-Dimensional Arrays (ndarray):**
```python
from numpy import array

# Create arrays
v = array([1, 2, 3])              # Vector (1D)
M = array([[1, 2], [3, 4]])       # Matrix (2D)
T = array([[[1,2],[3,4]],[[5,6],[7,8]]])  # Tensor (3D)
```

**Key Array Properties:**
| Property | Access | Example |
|----------|--------|---------|
| Shape | `.shape` | (3,), (2,2), (2,2,2) |
| Dimensions | `.ndim` | 1, 2, 3 |
| Data Type | `.dtype` | float64, int32 |
| Size | `.size` | Total element count |

---

## Part II: Vectors

### Chapter 4: Vectors and Vector Arithmetic

**Vector Definition:**
A vector is an ordered list of numbers (a 1D array).

**Vector Operations:**
```python
from numpy import array

a = array([1, 2, 3])
b = array([4, 5, 6])

# Element-wise operations
c = a + b      # Addition: [5, 7, 9]
d = a - b      # Subtraction: [-3, -3, -3]
e = a * b      # Hadamard product: [4, 10, 18]
f = a / b      # Division: [0.25, 0.4, 0.5]
```

### Chapter 5: Vector Norms

**Common Norms:**
| Norm | Formula | NumPy |
|------|---------|-------|
| **L1 (Manhattan)** | Σ|xᵢ| | `norm(v, 1)` |
| **L2 (Euclidean)** | √(Σxᵢ²) | `norm(v, 2)` or `norm(v)` |
| **L∞ (Max)** | max(|xᵢ|) | `norm(v, inf)` |

```python
from numpy.linalg import norm

v = array([1, 2, 3])
l1 = norm(v, 1)    # 6.0
l2 = norm(v, 2)    # 3.74...
linf = norm(v, inf) # 3.0
```

### Chapter 6: Vectors and Angles

**Dot Product:**
```python
from numpy import dot

a = array([1, 2, 3])
b = array([4, 5, 6])
result = dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
```

**Cosine Similarity:**
```
cos(θ) = (a · b) / (||a|| × ||b||)
```

---

## Part III: Matrices

### Chapter 7-8: Matrices and Matrix Arithmetic

**Matrix Definition:**
A matrix is a 2D array of numbers with rows and columns.

**Matrix Operations:**
```python
from numpy import array

A = array([[1, 2], [3, 4]])
B = array([[5, 6], [7, 8]])

# Element-wise
C = A + B      # Addition
D = A - B      # Subtraction
E = A * B      # Hadamard product

# Matrix multiplication
from numpy import dot
F = dot(A, B)  # or A @ B in Python 3.5+
```

### Chapter 9: Matrix-Vector Products

```python
# Matrix-vector multiplication
A = array([[1, 2], [3, 4], [5, 6]])  # 3x2
v = array([0.5, 0.5])                 # 2x1
result = dot(A, v)                    # 3x1: [1.5, 3.5, 5.5]
```

### Chapter 10: Matrix-Matrix Multiplication

**Rule:** (m × n) @ (n × p) = (m × p)

```python
A = array([[1, 2], [3, 4]])  # 2x2
B = array([[5, 6], [7, 8]])  # 2x2
C = dot(A, B)                # 2x2: [[19, 22], [43, 50]]
```

### Chapter 11: Matrix Operations

**Key Operations:**
| Operation | NumPy | Description |
|-----------|-------|-------------|
| **Transpose** | `A.T` | Flip rows and columns |
| **Inverse** | `inv(A)` | A⁻¹ such that A × A⁻¹ = I |
| **Determinant** | `det(A)` | Scalar volume measure |
| **Trace** | `trace(A)` | Sum of diagonal elements |
| **Rank** | `matrix_rank(A)` | Number of linearly independent rows/cols |

```python
from numpy.linalg import inv, det
from numpy import trace

A = array([[1, 2], [3, 4]])

A_T = A.T           # Transpose
A_inv = inv(A)      # Inverse
A_det = det(A)      # Determinant: -2.0
A_trace = trace(A)  # Trace: 5
```

---

## Part IV: Sparse Matrices

### Chapter 12: Sparse Matrices

**Definition:** Matrices with mostly zero values.

**Sparsity Score:**
```
sparsity = 1.0 - count_nonzero(A) / A.size
```

**Sparse Matrix Formats:**
| Format | Description |
|--------|-------------|
| **COO** | Coordinate list (row, col, value) |
| **CSR** | Compressed Sparse Row |
| **CSC** | Compressed Sparse Column |
| **DOK** | Dictionary of Keys |

```python
from scipy.sparse import csr_matrix

# Dense to sparse
A = array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
S = csr_matrix(A)

# Sparse to dense
B = S.todense()
```

**ML Applications of Sparse Matrices:**
- One-hot encoding
- TF-IDF text representations
- Recommender systems
- Word embeddings (sparse)

---

## Part V: Tensors

### Chapter 13: Tensors and Tensor Arithmetic

**Tensor Definition:**
Generalization of vectors (1D) and matrices (2D) to N dimensions.

```python
# 3D Tensor (e.g., RGB image batch)
T = array([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])  # Shape: (2, 2, 2)
```

**Tensor Products:**
```python
from numpy import tensordot

A = array([1, 2])
B = array([3, 4])
C = tensordot(A, B, axes=0)  # Outer product
```

---

## Part VI: Matrix Factorization

### Chapter 14: LU Decomposition

**Formula:** A = L × U
- L: Lower triangular matrix
- U: Upper triangular matrix

```python
from scipy.linalg import lu

A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
P, L, U = lu(A)
```

**Use Cases:**
- Solving linear systems
- Matrix inversion
- Computing determinants

### Chapter 15: QR Decomposition

**Formula:** A = Q × R
- Q: Orthogonal matrix (Q^T × Q = I)
- R: Upper triangular matrix

```python
from numpy.linalg import qr

A = array([[1, 2], [3, 4], [5, 6]])
Q, R = qr(A, mode='complete')
```

**Use Cases:**
- Least squares solutions
- Eigenvalue algorithms
- Numerical stability

### Chapter 16: Eigendecomposition

**Formula:** A × v = λ × v
- v: Eigenvector
- λ: Eigenvalue

```python
from numpy.linalg import eig

A = array([[1, 2], [3, 4]])
values, vectors = eig(A)
```

**Use Cases:**
- PCA (Principal Component Analysis)
- Spectral clustering
- Graph analysis

### Chapter 17: Singular Value Decomposition (SVD)

**Formula:** A = U × Σ × V^T
- U: Left singular vectors
- Σ: Diagonal matrix of singular values
- V: Right singular vectors

```python
from numpy.linalg import svd

A = array([[1, 2], [3, 4], [5, 6]])
U, s, VT = svd(A)
```

**Use Cases:**
- Dimensionality reduction
- Recommender systems
- Data compression
- Pseudoinverse calculation
- Noise reduction

---

## Part VII: Statistics

### Chapter 18: Multivariate Statistics

**Key Statistical Measures:**
| Measure | Formula | NumPy |
|---------|---------|-------|
| **Mean** | μ = (1/n)Σxᵢ | `mean(v)` |
| **Variance** | σ² = (1/(n-1))Σ(xᵢ-μ)² | `var(v, ddof=1)` |
| **Std Dev** | σ = √σ² | `std(v, ddof=1)` |

```python
from numpy import mean, var, std

v = array([1, 2, 3, 4, 5, 6])
mu = mean(v)          # 3.5
sigma2 = var(v, ddof=1)  # 3.5
sigma = std(v, ddof=1)   # 1.87
```

### Chapter 19: Covariance and Correlation

**Covariance:**
```python
from numpy import cov

x = array([1, 2, 3, 4, 5])
y = array([5, 4, 3, 2, 1])
Sigma = cov(x, y)[0, 1]  # -2.5 (negative correlation)
```

**Correlation (Pearson):**
```python
from numpy import corrcoef

corr = corrcoef(x, y)[0, 1]  # -1.0 (perfect negative)
```

**Covariance Matrix:**
```python
X = array([[1, 5, 8], [3, 5, 11], [2, 4, 9]])
Sigma = cov(X.T)  # Features as rows
```

---

## Part VIII: Applications

### Chapter 20: Principal Component Analysis (PCA)

**Steps:**
1. Center data (subtract mean)
2. Compute covariance matrix
3. Eigendecomposition
4. Select top k eigenvectors
5. Project data

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### Chapter 21: Linear Regression

**Normal Equation:**
```
w = (X^T × X)^(-1) × X^T × y
```

```python
from numpy.linalg import inv

w = inv(X.T @ X) @ X.T @ y
```

**Pseudoinverse (more stable):**
```python
from numpy.linalg import pinv

w = pinv(X) @ y
```

---

## Key Tools & Technologies

| Category | Tools |
|----------|-------|
| **Core Arrays** | NumPy |
| **Sparse Matrices** | SciPy (scipy.sparse) |
| **Linear Algebra** | numpy.linalg, scipy.linalg |
| **Statistics** | NumPy, SciPy |
| **ML Integration** | scikit-learn, Keras, TensorFlow |

---

## Best Practices Summary

1. **Use NumPy for Everything**: Vectorized operations are 10-100x faster than loops
2. **Check Shapes**: Most errors come from shape mismatches
3. **Understand Broadcasting**: NumPy's automatic shape alignment rules
4. **Use Sparse for Large Data**: Memory and compute savings when sparsity > 50%
5. **Know Your Decompositions**: SVD for rectangular, Eigen for square matrices
6. **Prefer Pseudoinverse**: More numerically stable than direct inversion
7. **Center Data for PCA**: Always subtract mean before decomposition
8. **Set ddof=1 for Sample Statistics**: Unbiased estimators need correction

---

## Target Audience

- ML practitioners needing mathematical foundations
- Data scientists implementing algorithms from scratch
- Engineers debugging numerical issues
- Students preparing for ML interviews
- Developers transitioning to ML roles

---

## Key Takeaways for ML Engineers

1. **Vectors Are Features**: ML inputs are vectors of feature values
2. **Matrices Are Datasets**: Rows are samples, columns are features
3. **Matrix Multiplication Is Transformation**: Neural network layers are matrix ops
4. **SVD Powers Everything**: Recommenders, PCA, pseudoinverse, compression
5. **Sparsity Is Common**: Text, categorical data, embeddings often sparse
6. **Covariance Matrix Is Central**: PCA, Gaussian distributions, whitening
7. **Norms Measure Distance**: L2 for similarity, L1 for sparsity (regularization)
8. **Eigenvalues Show Variance**: Principal components capture data directions

---

## NumPy Quick Reference

```python
# Creating arrays
array([1, 2, 3])           # From list
zeros((3, 3))              # All zeros
ones((3, 3))               # All ones
eye(3)                     # Identity matrix
arange(0, 10, 2)           # Range [0, 2, 4, 6, 8]

# Reshaping
A.reshape((2, 3))          # Change shape
A.flatten()                # To 1D
A.T                        # Transpose

# Operations
dot(A, B)                  # Matrix multiply
A @ B                      # Matrix multiply (Python 3.5+)
A * B                      # Element-wise multiply

# Linear algebra (numpy.linalg)
inv(A)                     # Inverse
pinv(A)                    # Pseudoinverse
det(A)                     # Determinant
eig(A)                     # Eigenvalues/vectors
svd(A)                     # Singular value decomposition
norm(v)                    # Vector/matrix norm
matrix_rank(A)             # Matrix rank

# Statistics
mean(A, axis=0)            # Column means
var(A, ddof=1, axis=0)     # Column variances
std(A, ddof=1, axis=0)     # Column std devs
cov(X.T)                   # Covariance matrix
corrcoef(x, y)             # Correlation matrix
```
