# Linear Algebra Notes

A running file of linear-algebra concepts (with a short calculus appendix), in Definition / Intuition / Example / Notes form. Written to stand alone. Math is kept light but concrete: one-line formulas plus worked 2×2 / 3×3 examples, not proofs.

Source: **3Blue1Brown — Essence of Linear Algebra** (and *Essence of Calculus* for the appendix). The through-line is geometric: matrices are *transformations of space*, and everything else reads off that picture. Cross-references to the ML and statistics notes are marked *(ML)* / *(stat)*.

---

## Contents

**Vectors and Spaces** *(3Blue1Brown)*
- [Vectors](#vectors)
- [Basis vectors and unit vectors](#basis-vectors-and-unit-vectors)
- [Linear combinations and span](#linear-combinations-and-span)
- [Linear dependence and independence](#linear-dependence-and-independence)

**Linear Transformations and Matrices** *(3Blue1Brown)*
- [Linear transformations](#linear-transformations)
- [Matrices as linear transformations](#matrices-as-linear-transformations)
- [Matrix multiplication (composition)](#matrix-multiplication-composition)
- [Determinant](#determinant)
- [Systems of linear equations and the matrix inverse](#systems-of-linear-equations-and-the-matrix-inverse)
- [Rank and column space](#rank-and-column-space)
- [Null space (kernel)](#null-space-kernel)

**Products, Bases, and Eigen-things** *(3Blue1Brown)*
- [Dot product](#dot-product)
- [Cross product](#cross-product)
- [Cramer's rule](#cramers-rule)
- [Change of basis](#change-of-basis)
- [Eigenvectors and eigenvalues](#eigenvectors-and-eigenvalues)

**Calculus foundations** *(3Blue1Brown)*
- [Derivatives](#derivatives)
- [Integrals and area under the curve](#integrals-and-area-under-the-curve)
- [Taylor series](#taylor-series)

**[Glossary](#glossary)** — alphabetical index

---

## Vectors and Spaces *(3Blue1Brown)*

### Vectors

**Definition.** A vector can be read three ways: geometrically as an **arrow** from the origin with a length and direction; algebraically as an **ordered list of numbers** (its coordinates); and abstractly as anything you can add and scale. In 2D, `v = [3, 2]` means "3 along x, 2 along y."

**Intuition.** The arrow view and the list view are two languages for one object, and linear algebra is largely translating between them. Adding vectors = a tip-to-tail walk; scaling by a number = stretching/squishing the arrow (flipping it if the number is negative).

**Example.** `[1, 2] + [3, 0] = [4, 2]`; `2·[1, 2] = [2, 4]`; `−1·[1, 2] = [−1, −2]` (same line, opposite way).

**Notes.** Coordinates only mean something relative to a chosen basis. → Basis vectors and unit vectors, Linear combinations and span.

### Basis vectors and unit vectors

**Definition.** The **basis** of a coordinate system is a set of vectors whose combinations produce every vector in the space. In 2D the standard basis is the **unit vectors** `î = [1, 0]` and `ĵ = [0, 1]` (length 1, along the axes), and any `[x, y] = x·î + y·ĵ`.

**Intuition.** Coordinates are just the scalars you multiply the basis vectors by — the basis is the "ruler." Pick a different basis and the same arrow gets different coordinates.

**Example.** `[3, 2] = 3·î + 2·ĵ`. A unit vector has length 1: `[3, 4]` (length 5) normalizes to `[3/5, 4/5]`.

**Notes.** Any linearly independent set that spans the space can serve as a basis. → Linear combinations and span, Change of basis.

### Linear combinations and span

**Definition.** A **linear combination** of vectors `v, w` is `a·v + b·w` for scalars `a, b`. Their **span** is the set of *all* such combinations — every point reachable by scaling and adding them.

**Intuition.** Fix the vectors and sweep every scalar: you trace out their reachable region. In 2D, two vectors pointing in different directions span the *whole plane*; if they point the same way (one is a scalar multiple of the other) the span collapses to a *line*.

**Example.** In 3D: one vector spans a line; two independent vectors span a plane (a "sheet" through the origin); three independent vectors span *all* of 3D space.

**Notes.** A span always passes through the origin (take `a = b = 0`). → Linear dependence and independence, Rank and column space.

### Linear dependence and independence

**Definition.** Vectors are **linearly dependent** if at least one can be written as a combination of the others (it lies in their span and adds no new direction). Otherwise they are **linearly independent**.

**Intuition.** A dependent vector is redundant — dropping it doesn't shrink the span. Independent vectors each contribute a genuinely new dimension.

**Example.** `[1, 0]` and `[2, 0]` are dependent (`[2, 0] = 2·[1, 0]`; span is just a line). `[1, 0]` and `[0, 1]` are independent (span the plane).

**Notes.** A basis is a linearly independent set that spans the space; its size is the space's dimension. → Linear combinations and span, Rank and column space.

---

## Linear Transformations and Matrices *(3Blue1Brown)*

### Linear transformations

**Definition.** A transformation of space that (1) keeps every **line straight** (grid lines stay parallel and evenly spaced) and (2) keeps the **origin fixed**. It is fully determined by where it sends the basis vectors `î` and `ĵ`.

**Intuition.** "Linear" = the grid stays a grid — no curving, no moving the origin. Since any vector is `x·î + y·ĵ`, once you know where `î` and `ĵ` land you know where *everything* lands: the image of `[x, y]` is `x·(new î) + y·(new ĵ)`.

**Example.** A 90° counter-clockwise rotation sends `î = [1, 0] → [0, 1]` and `ĵ = [0, 1] → [−1, 0]`. So `[2, 1] → 2·[0, 1] + 1·[−1, 0] = [−1, 2]`.

**Notes.** "Track the basis vectors" is exactly why matrices work. → Matrices as linear transformations.

### Matrices as linear transformations

**Definition.** A matrix stores a linear transformation: its **columns are the landing spots of the basis vectors**. The 2×2 matrix `[[a, b], [c, d]]` sends `î → [a, c]` and `ĵ → [b, d]`, and applying it is `[[a,b],[c,d]]·[x, y] = [a·x + b·y, c·x + d·y]`.

**Intuition.** Matrix-times-vector is just "scale each landed basis vector by the matching input coordinate, then add" — reading the columns as the new `î` and `ĵ`.

**Example.** The rotation matrix `[[0, −1], [1, 0]]` times `[2, 1]` = `[0·2 + (−1)·1, 1·2 + 0·1] = [−1, 2]` — matching the rotation above.

**Notes.** A matrix whose columns are linearly dependent squishes space onto a line or point (determinant 0). → Determinant, Matrix multiplication (composition).

### Matrix multiplication (composition)

**Definition.** The product `M₂·M₁` is the single transformation that does `M₁` **then** `M₂` (read right-to-left, like function composition). Compute it by applying `M₂` to each column of `M₁`.

**Intuition.** Multiplying matrices = chaining transformations. Order matters (`M₂·M₁ ≠ M₁·M₂` in general) because "rotate then shear" isn't "shear then rotate."

**Example.** `[[0,−1],[1,0]] · [[1,1],[0,1]]` (rotate ∘ shear): apply the rotation to the shear's columns — `[1,0] → [0,1]` and `[1,1] → [−1,1]` — giving `[[0,−1],[1,1]]`.

**Notes.** Associative (`(AB)C = A(BC)`) but not commutative. → Matrices as linear transformations.

### Determinant

**Definition.** The **determinant** measures how much a transformation **scales areas** (2D) or **volumes** (3D). For 2×2, `det[[a,b],[c,d]] = ad − bc`. For 3×3 `[[a,b,c],[d,e,f],[g,h,i]]`, expand along the top row: `det = a(ei − fh) − b(di − fg) + c(dh − eg)`.

**Intuition.** A unit square (area 1) becomes a parallelogram of area `|det|`. `det = 0` means space is squished into a lower dimension (columns are dependent). A **negative** determinant means orientation flipped — space turned inside-out.

**Example (2×2).** `det[[3,1],[0,2]] = 3·2 − 1·0 = 6` → areas scale ×6. **Example (3×3).** `det[[1,2,3],[0,1,4],[5,6,0]] = 1(1·0 − 4·6) − 2(0·0 − 4·5) + 3(0·6 − 1·5) = −24 + 40 − 15 = 1`.

**Notes.** `det = 0` ⇔ no inverse ⇔ columns dependent ⇔ rank drops. → Systems of linear equations and the matrix inverse, Rank and column space, Cross product.

### Systems of linear equations and the matrix inverse

**Definition.** A linear system is `A·x = b` for unknown `x`. If `A` is invertible (`det A ≠ 0`), the unique solution is `x = A⁻¹·b`. For 2×2 `A = [[a,b],[c,d]]`, `A⁻¹ = (1/det)·[[d, −b], [−c, a]]`. For 3×3, `A⁻¹ = (1/det)·adjugate` (transpose of the cofactor matrix) — or just solve by Gaussian elimination.

**Intuition.** `A⁻¹` is the transformation that *undoes* `A` (rotate back, un-stretch). Solving `A·x = b` asks "which input lands on `b`?" — apply the reverse transformation to `b`.

**Example (2×2).** `A = [[3,1],[0,2]]`, `det = 6`, so `A⁻¹ = (1/6)[[2,−1],[0,3]]`. To solve `A·x = [5, 4]`: `x = A⁻¹[5,4] = (1/6)[2·5 − 1·4, 3·4] = (1/6)[6, 12] = [1, 2]` (check: `A·[1,2] = [5,4]` ✓).

**Notes.** If `det A = 0` there is no inverse, and the system has either no solution or infinitely many. → Determinant, Null space (kernel), Cramer's rule.

### Rank and column space

**Definition.** The **column space** is the span of a matrix's columns (all outputs `A·x`). The **rank** is the *dimension* of that column space — how many independent directions survive the transformation.

**Intuition.** Rank = "how many dimensions come out the other side." A 2×2 matrix has rank 2 (full rank, output fills the plane), rank 1 (output collapses to a line), or rank 0 (everything → origin). Full rank ⇔ invertible ⇔ `det ≠ 0`.

**Example.** `[[1,2],[2,4]]` has rank 1 (column 2 = 2× column 1, so the output is a line). `[[1,0],[0,1]]` has rank 2.

**Notes.** Rank drop ⇔ determinant 0 ⇔ a non-trivial null space appears. → Null space (kernel), Determinant.

### Null space (kernel)

**Definition.** The **null space** (kernel) of `A` is every vector `x` with `A·x = 0` — all the vectors the transformation squishes onto the origin.

**Intuition.** When `A` collapses a dimension (rank drops), a whole line or plane of vectors gets crushed to zero; that set is the null space. For the homogeneous system `A·x = 0`, the null space **is** the complete set of solutions.

**Example.** `A = [[1,2],[2,4]]` (rank 1): `A·[2,−1] = [1·2 + 2·(−1), 2·2 + 4·(−1)] = [0, 0]`, so `[2,−1]` and all its multiples form the null space — a line.

**Notes.** Full-rank (invertible) matrices have only the trivial null space `{0}`. → Rank and column space, Systems of linear equations and the matrix inverse.

---

## Products, Bases, and Eigen-things *(3Blue1Brown)*

### Dot product

**Definition.** For vectors `a, b`, the **dot product** is `a·b = Σ aᵢbᵢ = |a||b|cos θ`, where `θ` is the angle between them. The result is a single number (a scalar).

**Intuition.** It measures **alignment**: positive when the vectors point the same general way, zero when perpendicular, negative when opposing. Geometrically it's the length of `a`'s projection onto `b`, scaled by `|b|`.

**Example.** `[3,4]·[2,1] = 3·2 + 4·1 = 10`; `[1,0]·[0,1] = 0` (perpendicular).

**Notes.** The workhorse of ML: it's the operation inside every matrix multiply, cosine similarity, and attention score. → Matrix multiplication (composition), Self-attention (ML).

### Cross product

**Definition.** In 3D, `a × b` is a **vector** perpendicular to both, whose length `|a||b|sin θ` equals the **area of the parallelogram** they span. Compute it as a symbolic determinant `det[[î, ĵ, k̂], [a₁, a₂, a₃], [b₁, b₂, b₃]]`.

**Intuition.** Length encodes the parallelogram's area; direction (given by the **right-hand rule**) encodes orientation. **Order matters**: `a × b = −(b × a)` (anti-commutative) — swapping the inputs flips the result.

**Example.** `[1,0,0] × [0,1,0] = [0,0,1]` (x cross y points along +z by the right-hand rule). In 2D the analogous signed area is `a₁b₂ − a₂b₁ = det[[a₁,b₁],[a₂,b₂]]`.

**Notes.** The `î, ĵ, k̂` determinant is a mnemonic for the formula, not a literal determinant. → Determinant.

### Cramer's rule

**Definition.** Solves a square system `A·x = b` (`det A ≠ 0`) one coordinate at a time with determinants: `xᵢ = det(Aᵢ) / det(A)`, where `Aᵢ` is `A` with its `i`-th column replaced by `b`.

**Intuition.** Swapping in `b` and taking a determinant ratio isolates each coordinate through signed-area/volume scaling. Elegant, but slow — Gaussian elimination is far faster for real systems.

**Example.** `A = [[2,1],[1,3]]`, `b = [3,5]`, `det A = 5`. `x₁ = det[[3,1],[5,3]]/5 = (9−5)/5 = 0.8`; `x₂ = det[[2,3],[1,5]]/5 = (10−3)/5 = 1.4`.

**Notes.** Mainly of geometric/theoretical value; impractical for large `n`. → Determinant, Systems of linear equations and the matrix inverse.

### Change of basis

**Definition.** Translating a vector's coordinates from one basis to another. If `B`'s columns are an alternate basis written in standard coordinates, then `B·[v]_B = [v]_standard`, and `B⁻¹·[v]_standard = [v]_B`. A transformation `A` becomes `B⁻¹ A B` when expressed in the new basis.

**Intuition.** The same arrow has different coordinate lists depending on the "language" (basis) describing it. `B` translates *from* the alternate language *to* standard; `B⁻¹` translates back. The `B⁻¹AB` sandwich means "translate in, apply the transformation, translate out."

**Example.** With alternate basis `[2,1]` and `[−1,1]` (the columns of `B`), a vector with alternate coordinates `[1,1]` is `B·[1,1] = [2 − 1, 1 + 1] = [1, 2]` in standard coordinates.

**Notes.** Choosing an *eigenbasis* makes a transformation diagonal (pure scaling) — the cleanest possible coordinates. → Eigenvectors and eigenvalues, Basis vectors and unit vectors.

### Eigenvectors and eigenvalues

**Definition.** An **eigenvector** of `A` is a nonzero vector that stays on its own span under the transformation — it only gets **stretched or squished**, never knocked off its line: `A·v = λ·v`. The scalar `λ` is its **eigenvalue** (the stretch factor). Find them by solving `det(A − λI) = 0` for the eigenvalues, then `(A − λI)v = 0` for each eigenvector.

**Intuition.** Most vectors get rotated off their span by a transformation; eigenvectors are the special axes that merely scale, revealing the transformation's "natural axes." `det(A − λI) = 0` is required because `(A − λI)` must squish `v` to zero (be non-invertible) for a nonzero solution to exist.

**Example (2×2).** `A = [[2,1],[0,3]]`. `det(A − λI) = (2−λ)(3−λ) = 0` → `λ = 2` or `3`. For `λ = 2`: `(A − 2I) = [[0,1],[0,1]]` ⇒ `y = 0` ⇒ `v = [1, 0]`. For `λ = 3`: `(A − 3I) = [[−1,1],[0,0]]` ⇒ `y = x` ⇒ `v = [1, 1]`. So `A` stretches `[1,0]` by 2 and `[1,1]` by 3.

**Notes.** A 3×3 gives a cubic in `λ` (up to three eigenvalues), same recipe. A full basis of eigenvectors *diagonalizes* `A` (→ change of basis), making powers `Aⁿ` trivial. In ML they underpin PCA (the top eigenvectors of the covariance matrix). → Change of basis, Determinant, Principal components analysis (PCA) (ML).

---

## Calculus foundations *(3Blue1Brown)*

### Derivatives

**Definition.** The **derivative** `f'(x)` is the instantaneous rate of change — the slope of the tangent line: `f'(x) = lim_{h→0} (f(x+h) − f(x)) / h`. Core rules: power `d/dx xⁿ = n·xⁿ⁻¹`; sum (term by term); product `(fg)' = f'g + fg'`; chain `(f(g(x)))' = f'(g(x))·g'(x)`.

**Intuition.** Zoom into a curve until it looks straight; the derivative is that local slope — how fast the output changes per tiny nudge of the input. The **chain rule** (a nudge propagating through composed functions) is exactly what backpropagation runs.

**Example.** `f(x) = 3x²` → `f'(x) = 6x`, so at `x = 2` the curve rises 12 per unit of `x`. `d/dx sin(x²) = cos(x²)·2x` (chain rule).

**Notes.** Handy standards: `d/dx eˣ = eˣ`, `d/dx ln x = 1/x`, `d/dx sin x = cos x`. → Taylor series, Backpropagation (ML), Gradient descent (ML).

### Integrals and area under the curve

**Definition.** The **definite integral** `∫ₐᵇ f(x) dx` is the (signed) **area under the curve** from `a` to `b`. The **Fundamental Theorem of Calculus**: if `F' = f`, then `∫ₐᵇ f dx = F(b) − F(a)` — integration undoes differentiation.

**Intuition.** Chop the region into infinitely many thin rectangles of width `dx` and add them up. Finding an antiderivative `F` (a function whose slope is `f`) gives that area exactly, with no summing.

**Example.** `∫₀² 3x² dx = [x³]₀² = 8 − 0 = 8` (since `d/dx x³ = 3x²`). Area below the axis counts as negative.

**Notes.** Basic antiderivatives: `∫ xⁿ dx = xⁿ⁺¹/(n+1)` (n ≠ −1), `∫ 1/x dx = ln|x|`, `∫ eˣ dx = eˣ`. Underlies probability densities, where area = probability. → Derivatives, Probability distributions (stat).

### Taylor series

**Definition.** Approximates a smooth function near a point `a` by a polynomial built from its derivatives: `f(x) ≈ f(a) + f'(a)(x−a) + f''(a)/2!·(x−a)² + f'''(a)/3!·(x−a)³ + …`.

**Intuition.** Match the function's value, then its slope, then its curvature, then higher bends — each term corrects the previous fit, so more terms means a better local approximation. Near `a`, a handful of terms is often plenty.

**Example.** Around `a = 0`: `eˣ ≈ 1 + x + x²/2 + x³/6 + …` and `cos x ≈ 1 − x²/2 + x⁴/24 − …`. At `x = 0.1`, `e^0.1 ≈ 1 + 0.1 + 0.005 = 1.105` (true value 1.1052).

**Notes.** The `n!` denominators fall out of repeatedly differentiating the polynomial. The first-order Taylor term is how optimization "linearizes" a function locally. → Derivatives.

---

## Glossary

- **Basis** — independent vectors whose combinations span the space; the coordinate "ruler." → [Basis vectors and unit vectors](#basis-vectors-and-unit-vectors).
- **Change of basis** — translate coordinates/transformations between bases via `B` and `B⁻¹`. → [Change of basis](#change-of-basis).
- **Column space** — the span of a matrix's columns (all its outputs). → [Rank and column space](#rank-and-column-space).
- **Cramer's rule** — solve `Ax=b` via ratios of determinants. → [Cramer's rule](#cramers-rule).
- **Cross product** — 3D perpendicular vector; length = parallelogram area; right-hand rule. → [Cross product](#cross-product).
- **Derivative** — instantaneous slope; `lim (f(x+h)−f(x))/h`. → [Derivatives](#derivatives).
- **Determinant** — area/volume scaling factor of a transformation; 0 ⇒ singular. → [Determinant](#determinant).
- **Dot product** — `Σ aᵢbᵢ = |a||b|cos θ`; measures alignment. → [Dot product](#dot-product).
- **Eigenvector / eigenvalue** — vector that only scales under `A` (`Av = λv`); `λ` is the factor. → [Eigenvectors and eigenvalues](#eigenvectors-and-eigenvalues).
- **Integral** — signed area under a curve; the inverse of the derivative. → [Integrals and area under the curve](#integrals-and-area-under-the-curve).
- **Linear combination** — `a·v + b·w` of vectors. → [Linear combinations and span](#linear-combinations-and-span).
- **Linear dependence / independence** — redundant vs genuinely-new directions. → [Linear dependence and independence](#linear-dependence-and-independence).
- **Linear transformation** — keeps lines straight and the origin fixed; set by where the basis lands. → [Linear transformations](#linear-transformations).
- **Matrix** — stores a linear transformation; its columns are the images of the basis vectors. → [Matrices as linear transformations](#matrices-as-linear-transformations).
- **Matrix inverse** — undoes a transformation; `x = A⁻¹b`; exists iff `det ≠ 0`. → [Systems of linear equations and the matrix inverse](#systems-of-linear-equations-and-the-matrix-inverse).
- **Matrix multiplication** — composition of transformations (applied right-to-left). → [Matrix multiplication (composition)](#matrix-multiplication-composition).
- **Null space (kernel)** — vectors mapped to 0; the solutions of `Ax=0`. → [Null space (kernel)](#null-space-kernel).
- **Rank** — dimension of the column space (surviving output directions). → [Rank and column space](#rank-and-column-space).
- **Span** — all linear combinations of a set of vectors. → [Linear combinations and span](#linear-combinations-and-span).
- **Taylor series** — polynomial approximation of a function from its derivatives. → [Taylor series](#taylor-series).
- **Unit vector** — a length-1 vector; the standard basis is `î, ĵ`. → [Basis vectors and unit vectors](#basis-vectors-and-unit-vectors).
- **Vector** — an arrow (length + direction) / an ordered list of numbers. → [Vectors](#vectors).