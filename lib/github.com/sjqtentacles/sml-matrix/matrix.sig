(* matrix.sig

   Dense real linear algebra over general m x n matrices, in pure Standard ML
   (Basis library only). This complements the fixed 2/3/4-dimensional vector
   and matrix types of a graphics library by handling arbitrary dimensions.

   Storage: a value of type `t` is a dense matrix stored as a flat
   `real array` in row-major order (entry (i, j) lives at index i * cols + j),
   together with its row and column counts. The type is opaque; build values
   with `fromRows`, `make`, `identity`, or `zeros`.

   Conventions:
   - Rows and columns are zero-indexed in `sub`/`update`.
   - All operations are pure unless noted; `update` mutates in place and is the
     only mutating operation (it returns unit).
   - Shape errors (e.g. adding mismatched matrices, multiplying with
     incompatible inner dimensions, decomposing a non-square matrix) raise
     `Dim` with a human-readable message.
   - `solve` and `inv` raise `Singular` when the matrix has no inverse (a zero
     pivot is encountered during elimination, within tolerance).
   - Real comparisons are the caller's concern; tests use an absolute
     tolerance. `eps` exposes the tolerance used internally for pivoting and
     singularity detection. *)

signature MATRIX =
sig
  type t

  exception Dim of string
  exception Singular

  (* Tolerance used for pivot / singularity decisions. *)
  val eps : real

  (* --- construction --- *)

  (* Build from a list of rows. All rows must have equal, non-zero length;
     otherwise raise Dim. An empty list raises Dim. *)
  val fromRows : real list list -> t
  (* make (r, c) x : an r x c matrix with every entry equal to x. r, c > 0. *)
  val make     : int * int -> real -> t
  (* zeros (r, c) : an r x c matrix of zeros. *)
  val zeros    : int * int -> t
  (* identity n : the n x n identity matrix. *)
  val identity : int -> t

  (* --- shape and elements --- *)

  val rows   : t -> int
  val cols   : t -> int
  (* sub (m, i, j) : entry at row i, column j (zero-indexed). Raises Dim if
     out of range. *)
  val sub    : t * int * int -> real
  (* update (m, i, j, x) : set entry (i, j) to x in place. Raises Dim if out
     of range. *)
  val update : t * int * int * real -> unit
  (* The rows of the matrix as a list of lists. *)
  val toRows : t -> real list list

  (* --- arithmetic --- *)

  (* Elementwise sum / difference; operands must share dimensions. *)
  val add       : t * t -> t
  val sub'      : t * t -> t
  val scale     : real -> t -> t
  (* Matrix product (a x b); cols a must equal rows b, else Dim. *)
  val mul       : t * t -> t
  val transpose : t -> t

  (* --- decompositions and solves --- *)

  (* LU decomposition with partial (row) pivoting of a square matrix A.
     Returns {l, u, p, sign} with:
       - l : unit lower-triangular (1s on the diagonal),
       - u : upper-triangular,
       - p : a permutation given as an int array where p[i] is the original
             row index now occupying row i, so that P*A = L*U,
       - sign : the determinant sign of P (+1.0 or ~1.0).
     Raises Dim if A is not square. A singular A still decomposes (U may have a
     zero on its diagonal); singularity is reported by `det`/`solve`/`inv`. *)
  val lu : t -> {l : t, u : t, p : int array, sign : real}

  (* Determinant via LU: sign * product of U's diagonal. Square only. *)
  val det : t -> real

  (* Solve A x = b for square A. b is given as a real list of length rows A;
     returns x as a real list. Raises Dim on shape mismatch, Singular if A is
     singular. *)
  val solve : t -> real list -> real list

  (* Inverse of a square matrix (via solving against the identity columns).
     Raises Singular if A is not invertible. *)
  val inv : t -> t

  (* QR decomposition (classical Gram-Schmidt with one reorthogonalisation
     pass for numerical stability) of an m x n matrix with m >= n and full
     column rank. Returns {q, r} where q is m x n with orthonormal columns, r
     is n x n upper-triangular, and q * r = A. Raises Dim if m < n, Singular
     if the columns are linearly dependent. *)
  val qr : t -> {q : t, r : t}

  (* --- scalar summaries and norms --- *)

  (* Sum of the diagonal of a square matrix. Raises Dim if not square. *)
  val trace : t -> real

  (* Maximum absolute column sum (the induced 1-norm). *)
  val norm1 : t -> real
  (* Maximum absolute row sum (the induced infinity-norm). *)
  val normInf : t -> real
  (* Frobenius norm: the square root of the sum of squared entries. *)
  val normFro : t -> real
  (* Spectral norm (the induced 2-norm) = the largest singular value. *)
  val norm2 : t -> real

  (* 2-norm condition number: the ratio of the largest to the smallest
     singular value. Real.posInf for a (numerically) rank-deficient matrix. *)
  val cond : t -> real

  (* --- further decompositions --- *)

  (* Cholesky factor of a symmetric positive-definite matrix: the lower-
     triangular L (with positive diagonal) such that A = L * Lᵀ. Raises Dim if
     A is not square, Singular if A is not (numerically) positive definite.
     Only the lower triangle of A is read. *)
  val cholesky : t -> t

  (* Singular value decomposition via one-sided Jacobi rotations. For an m x n
     matrix and k = min (m, n), returns {u, s, vt} where u is m x k with
     orthonormal columns, s is a length-k array of singular values in
     descending order, and vt is the k x n matrix Vᵀ, so that
     A = u * diag(s) * vt. *)
  val svd : t -> {u : t, s : real array, vt : t}

  (* Symmetric eigendecomposition via cyclic (one-sided) Jacobi rotations. A
     must be square and (numerically) symmetric. Returns {values, vectors}
     where values holds the eigenvalues in ascending order and the columns of
     vectors are the corresponding orthonormal eigenvectors, so that
     A * vectors = vectors * diag(values). Raises Dim if A is not square or not
     symmetric. *)
  val eigSym : t -> {values : real array, vectors : t}

  (* Least-squares solution of an overdetermined system: the x that minimises
     ‖A x - b‖₂ for an m x n matrix A with m >= n and full column rank, solved
     via QR. b is given as a list of length rows A; the result has length
     cols A. Raises Dim on a shape mismatch, Singular if A is rank-deficient. *)
  val lstsq : t -> real list -> real list

  (* Moore-Penrose pseudoinverse (n x m) via the SVD; defined for any shape and
     rank. Reduces to `inv` for a square invertible matrix. *)
  val pinv : t -> t

  (* Numerical rank: the number of singular values exceeding a tolerance scaled
     by the largest singular value and the matrix dimensions. *)
  val rank : t -> int
end
