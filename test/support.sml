(* support.sml -- shared helpers for the sml-hmm tests.

   Every HMM quantity (likelihoods, log-probabilities, re-estimated transition
   and emission entries) is floating point, so comparisons go through an
   explicit epsilon (`approx`) rather than string or structural equality:
   `Real.toString` differs between MLton and Poly/ML, and the closed-form
   expectations only match up to rounding. A loose `eps` (1e-9) pins the
   algebraic identities; `approxTol` lets a caller relax it for accumulated
   floating-point drift. *)

structure Support =
struct
  val eps = 1E~9

  fun approx (a, b) = Real.abs (a - b) <= eps

  (* approx with a caller-supplied tolerance. *)
  fun approxTol tol (a, b) = Real.abs (a - b) <= tol

  fun checkApprox name (expected, actual) =
    Harness.check name (approx (expected, actual))

  fun checkApproxTol tol name (expected, actual) =
    Harness.check name (approxTol tol (expected, actual))

  (* Each row of a stochastic matrix should sum to 1 within tolerance. *)
  fun rowSum xs = List.foldl (fn (x, acc) => x + acc) 0.0 xs

  fun checkRowsNormalized tol name rows =
    Harness.check name
      (List.all (fn r => approxTol tol (rowSum r, 1.0)) rows)
end
