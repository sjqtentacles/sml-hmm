(* test_baumwelch.sml -- Baum-Welch (EM) re-estimation.

   Baum-Welch iteratively re-estimates pi, A and B to (locally) maximise the
   likelihood of a set of training observation sequences. The two guarantees we
   pin down:

   1. Monotonicity: the total log-likelihood of the training data does not
      decrease from one iteration to the next (EM is monotone; we allow a tiny
      negative tolerance for floating-point drift). We check this by running
      Baum-Welch for k and k+1 iterations and comparing data log-likelihoods.

   2. Normalization: after re-estimation pi sums to 1, every row of A sums to 1,
      and every row of B sums to 1 (all within tolerance) -- the re-estimated
      model is still a valid HMM.

   We start from a deliberately mediocre, slightly-asymmetric initial model so
   that EM has somewhere to climb, and train on a fixed bank of sequences. The
   iteration count is fixed for determinism, so MLton and Poly/ML converge to
   the same numbers. *)

structure BaumWelchTests =
struct
  open Support
  structure H = Hmm

  val tol  = 1E~9
  val ltol = 1E~7   (* looser tolerance for accumulated EM arithmetic *)

  (* A 2-state, 2-symbol starting model, intentionally not yet fitted. *)
  val init0 =
    H.make
      { numStates = 2
      , numObs    = 2
      , init      = [0.5, 0.5]
      , trans     = [[0.6, 0.4], [0.4, 0.6]]
      , emit      = [[0.7, 0.3], [0.3, 0.7]] }

  (* Training data: sequences with clear structure (long runs of one symbol
     then the other) for EM to latch onto. *)
  val training =
    [ [0, 0, 0, 1, 1, 1]
    , [0, 0, 1, 1, 1, 1]
    , [1, 1, 1, 0, 0, 0]
    , [0, 0, 0, 0, 1, 1]
    , [1, 1, 0, 0, 0, 0] ]

  (* Total log-likelihood of the training bank under a model. *)
  fun dataLogLik model =
    List.foldl (fn (obs, acc) => acc + H.forward model obs) 0.0 training

  fun trainedFor k = H.baumWelch init0 training { iters = k }

  fun run () =
    let
      val () = Harness.section "baum-welch: data log-likelihood is non-decreasing"
      val lik0 = dataLogLik init0
      val lik1 = dataLogLik (trainedFor 1)
      val lik2 = dataLogLik (trainedFor 2)
      val lik5 = dataLogLik (trainedFor 5)
      val lik10 = dataLogLik (trainedFor 10)
      val () = Harness.check "L(1 iter)  >= L(0 iter)"  (lik1  >= lik0  - ltol)
      val () = Harness.check "L(2 iter)  >= L(1 iter)"  (lik2  >= lik1  - ltol)
      val () = Harness.check "L(5 iter)  >= L(2 iter)"  (lik5  >= lik2  - ltol)
      val () = Harness.check "L(10 iter) >= L(5 iter)"  (lik10 >= lik5  - ltol)
      val () = Harness.check "training strictly improved overall" (lik10 > lik0)

      val () = Harness.section "baum-welch: re-estimated model stays a valid HMM"
      val fitted = trainedFor 10
      val () = Harness.checkInt "numStates preserved" (2, H.numStates fitted)
      val () = Harness.checkInt "numObs preserved" (2, H.numObs fitted)
      val () = checkApproxTol ltol "pi sums to 1" (1.0, rowSum (H.init fitted))
      val () = checkRowsNormalized ltol "every A row sums to 1" (H.trans fitted)
      val () = checkRowsNormalized ltol "every B row sums to 1" (H.emit fitted)

      val () = Harness.section "baum-welch: probabilities stay in [0,1]"
      val allEntries =
        H.init fitted
        @ List.concat (H.trans fitted)
        @ List.concat (H.emit fitted)
      val () = Harness.check "all probabilities >= 0"
                 (List.all (fn x => x >= ~tol) allEntries)
      val () = Harness.check "all probabilities <= 1"
                 (List.all (fn x => x <= 1.0 + tol) allEntries)

      val () = Harness.section "baum-welch: determinism (0 iters is a no-op)"
      val noop = trainedFor 0
      val () = checkApproxTol tol "0-iter pi[0] unchanged"
                 (List.nth (H.init init0, 0), List.nth (H.init noop, 0))
      val () = checkApproxTol tol "0-iter data log-lik unchanged"
                 (lik0, dataLogLik noop)
    in
      ()
    end
end
