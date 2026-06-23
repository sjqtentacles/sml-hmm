(* test_forward_backward.sml -- the forward and backward likelihood algorithms.

   A classic tiny weather/umbrella HMM with two hidden states and three
   observation symbols:

     states : 0 = Rainy, 1 = Sunny
     obs    : 0 = walk,  1 = shop, 2 = clean
     init   pi = [0.6, 0.4]
     trans  A  = [[0.7, 0.3],
                  [0.4, 0.6]]
     emit   B  = [[0.1, 0.4, 0.5],   (* Rainy *)
                  [0.6, 0.3, 0.1]]   (* Sunny *)

   Closed-form likelihoods (computed by hand from the forward recursion):

     P([0])    = pi . B(:,0) = 0.6*0.1 + 0.4*0.6 = 0.30
     P([0,2])  = 0.069 + 0.0162 = 0.0852    (see the README derivation)

   `forward` and `backward` are exposed in log-space; exponentiating recovers
   the true probabilities. The defining identity is that the two agree on the
   total likelihood of any observation sequence, which we check with a loose
   tolerance (never string equality on reals). *)

structure ForwardBackwardTests =
struct
  open Support
  structure H = Hmm

  val tol = 1E~12

  val model =
    H.make
      { numStates = 2
      , numObs    = 3
      , init      = [0.6, 0.4]
      , trans     = [[0.7, 0.3], [0.4, 0.6]]
      , emit      = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]] }

  (* forward/backward are log-likelihoods; convert to plain probability. *)
  fun lik obs = Math.exp (H.forward model obs)

  fun run () =
    let
      val () = Harness.section "accessors round-trip the constructor inputs"
      val () = Harness.checkInt "numStates" (2, H.numStates model)
      val () = Harness.checkInt "numObs" (3, H.numObs model)
      val () = checkApprox "init[0]" (0.6, List.nth (H.init model, 0))
      val () = checkApprox "init[1]" (0.4, List.nth (H.init model, 1))
      val () = checkApprox "trans[0][1]" (0.3, List.nth (List.nth (H.trans model, 0), 1))
      val () = checkApprox "emit[1][0]" (0.6, List.nth (List.nth (H.emit model, 1), 0))

      val () = Harness.section "forward: closed-form likelihoods"
      val () = checkApproxTol tol "P([0]) = 0.30" (0.30, lik [0])
      val () = checkApproxTol tol "P([0,2]) = 0.0852" (0.0852, lik [0, 2])

      val () = Harness.section "forward and backward agree on the total likelihood"
      val seqs = [[0], [1], [2], [0, 2], [1, 0, 2], [2, 2, 1, 0, 0, 1]]
      val () =
        List.app
          (fn obs =>
             checkApproxTol tol
               ("fwd = bwd on " ^ Int.toString (List.length obs) ^ "-obs seq")
               (H.forward model obs, H.backward model obs))
          seqs

      val () = Harness.section "likelihood lies in [0,1] and longer sequences are rarer"
      val () = Harness.check "P([0]) <= 1" (lik [0] <= 1.0 + tol)
      val () = Harness.check "P([0,2]) <= P([0])" (lik [0, 2] <= lik [0])
      val () = Harness.check "P([0,2]) > 0" (lik [0, 2] > 0.0)
    in
      ()
    end
end
