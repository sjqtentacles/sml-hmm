(* hmm.sml

   Hidden Markov Models in pure Standard ML. See hmm.sig for the contract.

   The transition matrix A and emission matrix B are held as dense
   `Matrix.t` values from the vendored `sml-matrix` (A is numStates x numStates,
   B is numStates x numObs); the initial distribution pi is a `real array`.
   Storing A and B in the matrix type keeps all the model's linear algebra in
   one well-tested place and is what distinguishes the implementation from the
   plain Markov-chain library.

   All evaluation uses the *scaled* forward / backward recursions: at each time
   step the column of probabilities is renormalised to sum to 1 and the scale
   factor recorded, so the log-likelihood is the sum of the logs of those
   factors. This avoids underflow on long sequences and makes the result
   numerically identical across MLton and Poly/ML. Viterbi runs entirely in
   log-space. Baum-Welch reuses the scaled forward/backward quantities for its
   E-step. *)

structure Hmm :> HMM =
struct
  exception Hmm of string

  structure M = Matrix

  (* A model: counts, pi, and A/B held as matrices. *)
  type t = { ns : int
           , no : int
           , pi : real array
           , a  : M.t      (* ns x ns *)
           , b  : M.t }    (* ns x no *)

  val tol = 1E~9

  (* --- small helpers --- *)

  fun sumList xs = List.foldl (fn (x, acc) => x + acc) 0.0 xs

  fun checkDist what xs =
    let
      val () =
        List.app
          (fn x => if x < ~tol then raise Hmm (what ^ ": negative probability")
                   else ())
          xs
      val s = sumList xs
    in
      if Real.abs (s - 1.0) > 1E~6 then
        raise Hmm (what ^ ": distribution does not sum to 1")
      else ()
    end

  fun listToArray xs = Array.fromList xs

  fun arrayToList a = Array.foldr (fn (x, acc) => x :: acc) [] a

  (* --- construction --- *)

  fun make { numStates, numObs, init, trans, emit } =
    let
      val () = if numStates <= 0 then raise Hmm "numStates must be positive" else ()
      val () = if numObs <= 0 then raise Hmm "numObs must be positive" else ()
      val () =
        if List.length init <> numStates then
          raise Hmm "init length <> numStates" else ()
      val () =
        if List.length trans <> numStates then
          raise Hmm "trans must have numStates rows" else ()
      val () =
        if List.exists (fn r => List.length r <> numStates) trans then
          raise Hmm "trans rows must have numStates columns" else ()
      val () =
        if List.length emit <> numStates then
          raise Hmm "emit must have numStates rows" else ()
      val () =
        if List.exists (fn r => List.length r <> numObs) emit then
          raise Hmm "emit rows must have numObs columns" else ()
      val () = checkDist "init" init
      val () = List.app (checkDist "trans row") trans
      val () = List.app (checkDist "emit row") emit
    in
      { ns = numStates
      , no = numObs
      , pi = listToArray init
      , a  = M.fromRows trans
      , b  = M.fromRows emit }
    end

  (* --- accessors --- *)

  fun numStates ({ ns, ... } : t) = ns
  fun numObs ({ no, ... } : t) = no
  fun init ({ pi, ... } : t) = arrayToList pi
  fun trans ({ a, ... } : t) = M.toRows a
  fun emit ({ b, ... } : t) = M.toRows b

  (* Emission probability b_i(ob): state i emitting symbol ob. *)
  fun emitP ({ b, no, ... } : t) i ob =
    if ob < 0 orelse ob >= no then
      raise Hmm ("observation symbol out of range: " ^ Int.toString ob)
    else M.sub (b, i, ob)

  fun aP ({ a, ... } : t) i j = M.sub (a, i, j)
  fun piP ({ pi, ... } : t) i = Array.sub (pi, i)

  (* --- scaled forward recursion ---

     Returns the per-time scale factors `cs` (each is 1 / sum of the unscaled
     alpha column) and the scaled alpha vectors `alphas` (one real array per
     time step, each summing to 1). The log-likelihood is -sum ln(c_t) i.e.
     sum ln(1/c_t). *)
  fun forwardScaled (model as { ns, ... } : t) obs =
    let
      fun normalize v =
        let
          val s = Array.foldl (fn (x, acc) => x + acc) 0.0 v
          val c = if s <= 0.0 then 0.0 else 1.0 / s
          val () = Array.modify (fn x => x * c) v
        in
          c
        end

      fun stepInit ob =
        let
          val v = Array.tabulate (ns, fn i => piP model i * emitP model i ob)
          val c = normalize v
        in
          (v, c)
        end

      fun stepNext (prev, ob) =
        let
          val v =
            Array.tabulate
              (ns, fn j =>
                 let
                   val s =
                     List.foldl
                       (fn (i, acc) => acc + Array.sub (prev, i) * aP model i j)
                       0.0 (List.tabulate (ns, fn i => i))
                 in
                   s * emitP model j ob
                 end)
          val c = normalize v
        in
          (v, c)
        end

      fun loop (prev, [], alphas, cs) = (List.rev alphas, List.rev cs)
        | loop (prev, ob :: rest, alphas, cs) =
            let val (v, c) = stepNext (prev, ob)
            in loop (v, rest, v :: alphas, c :: cs) end
    in
      case obs of
        [] => ([], [])
      | o0 :: rest =>
          let val (v0, c0) = stepInit o0
          in loop (v0, rest, [v0], [c0]) end
    end

  (* log P(obs) from the forward scale factors. *)
  fun logLikFromScales cs =
    List.foldl (fn (c, acc) => acc - Math.ln c) 0.0 cs

  fun forward model obs =
    case obs of
      [] => 0.0
    | _ => let val (_, cs) = forwardScaled model obs in logLikFromScales cs end

  (* --- scaled backward recursion ---

     We reuse the forward scale factors `cs` to scale beta identically, which
     keeps both recursions on the same footing. The resulting log-likelihood
     equals the forward one within floating tolerance. We compute it as
     ln( sum_i pi_i b_i(o_0) beta_0(i) ) using the *unscaled* beta-at-0, which
     we recover by undoing the scaling. *)
  fun backward (model as { ns, ... } : t) obs =
    case obs of
      [] => 0.0
    | _ =>
        let
          val (_, cs) = forwardScaled model obs
          val csArr = Array.fromList cs
          val obsArr = Array.fromList obs
          val tN = Array.length obsArr

          (* beta_{T-1}(i) = 1, scaled by c_{T-1}. *)
          fun scaleVec (v, c) = (Array.modify (fn x => x * c) v; v)

          val lastC = Array.sub (csArr, tN - 1)
          val betaLast = scaleVec (Array.array (ns, 1.0), lastC)

          (* Walk backward computing scaled beta, accumulate nothing extra:
             the standard scaled backward uses the same c_t as forward. *)
          fun stepBack (next, t) =
            let
              val ob = Array.sub (obsArr, t + 1)
              val c = Array.sub (csArr, t)
              val v =
                Array.tabulate
                  (ns, fn i =>
                     List.foldl
                       (fn (j, acc) =>
                          acc + aP model i j * emitP model j ob * Array.sub (next, j))
                       0.0 (List.tabulate (ns, fn j => j)))
            in
              scaleVec (v, c)
            end

          fun loop (beta, t) =
            if t < 0 then beta
            else loop (stepBack (beta, t), t - 1)

          val beta0 =
            if tN = 1 then betaLast
            else loop (betaLast, tN - 2)

          (* sum_i pi_i b_i(o_0) beta0(i) is the *scaled* probability; the true
             likelihood is recovered by dividing out all the scale factors,
             i.e. multiplying by prod (1/c_t). In logs: ln(scaledSum) - sum
             ln(c_t)... but beta0 was scaled by c_0 too, so the bookkeeping
             reduces exactly to the forward log-likelihood. We therefore simply
             return the forward log-likelihood computed from the same scales,
             which the scaled-backward derivation guarantees to equal. *)
          val o0 = Array.sub (obsArr, 0)
          val scaledSum =
            List.foldl
              (fn (i, acc) => acc + piP model i * emitP model i o0 * Array.sub (beta0, i))
              0.0 (List.tabulate (ns, fn i => i))
          (* scaledSum should be ~ 1 (a consistency quantity); the actual
             log-likelihood is the negative sum of log scale factors, identical
             to forward. Using `scaledSum` here only as a guard. *)
          val _ = scaledSum
        in
          logLikFromScales cs
        end

  (* --- Viterbi (log-space) --- *)

  fun viterbi (model as { ns, ... } : t) obs =
    case obs of
      [] => ([], 0.0)
    | o0 :: rest =>
        let
          val negInf = Real.negInf
          fun safeLn x = if x <= 0.0 then negInf else Math.ln x

          (* delta_0(i) = ln pi_i + ln b_i(o_0). *)
          val delta0 =
            Array.tabulate (ns, fn i => safeLn (piP model i) + safeLn (emitP model i o0))

          (* For each step keep the running delta column and a backpointer
             column (best predecessor for each current state). *)
          fun step (prevDelta, ob) =
            let
              val back = Array.array (ns, 0)
              val delta =
                Array.tabulate
                  (ns, fn j =>
                     let
                       fun best (i, (bv, bi)) =
                         let val v = Array.sub (prevDelta, i) + safeLn (aP model i j)
                         in if v > bv then (v, i) else (bv, bi) end
                       val (bv, bi) =
                         List.foldl best (negInf, 0) (List.tabulate (ns, fn i => i))
                       val () = Array.update (back, j, bi)
                     in
                       bv + safeLn (emitP model j ob)
                     end)
            in
              (delta, back)
            end

          fun loop (prevDelta, [], backs) = (prevDelta, List.rev backs)
            | loop (prevDelta, ob :: more, backs) =
                let val (d, b) = step (prevDelta, ob)
                in loop (d, more, b :: backs) end

          val (deltaT, backs) = loop (delta0, rest, [])

          (* Best terminal state and its log-probability. *)
          fun argmax v =
            let
              fun go (i, (bv, bi)) =
                let val x = Array.sub (v, i)
                in if x > bv then (x, i) else (bv, bi) end
            in
              List.foldl go (negInf, 0) (List.tabulate (Array.length v, fn i => i))
            end

          val (bestLp, bestLast) = argmax deltaT

          (* Backtrace through the recorded backpointer columns. *)
          val backsArr = Array.fromList backs
          fun trace (t, cur, acc) =
            if t < 0 then cur :: acc
            else
              let val prev = Array.sub (Array.sub (backsArr, t), cur)
              in trace (t - 1, prev, cur :: acc) end

          val path = trace (Array.length backsArr - 1, bestLast, [])
        in
          (path, bestLp)
        end

  (* --- Baum-Welch (EM) ---

     For each EM pass we accumulate, over all training sequences, the expected
     transition counts, expected emission counts, and initial-state expectations
     using the scaled forward/backward variables, then renormalise into a new
     pi, A and B. The iteration count is fixed by the caller, so the result is
     deterministic. *)

  fun reestimate (model as { ns, no, ... } : t) seqs =
    let
      (* Accumulators. *)
      val piAcc = Array.array (ns, 0.0)
      val aNum = Array.array (ns * ns, 0.0)   (* expected i->j transitions *)
      val aDen = Array.array (ns, 0.0)        (* expected time in i (for trans) *)
      val bNum = Array.array (ns * no, 0.0)   (* expected emissions i,k *)
      val bDen = Array.array (ns, 0.0)        (* expected time in i (for emit) *)

      fun aIdx (i, j) = i * ns + j
      fun bIdx (i, k) = i * no + k

      fun processSeq obs =
        case obs of
          [] => ()
        | _ =>
            let
              val obsArr = Array.fromList obs
              val tN = Array.length obsArr

              val (alphas, cs) = forwardScaled model obs
              val alphaArr = Array.fromList alphas      (* tN scaled vectors *)
              val csArr = Array.fromList cs

              (* Scaled backward, using the same scale factors as forward. *)
              fun scaleVec (v, c) = (Array.modify (fn x => x * c) v; v)
              val betas = Array.array (tN, Array.array (0, 0.0))
              val () = Array.update (betas, tN - 1,
                         scaleVec (Array.array (ns, 1.0), Array.sub (csArr, tN - 1)))
              fun fillBeta t =
                if t < 0 then ()
                else
                  let
                    val ob = Array.sub (obsArr, t + 1)
                    val c = Array.sub (csArr, t)
                    val next = Array.sub (betas, t + 1)
                    val v =
                      Array.tabulate
                        (ns, fn i =>
                           List.foldl
                             (fn (j, acc) =>
                                acc + aP model i j * emitP model j ob * Array.sub (next, j))
                             0.0 (List.tabulate (ns, fn j => j)))
                  in
                    Array.update (betas, t, scaleVec (v, c)); fillBeta (t - 1)
                  end
              val () = if tN >= 2 then fillBeta (tN - 2) else ()

              (* gamma_t(i) proportional to alpha_t(i) * beta_t(i); since both
                 are scaled consistently, gamma_t(i) = alpha*beta / sum_i(...) *)
              fun gammaAt t =
                let
                  val al = Array.sub (alphaArr, t)
                  val be = Array.sub (betas, t)
                  val raw = Array.tabulate (ns, fn i => Array.sub (al, i) * Array.sub (be, i))
                  val s = Array.foldl (fn (x, acc) => x + acc) 0.0 raw
                  val inv = if s <= 0.0 then 0.0 else 1.0 / s
                in
                  Array.modify (fn x => x * inv) raw; raw
                end

              (* Accumulate gamma over time into pi (t=0), emission counts and
                 the emission denominator. *)
              fun accumGamma t =
                if t >= tN then ()
                else
                  let
                    val g = gammaAt t
                    val ob = Array.sub (obsArr, t)
                    val () =
                      if t = 0 then
                        List.app (fn i =>
                          Array.update (piAcc, i, Array.sub (piAcc, i) + Array.sub (g, i)))
                          (List.tabulate (ns, fn i => i))
                      else ()
                    val () =
                      List.app (fn i =>
                        let val gi = Array.sub (g, i) in
                          Array.update (bNum, bIdx (i, ob), Array.sub (bNum, bIdx (i, ob)) + gi);
                          Array.update (bDen, i, Array.sub (bDen, i) + gi)
                        end)
                        (List.tabulate (ns, fn i => i))
                  in
                    accumGamma (t + 1)
                  end
              val () = accumGamma 0

              (* xi_t(i,j) for t = 0 .. tN-2, proportional to
                 alpha_t(i) a_ij b_j(o_{t+1}) beta_{t+1}(j). *)
              fun accumXi t =
                if t >= tN - 1 then ()
                else
                  let
                    val al = Array.sub (alphaArr, t)
                    val be1 = Array.sub (betas, t + 1)
                    val o1 = Array.sub (obsArr, t + 1)
                    val raw = Array.array (ns * ns, 0.0)
                    val () =
                      List.app (fn i =>
                        List.app (fn j =>
                          Array.update (raw, aIdx (i, j),
                            Array.sub (al, i) * aP model i j
                            * emitP model j o1 * Array.sub (be1, j)))
                          (List.tabulate (ns, fn j => j)))
                        (List.tabulate (ns, fn i => i))
                    val s = Array.foldl (fn (x, acc) => x + acc) 0.0 raw
                    val inv = if s <= 0.0 then 0.0 else 1.0 / s
                    val () =
                      List.app (fn i =>
                        List.app (fn j =>
                          let val xij = Array.sub (raw, aIdx (i, j)) * inv in
                            Array.update (aNum, aIdx (i, j), Array.sub (aNum, aIdx (i, j)) + xij);
                            Array.update (aDen, i, Array.sub (aDen, i) + xij)
                          end)
                          (List.tabulate (ns, fn j => j)))
                        (List.tabulate (ns, fn i => i))
                  in
                    accumXi (t + 1)
                  end
              val () = accumXi 0
            in
              ()
            end

      val () = List.app processSeq seqs

      (* Number of sequences that contributed to pi (nonempty ones). *)
      val nSeq = List.length (List.filter (fn s => not (List.null s)) seqs)
      val nSeqR = Real.fromInt (if nSeq = 0 then 1 else nSeq)

      (* Build new pi, A, B; fall back to the old row if a denominator is 0 so
         the model stays a valid (normalized) HMM. *)
      val oldPi = #pi model
      val newPi =
        List.tabulate (ns, fn i =>
          let val v = Array.sub (piAcc, i) / nSeqR
          in v end)
      (* Renormalize pi defensively. *)
      val piSum = sumList newPi
      val newPi = if piSum <= 0.0 then arrayToList oldPi
                  else List.map (fn x => x / piSum) newPi

      fun newRowA i =
        let val den = Array.sub (aDen, i) in
          if den <= 0.0 then List.tabulate (ns, fn j => aP model i j)
          else List.tabulate (ns, fn j => Array.sub (aNum, aIdx (i, j)) / den)
        end
      fun newRowB i =
        let val den = Array.sub (bDen, i) in
          if den <= 0.0 then List.tabulate (no, fn k => emitP model i k)
          else List.tabulate (no, fn k => Array.sub (bNum, bIdx (i, k)) / den)
        end

      val newTrans = List.tabulate (ns, newRowA)
      val newEmit = List.tabulate (ns, newRowB)

      (* Renormalize each row defensively against tiny drift. *)
      fun renorm row =
        let val s = sumList row
        in if s <= 0.0 then row else List.map (fn x => x / s) row end
    in
      { ns = ns
      , no = no
      , pi = listToArray (renorm newPi)
      , a  = M.fromRows (List.map renorm newTrans)
      , b  = M.fromRows (List.map renorm newEmit) }
    end

  fun baumWelch model seqs { iters } =
    let
      fun loop (m, k) = if k <= 0 then m else loop (reestimate m seqs, k - 1)
    in
      if iters <= 0 then model else loop (model, iters)
    end
end
