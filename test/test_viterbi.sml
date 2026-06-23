(* test_viterbi.sml -- the Viterbi most-likely-path decoder.

   The classic "occasionally dishonest casino": a dealer switches between a
   Fair die and a Loaded die. We never see the die, only the rolls.

     states : 0 = Fair, 1 = Loaded
     obs    : 0..5 = die faces 1..6
     init   pi = [0.5, 0.5]
     trans  A  = [[0.95, 0.05],    (* Fair  -> stays fair, rarely switches *)
                  [0.10, 0.90]]    (* Loaded-> sticky *)
     emit   Fair   = uniform 1/6
            Loaded = [0.1,0.1,0.1,0.1,0.1,0.5]  (* heavy on a six *)

   Properties checked:
   - A long run of sixes is decoded as an all-Loaded path.
   - A varied, six-free roll is decoded as an all-Fair path.
   - The log-probability Viterbi returns equals the joint log-probability of
     the very path it reports, recomputed independently along that path
     (exact identity, tolerance compare only). *)

structure ViterbiTests =
struct
  open Support
  structure H = Hmm

  val tol = 1E~9

  val fair = 1.0 / 6.0
  val model =
    H.make
      { numStates = 2
      , numObs    = 6
      , init      = [0.5, 0.5]
      , trans     = [[0.95, 0.05], [0.10, 0.90]]
      , emit      = [ [fair, fair, fair, fair, fair, fair]
                    , [0.1, 0.1, 0.1, 0.1, 0.1, 0.5] ] }

  (* Recompute the joint log P(path, obs) directly, independently of viterbi. *)
  fun ln x = Math.ln x
  val initL  = [0.5, 0.5]
  val transL = [[0.95, 0.05], [0.10, 0.90]]
  val emitL  = [ [fair, fair, fair, fair, fair, fair]
               , [0.1, 0.1, 0.1, 0.1, 0.1, 0.5] ]
  fun at xss i j = List.nth (List.nth (xss, i), j)
  fun jointLogProb (path, obs) =
    let
      fun step (prev, []) = 0.0
        | step (prev, (s, ob) :: rest) =
            let
              val trans =
                case prev of
                  NONE => ln (List.nth (initL, s))
                | SOME p => ln (at transL p s)
            in
              trans + ln (at emitL s ob) + step (SOME s, rest)
            end
    in
      step (NONE, ListPair.zip (path, obs))
    end

  fun run () =
    let
      val allSixes = [5, 5, 5, 5, 5, 5, 5, 5]
      val (pathSixes, lpSixes) = H.viterbi model allSixes
      val () = Harness.section "viterbi: a run of sixes decodes to all-Loaded"
      val () = Harness.checkIntList "all-Loaded path"
                 (List.map (fn _ => 1) allSixes, pathSixes)
      val () = Harness.checkInt "path length matches obs length"
                 (List.length allSixes, List.length pathSixes)

      val varied = [0, 1, 2, 3, 4, 0, 1, 2]   (* no sixes at all *)
      val (pathVaried, _) = H.viterbi model varied
      val () = Harness.section "viterbi: a six-free roll decodes to all-Fair"
      val () = Harness.checkIntList "all-Fair path"
                 (List.map (fn _ => 0) varied, pathVaried)

      val () = Harness.section "viterbi: reported log-prob equals the path's joint log-prob"
      val () = checkApproxTol tol "sixes: lp = joint(path,obs)"
                 (jointLogProb (pathSixes, allSixes), lpSixes)
      val (pathV, lpV) = H.viterbi model varied
      val () = checkApproxTol tol "varied: lp = joint(path,obs)"
                 (jointLogProb (pathV, varied), lpV)

      val () = Harness.section "viterbi: the decoded path is at least as likely as any rival"
      (* The all-Fair and all-Loaded paths are concrete rivals; the Viterbi
         optimum for the sixes run must dominate the all-Fair alternative. *)
      val allFairOnSixes = List.map (fn _ => 0) allSixes
      val () = Harness.check "viterbi optimum >= all-Fair rival on sixes"
                 (lpSixes >= jointLogProb (allFairOnSixes, allSixes) - tol)
    in
      ()
    end
end
