(* demo.sml

   A tour of `sml-hmm`: the classic occasionally-dishonest-casino HMM, built on
   the vendored `sml-matrix` for the transition and emission matrices. We
   evaluate the likelihood of a roll sequence (forward = backward), decode the
   most likely Fair/Loaded path with Viterbi, then re-fit the model from a bank
   of sequences with a fixed number of Baum-Welch iterations.

   This is hidden-state inference -- unlike `sml-markov`, which models only
   observable chains. The output is byte-identical across MLton and Poly/ML
   (fixed-decimal formatting, log-space and scaled arithmetic, a fixed iteration
   count).

   Build and run with `make example`. *)

structure H = Hmm

fun fmt k x =
  let val s = Real.fmt (StringCvt.FIX (SOME k)) x
  in String.translate (fn c => if c = #"~" then "-" else str c) s end
fun line s = print (s ^ "\n")

fun intList xs = "[" ^ String.concatWith "," (List.map Int.toString xs) ^ "]"
fun stateName i = if i = 0 then "Fair" else "Loaded"
fun pathNames xs = "[" ^ String.concatWith "," (List.map stateName xs) ^ "]"

(* states: 0 = Fair, 1 = Loaded; obs 0..5 = die faces 1..6. *)
val fair = 1.0 / 6.0
val model =
  H.make
    { numStates = 2
    , numObs    = 6
    , init      = [0.5, 0.5]
    , trans     = [[0.95, 0.05], [0.10, 0.90]]
    , emit      = [ [fair, fair, fair, fair, fair, fair]
                  , [0.1, 0.1, 0.1, 0.1, 0.1, 0.5] ] }

(* a roll sequence: a fair-looking opening, then a suspicious run of sixes *)
val rolls = [0, 2, 4, 1, 5, 5, 5, 5, 5, 5]

val () = line "=== sml-hmm demo =============================================="
val () = line ""
val () = line "Occasionally-dishonest casino (2 hidden states, 6 symbols)."
val () = line ("  transition A built as a " ^ Int.toString (H.numStates model)
               ^ "x" ^ Int.toString (H.numStates model) ^ " sml-matrix")
val () = line ("  emission B   built as a " ^ Int.toString (H.numStates model)
               ^ "x" ^ Int.toString (H.numObs model) ^ " sml-matrix")
val () = line ""

val () = line ("Observed rolls (0-indexed faces): " ^ intList rolls)
val () = line ""

val () = line "Evaluation -- how likely is this sequence?"
val lf = H.forward model rolls
val lb = H.backward model rolls
val () = line ("  log P(obs)  via forward  = " ^ fmt 6 lf)
val () = line ("  log P(obs)  via backward = " ^ fmt 6 lb)
val () = line ("  P(obs)                   = " ^ fmt 10 (Math.exp lf))
val () = line ""

val () = line "Decoding -- the most likely hidden path (Viterbi)"
val (path, lp) = H.viterbi model rolls
val () = line ("  path    = " ^ pathNames path)
val () = line ("  indices = " ^ intList path)
val () = line ("  log P(path, obs) = " ^ fmt 6 lp)
val () = line ""

val () = line "Learning -- Baum-Welch re-estimation (fixed iterations)"
val training =
  [ [0, 2, 4, 1, 5, 5, 5, 5]
  , [5, 5, 5, 5, 0, 1, 2, 3]
  , [0, 1, 2, 3, 4, 0, 1, 2]
  , [5, 5, 5, 5, 5, 5, 0, 1] ]
fun dataLL m = List.foldl (fn (s, acc) => acc + H.forward m s) 0.0 training
val () = line ("  training sequences: " ^ Int.toString (List.length training))
val () = line ("  data log-likelihood before = " ^ fmt 6 (dataLL model))
val fitted5  = H.baumWelch model training { iters = 5 }
val fitted20 = H.baumWelch model training { iters = 20 }
val () = line ("  data log-likelihood after  5 iters = " ^ fmt 6 (dataLL fitted5))
val () = line ("  data log-likelihood after 20 iters = " ^ fmt 6 (dataLL fitted20))
val () = line ""

val () = line "Re-estimated initial distribution pi:"
val () = line ("  " ^ "[" ^ String.concatWith ", " (List.map (fmt 4) (H.init fitted20)) ^ "]")
val () = line "Re-estimated transition matrix A (rows out of each state):"
val () =
  List.app (fn r => line ("  [" ^ String.concatWith ", " (List.map (fmt 4) r) ^ "]"))
    (H.trans fitted20)
val () = line ""
val () = line "==============================================================="
