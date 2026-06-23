# sml-hmm

Hidden Markov Models in pure Standard ML — the **forward** and **backward**
likelihood recursions, **Viterbi** decoding, and **Baum-Welch** (EM)
re-estimation — built on top of
[`sml-matrix`](https://github.com/sjqtentacles/sml-matrix) for the dense
transition / emission linear algebra. No FFI, no external dependencies at
runtime, and **deterministic**, byte-identically under both
[MLton](http://mlton.org/) and [Poly/ML](https://www.polyml.org/).

[![CI](https://github.com/sjqtentacles/sml-hmm/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-hmm/actions/workflows/ci.yml)

## HMM, not a plain Markov chain

This library is deliberately **distinct** from
[`sml-markov`](https://github.com/sjqtentacles/sml-markov):

- **`sml-markov`** models *observable* Markov chains — there are no hidden
  states. It provides n-step distributions, matrix powers, the stationary
  distribution by power iteration, and seeded trajectory sampling. The state at
  every step is directly observed.
- **`sml-hmm`** adds a *hidden* layer: the underlying state sequence is latent
  and only a sequence of emitted observations is seen. It answers the three
  classic HMM questions that a plain Markov chain cannot:
  **evaluation** (`forward`/`backward`), **decoding** (`viterbi`), and
  **learning** (`baumWelch`). Hidden-state inference is exactly what
  `sml-markov` does *not* have.

Both build on `sml-matrix` for the linear algebra; `sml-hmm` stores its
transition matrix `A` and emission matrix `B` as `Matrix.t` values.

## Status

- 37 assertions, green on MLton and Poly/ML.
- Basis-library + vendored `sml-matrix` only; deterministic across compilers.
- Vendors `sml-matrix` (Layout B) under
  `lib/github.com/sjqtentacles/sml-matrix/`, so the repo builds standalone.

## Purity

No FFI, no IO inside the library, no wall-clock, no ambient randomness, and no
threads. Evaluation runs in **log-space**, and the forward/backward recursions
use **per-step scaling** to avoid underflow, so the same model and observation
sequence always produce the same answer across runs, machines, and compilers.
Baum-Welch uses a **fixed iteration count** for determinism — the test suite and
the demo are byte-identical under MLton and Poly/ML. All returned quantities are
`real`, so tests compare them through an explicit tolerance (`Support.approx`),
never string or structural equality on reals.

## Install

With [`smlpkg`](https://github.com/diku-dk/smlpkg):

```
smlpkg add github.com/sjqtentacles/sml-hmm
smlpkg sync
```

Include the MLB from your own (it pulls in the vendored `sml-matrix`):

```
local
  $(SML_LIB)/basis/basis.mlb
  lib/github.com/sjqtentacles/sml-hmm/... (via smlpkg)
in
  ...
end
```

This brings `structure Hmm` (and the vendored `Matrix`) into scope.

## Quick start

```sml
(* a tiny weather/umbrella HMM: 2 hidden states, 3 observation symbols *)
val model =
  Hmm.make
    { numStates = 2
    , numObs    = 3
    , init      = [0.6, 0.4]
    , trans     = [[0.7, 0.3], [0.4, 0.6]]
    , emit      = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]] }

(* evaluation: log-likelihood of an observation sequence (forward = backward) *)
val ll = Hmm.forward  model [0, 2]      (* ln 0.0852                          *)
val _  = Hmm.backward model [0, 2]      (* agrees with forward                *)
val p  = Math.exp ll                    (* 0.0852                             *)

(* decoding: the most likely hidden path and its joint log-probability *)
val (path, logp) = Hmm.viterbi model [0, 2, 2, 1]

(* learning: re-estimate from observation sequences with a fixed iteration count *)
val fitted = Hmm.baumWelch model [[0,2,2], [1,0,0], [2,2,1]] { iters = 20 }
```

## API (`signature HMM`)

```sml
type t
exception Hmm of string

val make : { numStates : int
           , numObs    : int
           , init      : real list         (* pi,  length numStates           *)
           , trans     : real list list     (* A,   numStates x numStates      *)
           , emit      : real list list }   (* B,   numStates x numObs         *)
           -> t

val numStates : t -> int
val numObs    : t -> int
val init      : t -> real list
val trans     : t -> real list list
val emit      : t -> real list list

val forward   : t -> int list -> real               (* ln P(obs)              *)
val backward  : t -> int list -> real               (* ln P(obs), = forward   *)
val viterbi   : t -> int list -> int list * real     (* best path, ln P(path,obs) *)
val baumWelch : t -> int list list -> { iters : int } -> t
```

- **States and symbols** are 0-indexed: states `0 .. numStates-1`, observation
  symbols `0 .. numObs-1`.
- **`make`** validates shapes and that `pi`, every row of `A`, and every row of
  `B` are nonnegative distributions summing to 1 (within tolerance); it raises
  `Hmm` otherwise.
- **`forward` / `backward`** return the *natural-log* likelihood
  `ln P(obs | model)`. They agree on the total likelihood within tolerance. The
  empty sequence has likelihood 1, so both return `0.0`.
- **`viterbi`** returns the most likely hidden-state path together with its
  joint log-probability `ln P(path, obs)`. The empty sequence yields `([], 0.0)`.
- **`baumWelch`** runs `iters` EM passes; the training data log-likelihood is
  non-decreasing across passes, the re-estimated model stays a valid (row-
  normalized) HMM, and the result is deterministic for a fixed `iters`
  (`iters = 0` is the identity).

### Conventions and math

- **Forward (evaluation).** `alpha_1(i) = pi_i b_i(o_1)`, then
  `alpha_{t+1}(j) = (sum_i alpha_t(i) a_ij) b_j(o_{t+1})`, with each column
  renormalised to sum to 1 and the scale factor recorded; the log-likelihood is
  the sum of the logs of those factors.
- **Backward.** `beta_T(i) = 1`, then
  `beta_t(i) = sum_j a_ij b_j(o_{t+1}) beta_{t+1}(j)`, scaled with the same
  per-step factors; it reconstructs the same total likelihood as forward.
- **Viterbi (decoding).** The same recursion with `max` instead of `sum`, run
  entirely in log-space (`delta_t(j) = max_i (delta_{t-1}(i) + ln a_ij) +
  ln b_j(o_t)`), with backpointers for the optimal path.
- **Baum-Welch (learning).** The E-step forms the state posteriors `gamma` and
  transition posteriors `xi` from the scaled forward/backward variables; the
  M-step renormalises them into a new `pi`, `A` and `B`. Rows are renormalised
  defensively so the re-estimated model is always a valid HMM.
- **Reals via tolerance.** Real formatting differs across compilers, so every
  comparison goes through an explicit epsilon — never string equality on reals.

## Build & test

```
make test        # MLton
make test-poly   # Poly/ML
make all-tests   # both
make example     # build + run examples/demo.sml
make clean
```

Both compilers run the same strict-TDD suite (37 assertions):

- **forward closed-form** — `P([0]) = 0.30` and `P([0,2]) = 0.0852` on the
  weather/umbrella HMM, compared with a tight tolerance (never string equality);
- **forward = backward** — the two recursions agree on the total likelihood of
  several observation sequences;
- **Viterbi recovery** — on the occasionally-dishonest casino, a run of sixes
  decodes to an all-Loaded path and a six-free roll to an all-Fair path, and the
  log-probability Viterbi reports equals the joint log-probability recomputed
  independently along that path;
- **Baum-Welch monotonicity** — the training-data log-likelihood is
  non-decreasing across 1, 2, 5 and 10 iterations and strictly improves overall;
- **normalization** — after re-estimation `pi`, every row of `A`, and every row
  of `B` still sum to 1 and stay in `[0, 1]`, and `iters = 0` is a no-op.

## Vendoring

This library depends on
[`sml-matrix`](https://github.com/sjqtentacles/sml-matrix), whose sources are
vendored verbatim under `lib/github.com/sjqtentacles/sml-matrix/`
(`matrix.sig`, `matrix.sml`, `sml-matrix.mlb`, and `sources.mlb`; the
dependency's own tests are *not* vendored). `src/hmm.mlb` references that
`sources.mlb` first, then `hmm.sig`/`hmm.sml`; the Poly/ML `use`-chain loads the
vendored signature and structure first, in dependency order. `sml.pkg` records
the dependency in its `require` block so `smlpkg sync` can refresh it.

## Example

`make example` evaluates, decodes, and re-fits the occasionally-dishonest
casino (output is byte-identical under MLton and Poly/ML):

```
=== sml-hmm demo ==============================================

Occasionally-dishonest casino (2 hidden states, 6 symbols).
  transition A built as a 2x2 sml-matrix
  emission B   built as a 2x6 sml-matrix

Observed rolls (0-indexed faces): [0,2,4,1,5,5,5,5,5,5]

Evaluation -- how likely is this sequence?
  log P(obs)  via forward  = -14.036803
  log P(obs)  via backward = -14.036803
  P(obs)                   = 0.0000008015

Decoding -- the most likely hidden path (Viterbi)
  path    = [Loaded,Loaded,Loaded,Loaded,Loaded,Loaded,Loaded,Loaded,Loaded,Loaded]
  indices = [1,1,1,1,1,1,1,1,1,1]
  log P(path, obs) = -15.010615

Learning -- Baum-Welch re-estimation (fixed iterations)
  training sequences: 4
  data log-likelihood before = -49.180442
  data log-likelihood after  5 iters = -39.905129
  data log-likelihood after 20 iters = -39.642309

Re-estimated initial distribution pi:
  [0.5000, 0.5000]
Re-estimated transition matrix A (rows out of each state):
  [0.9333, 0.0667]
  [0.1538, 0.8462]

===============================================================
```

### Poly/ML note

CI builds Poly/ML 5.9.1 from source rather than using the Ubuntu package
(Poly/ML 5.7.1), whose X86 code generator crashes (`asGenReg raised while
compiling`) on some code. See `.github/workflows/ci.yml`.

## License

MIT — see [LICENSE](LICENSE).
