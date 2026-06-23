(* hmm.sig

   Hidden Markov Models in pure Standard ML, layered on top of the vendored
   `sml-matrix` library for the dense transition / emission linear algebra: a
   model is a set of hidden states, an observation alphabet, an initial state
   distribution `pi`, a row-stochastic transition matrix `A`, and a
   row-stochastic emission matrix `B`. Given an observation sequence the module
   answers the three classic HMM questions:

     - Evaluation:  how likely is this observation sequence? (`forward`,
       `backward` -- they agree on the total likelihood).
     - Decoding:    what is the single most likely hidden-state path?
       (`viterbi`).
     - Learning:    re-estimate the model from unlabelled data by
       expectation-maximisation (`baumWelch`).

   This is *hidden* Markov modelling -- inference over latent states -- and is
   deliberately distinct from `sml-markov`, which models *observable* Markov
   chains (n-step distributions, stationary distribution, seeded sampling) with
   no hidden layer. `sml-hmm` adds the hidden-state inference that `sml-markov`
   does not have.

   Determinism and purity. Everything is built on Basis-library `int`/`real`
   values threaded through pure helpers and the deterministic `Matrix`
   structure: no FFI, no wall-clock, no ambient randomness, and no threads. All
   evaluation runs in log-space (and Baum-Welch uses per-time scaling) so the
   arithmetic is numerically stable and byte-identical under both MLton and
   Poly/ML for a fixed iteration count.

   Conventions:
   - States are indexed 0 .. numStates-1; observation symbols 0 .. numObs-1.
   - `A` is numStates x numStates, row i giving the transition distribution
     out of state i; `B` is numStates x numObs, row i giving the emission
     distribution in state i; `pi` has length numStates. Each must be a valid
     (nonnegative, row-summing-to-1 within tolerance) distribution, else `make`
     raises `Hmm`.
   - `forward` and `backward` return *log* likelihoods (natural log). For a
     sequence of probability p they return ln p; exponentiate to recover p.
   - `viterbi` returns the most likely state path together with its joint
     *log*-probability ln P(path, obs).
   - All returned quantities are `real`; callers comparing them in tests must
     use a tolerance, never string or structural equality on reals. *)

signature HMM =
sig
  (* An immutable hidden Markov model. *)
  type t

  exception Hmm of string   (* malformed dimensions / distributions / obs *)

  (* `make {numStates, numObs, init, trans, emit}` builds a model. `init` must
     have length numStates; `trans` must be numStates x numStates; `emit` must
     be numStates x numObs. Every distribution must be nonnegative and sum to 1
     within tolerance. Raises `Hmm` on any shape or normalization violation. *)
  val make : { numStates : int
             , numObs    : int
             , init      : real list
             , trans     : real list list
             , emit      : real list list } -> t

  (* --- accessors --- *)

  val numStates : t -> int
  val numObs    : t -> int
  val init      : t -> real list        (* pi                          *)
  val trans     : t -> real list list   (* A, rows out of each state   *)
  val emit      : t -> real list list   (* B, rows per state           *)

  (* --- evaluation (question 1) --- *)

  (* `forward model obs` is ln P(obs | model) via the forward recursion (with
     per-step scaling for stability). The empty sequence has likelihood 1, so
     `forward model [] = 0.0`. Raises `Hmm` if any symbol is out of range. *)
  val forward : t -> int list -> real

  (* `backward model obs` is ln P(obs | model) via the backward recursion. It
     agrees with `forward` on the total likelihood (within tolerance). *)
  val backward : t -> int list -> real

  (* --- decoding (question 2) --- *)

  (* `viterbi model obs` is the most likely hidden-state path and its joint
     log-probability ln P(path, obs). For the empty sequence the path is `[]`
     with log-probability 0.0. Raises `Hmm` if any symbol is out of range. *)
  val viterbi : t -> int list -> int list * real

  (* --- learning (question 3) --- *)

  (* `baumWelch model seqs {iters}` runs `iters` expectation-maximisation
     passes over the observation sequences `seqs`, returning the re-estimated
     model. The data log-likelihood is non-decreasing across passes. With a
     fixed `iters` the result is deterministic and identical across compilers.
     `iters = 0` returns the input model unchanged. *)
  val baumWelch : t -> int list list -> { iters : int } -> t
end
