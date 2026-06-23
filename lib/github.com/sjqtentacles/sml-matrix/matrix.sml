(* matrix.sml

   Dense real linear algebra over general m x n matrices. See matrix.sig for
   the contract. Storage is a flat `real array` in row-major order plus the
   row/column counts: entry (i, j) lives at flat index i * c + j. *)

structure Matrix :> MATRIX =
struct
  exception Dim of string
  exception Singular

  val eps = 1E~9

  (* A matrix is its row count, column count, and a row-major data array. *)
  type t = {r : int, c : int, a : real array}

  fun rows ({r, ...} : t) = r
  fun cols ({c, ...} : t) = c

  fun idx (c, i, j) = i * c + j

  fun checkIndex ({r, c, ...} : t, i, j) =
    if i < 0 orelse i >= r orelse j < 0 orelse j >= c then
      raise Dim ("index (" ^ Int.toString i ^ ", " ^ Int.toString j ^
                 ") out of range for " ^ Int.toString r ^ "x" ^ Int.toString c)
    else ()

  fun sub (m as {c, a, ...} : t, i, j) =
    (checkIndex (m, i, j); Array.sub (a, idx (c, i, j)))

  fun update (m as {c, a, ...} : t, i, j, x) =
    (checkIndex (m, i, j); Array.update (a, idx (c, i, j), x))

  (* --- construction --- *)

  fun make (r, c) x =
    if r <= 0 orelse c <= 0 then
      raise Dim ("make: non-positive dimension " ^
                 Int.toString r ^ "x" ^ Int.toString c)
    else {r = r, c = c, a = Array.array (r * c, x)}

  fun zeros (r, c) = make (r, c) 0.0

  fun identity n =
    let
      val m = zeros (n, n)
      val {a, ...} = m
      fun loop i = if i >= n then () else (Array.update (a, idx (n, i, i), 1.0); loop (i + 1))
    in
      loop 0; m
    end

  fun fromRows rs =
    let
      val r = List.length rs
      val () = if r = 0 then raise Dim "fromRows: empty matrix" else ()
      val c = List.length (List.hd rs)
      val () = if c = 0 then raise Dim "fromRows: empty row" else ()
      val () =
        List.app
          (fn row =>
             if List.length row <> c then
               raise Dim "fromRows: ragged rows"
             else ())
          rs
      val a = Array.array (r * c, 0.0)
      val _ =
        List.foldl
          (fn (row, i) =>
             (ignore (List.foldl
                (fn (x, j) => (Array.update (a, idx (c, i, j), x); j + 1))
                0 row);
              i + 1))
          0 rs
    in
      {r = r, c = c, a = a}
    end

  fun toRows ({r, c, a} : t) =
    List.tabulate
      (r, fn i => List.tabulate (c, fn j => Array.sub (a, idx (c, i, j))))

  (* --- arithmetic --- *)

  fun sameShape (x : t, y : t) = rows x = rows y andalso cols x = cols y

  fun zipWith name f (x as {r, c, a} : t, y : t) =
    if not (sameShape (x, y)) then
      raise Dim (name ^ ": shape mismatch " ^
                 Int.toString (rows x) ^ "x" ^ Int.toString (cols x) ^ " vs " ^
                 Int.toString (rows y) ^ "x" ^ Int.toString (cols y))
    else
      let
        val {a = b, ...} = y
        val out = Array.array (r * c, 0.0)
        fun loop k =
          if k >= r * c then ()
          else (Array.update (out, k, f (Array.sub (a, k), Array.sub (b, k)));
                loop (k + 1))
      in
        loop 0; {r = r, c = c, a = out}
      end

  fun add (x, y) = zipWith "add" (op +) (x, y)
  fun sub' (x, y) = zipWith "sub'" (op -) (x, y)

  fun scale s ({r, c, a} : t) =
    let
      val out = Array.array (r * c, 0.0)
      fun loop k =
        if k >= r * c then ()
        else (Array.update (out, k, s * Array.sub (a, k)); loop (k + 1))
    in
      loop 0; {r = r, c = c, a = out}
    end

  fun transpose ({r, c, a} : t) =
    let
      val out = Array.array (r * c, 0.0)
      fun loop (i, j) =
        if i >= r then ()
        else if j >= c then loop (i + 1, 0)
        else (Array.update (out, idx (r, j, i), Array.sub (a, idx (c, i, j)));
              loop (i, j + 1))
    in
      loop (0, 0); {r = c, c = r, a = out}
    end

  fun mul (x as {r = rx, c = cx, a = ax} : t, y as {r = ry, c = cy, a = ay} : t) =
    if cx <> ry then
      raise Dim ("mul: inner dimensions disagree " ^
                 Int.toString rx ^ "x" ^ Int.toString cx ^ " * " ^
                 Int.toString ry ^ "x" ^ Int.toString cy)
    else
      let
        val out = Array.array (rx * cy, 0.0)
        fun cell (i, j) =
          let
            fun acc (k, s) =
              if k >= cx then s
              else acc (k + 1,
                        s + Array.sub (ax, idx (cx, i, k))
                            * Array.sub (ay, idx (cy, k, j)))
          in
            acc (0, 0.0)
          end
        fun loop (i, j) =
          if i >= rx then ()
          else if j >= cy then loop (i + 1, 0)
          else (Array.update (out, idx (cy, i, j), cell (i, j)); loop (i, j + 1))
      in
        loop (0, 0); {r = rx, c = cy, a = out}
      end

  (* --- LU with partial pivoting --- *)

  fun requireSquare name (m : t) =
    if rows m <> cols m then
      raise Dim (name ^ ": matrix is not square (" ^
                 Int.toString (rows m) ^ "x" ^ Int.toString (cols m) ^ ")")
    else rows m

  (* Returns (lu, perm, sign): `lu` is a fresh n x n array holding L (below the
     diagonal, unit diagonal implied) and U (on and above the diagonal) packed
     together; `perm` maps result row -> original row; `sign` is the
     permutation parity. Does not raise on singular input; a zero pivot simply
     yields a zero on U's diagonal. *)
  fun factor (m : t) =
    let
      val n = requireSquare "lu" m
      val {a = src, ...} = m
      val a = Array.tabulate (n * n, fn k => Array.sub (src, k))
      val perm = Array.tabulate (n, fn i => i)
      val sign = ref 1.0
      fun get (i, j) = Array.sub (a, idx (n, i, j))
      fun set (i, j, x) = Array.update (a, idx (n, i, j), x)
      fun swapRows (p, q) =
        if p = q then ()
        else
          let
            val () =
              List.app
                (fn j =>
                   let val t = get (p, j) in set (p, j, get (q, j)); set (q, j, t) end)
                (List.tabulate (n, fn j => j))
            val tp = Array.sub (perm, p)
          in
            Array.update (perm, p, Array.sub (perm, q));
            Array.update (perm, q, tp);
            sign := ~ (!sign)
          end
      fun pivot k =
        let
          fun best (i, bi, bv) =
            if i >= n then bi
            else
              let val v = Real.abs (get (i, k))
              in if v > bv then best (i + 1, i, v) else best (i + 1, bi, bv) end
        in
          best (k + 1, k, Real.abs (get (k, k)))
        end
      fun eliminate k =
        if k >= n then ()
        else
          let
            val () = swapRows (k, pivot k)
            val pivv = get (k, k)
          in
            if Real.abs pivv <= eps then eliminate (k + 1)
            else
              let
                fun rowloop i =
                  if i >= n then ()
                  else
                    let
                      val f = get (i, k) / pivv
                      val () = set (i, k, f)
                      fun colloop j =
                        if j >= n then ()
                        else (set (i, j, get (i, j) - f * get (k, j));
                              colloop (j + 1))
                    in
                      colloop (k + 1); rowloop (i + 1)
                    end
              in
                rowloop (k + 1); eliminate (k + 1)
              end
          end
    in
      eliminate 0;
      {n = n, a = a, perm = perm, sign = !sign}
    end

  fun lu (m : t) =
    let
      val {n, a, perm, sign} = factor m
      val L = identity n
      val U = zeros (n, n)
      val () =
        List.app
          (fn i =>
             List.app
               (fn j =>
                  let val v = Array.sub (a, idx (n, i, j))
                  in
                    if j < i then update (L, i, j, v)
                    else update (U, i, j, v)
                  end)
               (List.tabulate (n, fn j => j)))
          (List.tabulate (n, fn i => i))
    in
      {l = L, u = U, p = perm, sign = sign}
    end

  fun det (m : t) =
    let
      val {n, a, sign, ...} = factor m
      fun prod (i, acc) =
        if i >= n then acc else prod (i + 1, acc * Array.sub (a, idx (n, i, i)))
    in
      sign * prod (0, 1.0)
    end

  (* Solve using a precomputed factorisation against a single right-hand side. *)
  fun solveFactored ({n, a, perm, ...}, b : real array) =
    let
      val () =
        List.app
          (fn i =>
             if Real.abs (Array.sub (a, idx (n, i, i))) <= eps then
               raise Singular
             else ())
          (List.tabulate (n, fn i => i))
      (* Permute b: y[i] starts as b[perm[i]]. *)
      val y = Array.tabulate (n, fn i => Array.sub (b, Array.sub (perm, i)))
      (* Forward substitution: L y = Pb, unit lower diagonal. *)
      fun fwd i =
        if i >= n then ()
        else
          let
            fun acc (j, s) =
              if j >= i then s
              else acc (j + 1, s + Array.sub (a, idx (n, i, j)) * Array.sub (y, j))
          in
            Array.update (y, i, Array.sub (y, i) - acc (0, 0.0)); fwd (i + 1)
          end
      (* Back substitution: U x = y. *)
      val x = Array.array (n, 0.0)
      fun back i =
        if i < 0 then ()
        else
          let
            fun acc (j, s) =
              if j >= n then s
              else acc (j + 1, s + Array.sub (a, idx (n, i, j)) * Array.sub (x, j))
          in
            Array.update (x, i,
              (Array.sub (y, i) - acc (i + 1, 0.0)) / Array.sub (a, idx (n, i, i)));
            back (i - 1)
          end
    in
      fwd 0; back (n - 1); x
    end

  fun solve (m : t) b =
    let
      val n = requireSquare "solve" m
      val () =
        if List.length b <> n then
          raise Dim ("solve: rhs length " ^ Int.toString (List.length b) ^
                     " <> matrix size " ^ Int.toString n)
        else ()
      val f = factor m
      val bv = Array.fromList b
      val x = solveFactored (f, bv)
    in
      List.tabulate (n, fn i => Array.sub (x, i))
    end

  fun inv (m : t) =
    let
      val n = requireSquare "inv" m
      val f = factor m
      val out = zeros (n, n)
      val () =
        List.app
          (fn j =>
             let
               val e = Array.tabulate (n, fn i => if i = j then 1.0 else 0.0)
               val col = solveFactored (f, e)
             in
               List.app (fn i => update (out, i, j, Array.sub (col, i)))
                 (List.tabulate (n, fn i => i))
             end)
          (List.tabulate (n, fn j => j))
    in
      out
    end

  (* --- QR via classical Gram-Schmidt with one reorthogonalisation pass --- *)

  fun qr (m : t) =
    let
      val mm = rows m
      val nn = cols m
      val () =
        if mm < nn then
          raise Dim ("qr: needs rows >= cols, got " ^
                     Int.toString mm ^ "x" ^ Int.toString nn)
        else ()
      val Q = zeros (mm, nn)
      val R = zeros (nn, nn)
      fun colDot (Acol, Bcol) =
        let
          fun acc (i, s) =
            if i >= mm then s
            else acc (i + 1, s + sub (Q, i, Acol) * sub (Q, i, Bcol))
        in
          acc (0, 0.0)
        end
      (* Copy column j of A into column j of Q to start. *)
      fun initCol j =
        List.app (fn i => update (Q, i, j, sub (m, i, j)))
          (List.tabulate (mm, fn i => i))
      (* Subtract projection onto already-orthonormal column k, accumulating
         the coefficient into R[k,j]. *)
      fun project (j, k) =
        let
          val d = colDot (k, j)
          val () = update (R, k, j, sub (R, k, j) + d)
        in
          List.app
            (fn i => update (Q, i, j, sub (Q, i, j) - d * sub (Q, i, k)))
            (List.tabulate (mm, fn i => i))
        end
      fun norm j =
        let
          fun acc (i, s) =
            if i >= mm then s else acc (i + 1, s + sub (Q, i, j) * sub (Q, i, j))
        in
          Math.sqrt (acc (0, 0.0))
        end
      fun gs j =
        if j >= nn then ()
        else
          let
            val () = initCol j
            (* Two passes of orthogonalisation against columns 0..j-1. *)
            val cols' = List.tabulate (j, fn k => k)
            val () = List.app (fn k => project (j, k)) cols'
            val () = List.app (fn k => project (j, k)) cols'
            val nj = norm j
            val () = if nj <= eps then raise Singular else ()
            val () = update (R, j, j, nj)
            val () =
              List.app (fn i => update (Q, i, j, sub (Q, i, j) / nj))
                (List.tabulate (mm, fn i => i))
          in
            gs (j + 1)
          end
    in
      gs 0; {q = Q, r = R}
    end

  (* --- scalar summaries and norms --- *)

  fun trace (m : t) =
    let
      val n = requireSquare "trace" m
      fun acc (i, s) = if i >= n then s else acc (i + 1, s + sub (m, i, i))
    in
      acc (0, 0.0)
    end

  fun norm1 (m : t) =
    let
      val r = rows m and c = cols m
      fun colSum j =
        let fun acc (i, s) = if i >= r then s
                             else acc (i + 1, s + Real.abs (sub (m, i, j)))
        in acc (0, 0.0) end
      fun loop (j, best) = if j >= c then best
                           else loop (j + 1, Real.max (best, colSum j))
    in
      loop (0, 0.0)
    end

  fun normInf (m : t) =
    let
      val r = rows m and c = cols m
      fun rowSum i =
        let fun acc (j, s) = if j >= c then s
                             else acc (j + 1, s + Real.abs (sub (m, i, j)))
        in acc (0, 0.0) end
      fun loop (i, best) = if i >= r then best
                           else loop (i + 1, Real.max (best, rowSum i))
    in
      loop (0, 0.0)
    end

  fun normFro ({r, c, a} : t) =
    let
      fun acc (k, s) =
        if k >= r * c then s
        else let val x = Array.sub (a, k) in acc (k + 1, s + x * x) end
    in
      Math.sqrt (acc (0, 0.0))
    end

  (* --- Cholesky (Cholesky-Banachiewicz), lower triangle of A only --- *)

  fun cholesky (m : t) =
    let
      val n = requireSquare "cholesky" m
      val L = zeros (n, n)
      fun col j =
        if j >= n then ()
        else
          let
            fun dacc (k, s) =
              if k >= j then s
              else dacc (k + 1, s + sub (L, j, k) * sub (L, j, k))
            val d = sub (m, j, j) - dacc (0, 0.0)
            val () = if d <= eps then raise Singular else ()
            val ljj = Math.sqrt d
            val () = update (L, j, j, ljj)
            fun row i =
              if i >= n then ()
              else
                let
                  fun acc (k, s) =
                    if k >= j then s
                    else acc (k + 1, s + sub (L, i, k) * sub (L, j, k))
                in
                  update (L, i, j, (sub (m, i, j) - acc (0, 0.0)) / ljj);
                  row (i + 1)
                end
          in
            row (j + 1); col (j + 1)
          end
    in
      col 0; L
    end

  (* --- shared helpers for the Jacobi routines --- *)

  fun copyMat (m as {r, c, a} : t) : t =
    {r = r, c = c, a = Array.tabulate (r * c, fn k => Array.sub (a, k))}

  (* Rotate columns p and q of X (which has `rr` rows) by the Givens factors
     (cs, sn): new col p = cs*p - sn*q, new col q = sn*p + cs*q. This realises
     the right-multiplication X <- X J with J = [[cs, sn], [~sn, cs]]. *)
  fun rotateCols (X : t, rr, p, q, cs, sn) =
    let
      fun loop i =
        if i >= rr then ()
        else
          let val xip = sub (X, i, p) and xiq = sub (X, i, q)
          in
            update (X, i, p, cs * xip - sn * xiq);
            update (X, i, q, sn * xip + cs * xiq);
            loop (i + 1)
          end
    in
      loop 0
    end

  (* Sum over indices 0..rr-1 of X[i][p] * X[i][q]. *)
  fun colDot (X : t, rr, p, q) =
    let fun acc (i, s) = if i >= rr then s
                         else acc (i + 1, s + sub (X, i, p) * sub (X, i, q))
    in acc (0, 0.0) end

  (* Stable sort of the indices 0..n-1; `precedes (i, j)` is true when index i
     should come before index j. Equal keys keep ascending index order, so the
     result is deterministic. *)
  fun sortIndices (n, precedes) =
    let
      fun insert (x, []) = [x]
        | insert (x, y :: ys) =
            if precedes (x, y) then x :: y :: ys else y :: insert (x, ys)
      fun build (i, acc) = if i >= n then acc else build (i + 1, insert (i, acc))
    in
      build (0, [])
    end

  (* --- one-sided Jacobi SVD --- *)

  val jacobiSweeps = 100

  (* Operates on an m x n matrix with m >= n. Returns (U, s, V) where U is
     m x n with orthonormal columns, s the length-n singular values in
     descending order, and V the n x n right factor, so A = U diag(s) Vᵀ. *)
  fun svdTall (A : t) =
    let
      val m = rows A and n = cols A
      val W = copyMat A          (* converges to U * diag(s) by columns *)
      val V = identity n
      fun sweep () =
        let
          val rotated = ref false
          fun pairs (p, q) =
            if p >= n - 1 then ()
            else if q >= n then pairs (p + 1, p + 2)
            else
              let
                val alpha = colDot (W, m, p, p)
                val beta = colDot (W, m, q, q)
                val gamma = colDot (W, m, p, q)
              in
                (if Real.abs gamma > eps * Math.sqrt (alpha * beta)
                    andalso alpha * beta > 0.0
                 then
                   let
                     val tau = (beta - alpha) / (2.0 * gamma)
                     val t = (if tau >= 0.0 then 1.0 else ~1.0)
                             / (Real.abs tau + Math.sqrt (1.0 + tau * tau))
                     val cs = 1.0 / Math.sqrt (1.0 + t * t)
                     val sn = cs * t
                   in
                     rotateCols (W, m, p, q, cs, sn);
                     rotateCols (V, n, p, q, cs, sn);
                     rotated := true
                   end
                 else ());
                pairs (p, q + 1)
              end
          val () = pairs (0, 1)
        in
          !rotated
        end
      fun loop k = if k >= jacobiSweeps then ()
                   else if sweep () then loop (k + 1) else ()
      val () = loop 0
      val sRaw = Array.tabulate (n, fn j => Math.sqrt (colDot (W, m, j, j)))
      val order = sortIndices (n, fn (i, j) =>
                    Array.sub (sRaw, i) > Array.sub (sRaw, j))
      val ord = Array.fromList order
      val s = Array.tabulate (n, fn j => Array.sub (sRaw, Array.sub (ord, j)))
      val U = zeros (m, n)
      val Vs = zeros (n, n)
      val () =
        List.app
          (fn j =>
             let
               val src = Array.sub (ord, j)
               val sj = Array.sub (sRaw, src)
             in
               List.app
                 (fn i =>
                    (update (U, i, j,
                       if sj > 0.0 then sub (W, i, src) / sj else 0.0)))
                 (List.tabulate (m, fn i => i));
               List.app
                 (fn i => update (Vs, i, j, sub (V, i, src)))
                 (List.tabulate (n, fn i => i))
             end)
          (List.tabulate (n, fn j => j))
    in
      {u = U, s = s, v = Vs}
    end

  fun svd (A : t) =
    let
      val m = rows A and n = cols A
    in
      if m >= n then
        let val {u, s, v} = svdTall A
        in {u = u, s = s, vt = transpose v} end
      else
        (* A = (Aᵀ)ᵀ: with Aᵀ = U' diag(s) V'ᵀ, A = V' diag(s) U'ᵀ. *)
        let val {u = u', s, v = v'} = svdTall (transpose A)
        in {u = v', s = s, vt = transpose u'} end
    end

  fun svTol (A : t, s : real array) =
    let val k = Array.length s
    in
      if k = 0 then 0.0
      else Array.sub (s, 0)
           * Real.fromInt (Int.max (rows A, cols A)) * eps
    end

  fun norm2 (A : t) =
    let val {s, ...} = svd A
    in if Array.length s = 0 then 0.0 else Array.sub (s, 0) end

  fun cond (A : t) =
    let
      val {s, ...} = svd A
      val k = Array.length s
    in
      if k = 0 then 0.0
      else
        let
          val smax = Array.sub (s, 0)
          val smin = Array.sub (s, k - 1)
        in
          if smin <= svTol (A, s) then Real.posInf else smax / smin
        end
    end

  fun rank (A : t) =
    let
      val {s, ...} = svd A
      val k = Array.length s
      val tol = svTol (A, s)
      fun count (i, acc) =
        if i >= k then acc
        else count (i + 1, if Array.sub (s, i) > tol then acc + 1 else acc)
    in
      count (0, 0)
    end

  fun pinv (A : t) =
    let
      val m = rows A and n = cols A
      val {u, s, vt} = svd A           (* u: m x k, s: k, vt: k x n *)
      val k = Array.length s
      val tol = svTol (A, s)
      val P = zeros (n, m)             (* pinv = V diag(s+) Uᵀ *)
      fun term i =
        if i >= k then ()
        else
          (if Array.sub (s, i) > tol then
             let val inv = 1.0 / Array.sub (s, i)
             in
               List.app
                 (fn j =>
                    List.app
                      (fn l =>
                         update (P, j, l,
                           sub (P, j, l) + inv * sub (vt, i, j) * sub (u, l, i)))
                      (List.tabulate (m, fn l => l)))
                 (List.tabulate (n, fn j => j))
             end
           else ();
           term (i + 1))
    in
      term 0; P
    end

  (* --- symmetric (cyclic) Jacobi eigendecomposition --- *)

  fun eigSym (A : t) =
    let
      val n = requireSquare "eigSym" A
      val () =
        List.app
          (fn i =>
             List.app
               (fn j =>
                  if Real.abs (sub (A, i, j) - sub (A, j, i))
                     > 1E~9 * (1.0 + Real.abs (sub (A, i, j))
                                   + Real.abs (sub (A, j, i)))
                  then raise Dim "eigSym: matrix is not symmetric"
                  else ())
               (List.tabulate (i, fn j => j)))
          (List.tabulate (n, fn i => i))
      val D = copyMat A
      val V = identity n
      val scale = let val f = normFro A in if f <= 0.0 then 1.0 else f end
      val threshold = eps * scale
      fun offNorm () =
        let
          fun acc (i, j, s) =
            if i >= n then s
            else if j >= n then acc (i + 1, 0, s)
            else if i = j then acc (i, j + 1, s)
            else let val x = sub (D, i, j) in acc (i, j + 1, s + x * x) end
        in
          Math.sqrt (acc (0, 0, 0.0))
        end
      (* Rotate (p, q) so D[p][q] becomes 0: D <- Jᵀ D J, V <- V J. *)
      fun rotate (p, q) =
        let
          val dpq = sub (D, p, q)
        in
          if Real.abs dpq <= 0.0 then ()
          else
            let
              val dpp = sub (D, p, p) and dqq = sub (D, q, q)
              val tau = (dqq - dpp) / (2.0 * dpq)
              val t = (if tau >= 0.0 then 1.0 else ~1.0)
                      / (Real.abs tau + Math.sqrt (1.0 + tau * tau))
              val cs = 1.0 / Math.sqrt (1.0 + t * t)
              val sn = cs * t
              (* D J : update columns p, q. *)
              val () =
                List.app
                  (fn i =>
                     let val dip = sub (D, i, p) and diq = sub (D, i, q)
                     in
                       update (D, i, p, cs * dip - sn * diq);
                       update (D, i, q, sn * dip + cs * diq)
                     end)
                  (List.tabulate (n, fn i => i))
              (* Jᵀ (D J) : update rows p, q from the column-updated D. *)
              val () =
                List.app
                  (fn j =>
                     let val dpj = sub (D, p, j) and dqj = sub (D, q, j)
                     in
                       update (D, p, j, cs * dpj - sn * dqj);
                       update (D, q, j, sn * dpj + cs * dqj)
                     end)
                  (List.tabulate (n, fn j => j))
              val () = update (D, p, q, 0.0)
              val () = update (D, q, p, 0.0)
            in
              rotateCols (V, n, p, q, cs, sn)
            end
        end
      fun sweep () =
        List.app
          (fn p =>
             List.app (fn q => rotate (p, q))
               (List.tabulate (n - p - 1, fn k => p + 1 + k)))
          (List.tabulate (n, fn p => p))
      fun loop k =
        if k >= jacobiSweeps then ()
        else if offNorm () <= threshold then ()
        else (sweep (); loop (k + 1))
      val () = loop 0
      val raw = Array.tabulate (n, fn i => sub (D, i, i))
      val order = sortIndices (n, fn (i, j) =>
                    Array.sub (raw, i) < Array.sub (raw, j))
      val ord = Array.fromList order
      val values = Array.tabulate (n, fn j => Array.sub (raw, Array.sub (ord, j)))
      val vectors = zeros (n, n)
      val () =
        List.app
          (fn j =>
             let val src = Array.sub (ord, j)
             in
               List.app (fn i => update (vectors, i, j, sub (V, i, src)))
                 (List.tabulate (n, fn i => i))
             end)
          (List.tabulate (n, fn j => j))
    in
      {values = values, vectors = vectors}
    end

  (* --- least squares via QR --- *)

  fun lstsq (m : t) b =
    let
      val mm = rows m and nn = cols m
      val () =
        if List.length b <> mm then
          raise Dim ("lstsq: rhs length " ^ Int.toString (List.length b) ^
                     " <> rows " ^ Int.toString mm)
        else ()
      val {q, r} = qr m
      val bv = Array.fromList b
      (* y = Qᵀ b, length nn. *)
      val y =
        Array.tabulate (nn, fn j =>
          let fun acc (i, s) =
                if i >= mm then s
                else acc (i + 1, s + sub (q, i, j) * Array.sub (bv, i))
          in acc (0, 0.0) end)
      (* Back-substitute R x = y. *)
      val x = Array.array (nn, 0.0)
      fun back i =
        if i < 0 then ()
        else
          let
            fun acc (j, s) =
              if j >= nn then s
              else acc (j + 1, s + sub (r, i, j) * Array.sub (x, j))
            val rii = sub (r, i, i)
            val () = if Real.abs rii <= eps then raise Singular else ()
          in
            Array.update (x, i, (Array.sub (y, i) - acc (i + 1, 0.0)) / rii);
            back (i - 1)
          end
    in
      back (nn - 1);
      List.tabulate (nn, fn i => Array.sub (x, i))
    end
end
