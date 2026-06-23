(* entry.sml -- runs every suite and exits with a status code. *)

fun runAllSuites () =
  ( Harness.reset ()
  ; ForwardBackwardTests.run ()
  ; ViterbiTests.run ()
  ; BaumWelchTests.run ()
  ; Harness.run () )

fun main () =
  OS.Process.exit
    (if runAllSuites () then OS.Process.success else OS.Process.failure)
