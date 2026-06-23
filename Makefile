# sml-hmm build
#
#   make            build the test binary with MLton (default)
#   make test       build + run tests under MLton
#   make test-poly  run tests under Poly/ML (use-and-run; no link step)
#   make all-tests  run the suite under both compilers
#   make example    build + run examples/demo.sml
#   make clean      remove build artifacts
#
# Layout B (dependent): own sources live in src/; sml-matrix is vendored under
# lib/ and loaded first.

MLTON   ?= mlton
POLY    ?= poly
BIN     := bin
MATDIR  := lib/github.com/sjqtentacles/sml-matrix
TEST_MLB := test/sources.mlb
SRCS    := $(wildcard $(MATDIR)/* src/* test/*.sml) $(TEST_MLB)

.PHONY: all test poly test-poly all-tests example clean

all: $(BIN)/test-mlton

example: $(BIN)/demo
	./$(BIN)/demo

$(BIN)/demo: $(SRCS) examples/demo.sml examples/sources.mlb | $(BIN)
	$(MLTON) -output $@ examples/sources.mlb

$(BIN)/test-mlton: $(SRCS) | $(BIN)
	$(MLTON) -output $@ $(TEST_MLB)

test: $(BIN)/test-mlton
	$(BIN)/test-mlton

# Poly/ML has no native .mlb support; the suite runs at top level and exits on
# its own. Load the vendored sml-matrix first, then the hmm sources, then the
# test driver, in dependency order.
poly test-poly:
	printf 'use "$(MATDIR)/matrix.sig";\nuse "$(MATDIR)/matrix.sml";\nuse "src/hmm.sig";\nuse "src/hmm.sml";\nuse "test/harness.sml";\nuse "test/support.sml";\nuse "test/test_forward_backward.sml";\nuse "test/test_viterbi.sml";\nuse "test/test_baumwelch.sml";\nuse "test/entry.sml";\nuse "test/main.sml";\n' | $(POLY) -q --error-exit

all-tests: test test-poly

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f $(BIN)/test-mlton $(BIN)/demo
