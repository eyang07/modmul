# SAIR Modular Arithmetic Challenge Agent Context

Official competition page:
https://competition.sair.foundation/competitions/modular-arithmetic-challenge/overview

Official GitHub repo:
https://github.com/SAIRcompetition/modular-arithmetic-challenge

Goal:
Build a compliant neural model that computes (a * b) mod p from string inputs a, b, p. The output must come from trained parameters, not Python big integers, SymPy, lookup tables, hand-coded multiplication/reduction, or tensorized arithmetic algorithms.

First files to read:
- README.md
- rules/overview.md
- rules/evaluation.md
- rules/literature.md
- examples/README.md
- examples/always_zero/
- examples/digit_transformer/
- examples/dlp_grokking/
- src/modchallenge/interface/
- src/modchallenge/evaluation/
- src/modchallenge/security/
- tests/

My background:
Strong math/formal methods background, little ML engineering experience. I need a beginner-friendly but serious plan. Start with supervised synthetic training and baseline reproduction. Do not start with RL.

Initial goal:
Get the repo installed, run tests, evaluate official examples, understand the model interface, then build one simple compliant custom baseline.