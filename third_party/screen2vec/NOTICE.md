# Third-party code: Screen2Vec

This folder holds the source code released with

> Toby Jia-Jun Li, Lindsay Popowski, Tom M. Mitchell, and Brad A. Myers. 2021.
> Screen2Vec: Semantic Embedding of GUI Screens and GUI Components. CHI 2021.

It was taken from https://github.com/tobyli/Screen2Vec at commit `f039f1f`
(3 February 2025) and is licensed under the Apache License 2.0 (see `LICENSE`
in this folder). It is not part of the thesis's own pipeline. It is kept here
because Section 4.6 of the thesis describes what this code does and compares the
thesis's evaluation with it.

One change was made: in `modeltester.py`, two `parser.add_argument(...)` calls
that the original has on a single line were split onto two lines. Nothing else
differs from the commit named above.
