# DSA & Algorithms — From-Scratch Java

A personal repository of data structures and algorithms implemented from scratch in Java, for interview and research-engineering preparation. Every structure is written by hand (no `java.util` collections backing them), documented in prose notes, and validated by a self-contained test runner. The work is paced and deliberate: the goal is durable understanding, not just passing tests.

## Repo map

- **`Data_structures/`** — generic structures plus shared types (`Graph` interface, both graph representations, `Tree_node`), each in its own file with a `<Name>_Main` test runner.
- **`Algorithms/`** — reusable algorithms that build on those structures: suffix-array string algorithms, merge sort, graph traversals, tree algorithms, shortest paths, topological sorts, SCC, MST, Eulerian paths, and four max-flow implementations. Same per-file test-runner pattern.
- **`Problems/`** — specific named problems and exercises: the recursion/divide-and-conquer exercises, tilings, knapsacks, Kattis problems, Held-Karp TSP, grid shortest path, and the Mice-and-Owls bipartite-flow problem. The rule of thumb separating the two code packages: `Algorithms/` holds reusable procedures another file might import; `Problems/` holds leaf files that answer one specific question.
- **`Data_structures_notes.md`, `Algorithms_notes.md`, `Problems_notes.md`** — prose notes, one file per package (see the topic index).
- **`Images/`** — figures referenced by the notes.
- **`Makefile`** — root build/run harness for all three packages.

## Topic index (notes)

One notes file per package — a class's note is always in the file matching its folder:

| File | Covers |
|---|---|
| `Data_structures_notes.md` | Arrays, linked lists, stack/queue, heap & priority queues, union-find, BST & AVL, hash tables, Fenwick tree, suffix array, indexed PQ, sparse table, and both graph representations (adjacency matrix & list) |
| `Algorithms_notes.md` | Suffix-array string algorithms (unique substrings, LRS, LCS), divide & conquer + merge sort, graphs overview, DFS/BFS, tree algorithms (rooting, leaf sum, center, isomorphism, LCA), topological sorts, DAG paths, Dijkstra, Bellman-Ford, Floyd-Warshall, Tarjan SCC, Eulerian path, Prim, and the four max-flow algorithms |
| `Problems_notes.md` | Recursion/D&C exercises (multiplication, list sum, string reversal, max-2D, three-way min), DP problems (magical cows, tilings, mountain scenes, narrow art gallery, knapsacks), grid shortest path, Held-Karp TSP, and bipartite matching via max flow (Mice and Owls, Elementary Math) |

## Conventions

Every class follows the same rules so the codebase reads consistently:

- **Generic backing** via `Object[]` with `@SuppressWarnings("unchecked")` casts on read (Java erasure rules out `new E[n]`). Numeric-only structures (e.g. `Fenwick_tree`) use a primitive backing like `long[]` instead.
- **Contract-only comments** above each method — what it does, returns, and throws — with no implementation hints, so the method body is the exercise.
- **Self-contained test runner**: each file declares a `<Name>_Main` class with `checkEquals`, `checkTrue`, and `checkThrows` helpers that print PASS/FAIL inline, tally results, print a summary, and `System.exit(1)` on any failure. Tests cover edge cases (empty, single element, bounds, null-safety) and, where useful, cross-check against a brute-force reference; test constants are verified against an independent method before being trusted.
- **Null-safety** with `java.util.Objects.equals` for value comparisons.
- Structure-specific touches, e.g. doubly linked lists carry a `toStringReverse` that walks `prev` pointers to verify back-pointer maintenance.

## Build & run

Requires a JDK (developed and tested on JDK 21; a fresh container needs `apt-get install -y default-jdk-headless`). From the repo root:

```sh
make run F=Stack              # compile everything, run Data_structures.Stack_Main, then clean
make run F=Magical_cows       # package auto-detected (Problems here)
make run-latest               # run the most recently modified file's _Main
make run-all                  # run every *_Main, log to test-results.log, print only failures
make list                     # list runnable classes, newest first
make clean                    # delete all .class files
```

`make run` picks the package automatically based on which folder contains `<F>.java`, then auto-cleans the generated `.class` files. Interface/type files with no `_Main` are skipped by `run-all` automatically.

## Notes format

Data-structure notes use a three-part template — **Positives / Negatives / Algorithm (thought process)** — and algorithm/problem notes use **Idea / Complexity / Notes**. Notes live in three root-level files mirroring the three packages, each with a grouped table of contents. Complexity claims describe *this repo's implementations*; faster textbook variants are noted as upgrades. Where an implementation has a known deviation or bug, the note carries a **Code flag** line pointing at the reconciliation report.

## How Claude is used in this project

Claude (Anthropic) is used as a coding tutor and scaffolding tool, not as a code generator that does the work:

- **Skeletons.** Claude produces the class scaffold — fields, contract-only method stubs, and a complete `<Name>_Main` test runner (known-answer cases plus brute-force cross-checks) — matching the conventions above. The method bodies are left empty for me to implement.
- **Socratic debugging.** When I share buggy code, Claude points at the bug and walks the failing trace, edge cases, and complexity trade-offs rather than handing over a fix. It only writes the fix when I explicitly ask.
- **Notes.** Claude converts my handwritten notes into the templates above, flagging and correcting errors rather than copying them verbatim, and periodically reconciles the notes against the implemented code (the code is the source of truth).

The intent is that the learning happens while filling in the skeletons and chasing down the test failures; the scaffolding just removes boilerplate and keeps the structure uniform.

## Study resources & curriculum

### Coding / DSA spine
- **NeetCode 150** — https://neetcode.io/practice
- **Grokking the Coding Interview** (pattern-based prep) — https://www.designgurus.io/course/grokking-the-coding-interview (also on Educative: https://www.educative.io/courses/grokking-coding-interview)
- **William Fiset — Data Structures** (YouTube) — https://www.youtube.com/playlist?list=PLDV1Zeh2NRsB6SWUrDFW2RmDotAfPbeHu — *completed*
- **William Fiset — Graph Theory** (YouTube) — https://www.youtube.com/playlist?list=PLDV1Zeh2NRsDGO4--qE8yH72HFL1Km93P — *completed*
- **William Fiset — Algorithms** (companion Java repo) — https://github.com/williamfiset/Algorithms
- **Abdul Bari — Algorithms** (YouTube) — https://www.youtube.com/playlist?list=PLAPEtbmG9XgTQqVYWAgAR6MilRB93OeMQ — *up next*

<!-- ### ML / math track
- **Introduction to Statistical Learning (ISL)** — https://www.statlearning.com
- **Harvard Stat 110 — Probability** (Joe Blitzstein) — https://stat110.hsites.harvard.edu/ · lectures: https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo · free book: https://probabilitybook.net
- **Andrej Karpathy — Neural Networks: Zero to Hero** — https://karpathy.ai/zero-to-hero.html · playlist: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- **Aurélien Géron — Hands-On Machine Learning** (ch. 10–16; code repo) — https://github.com/ageron/handson-ml3
- **Chip Huyen — Designing Machine Learning Systems** — https://huyenchip.com/books/ · **Stanford CS329S** — https://stanford-cs329s.github.io/
- **Khang Pham — Machine Learning Interviews** (late-stage drilling) — https://mlengineer.io/ -->

---

*Environment: macOS, zsh. Links above were checked against their sources; the ML-track entries should be re-verified periodically as course pages move.*