## Traveling_salesman

**Problem.** Given `n` cities and the cost to travel between each pair (an `n × n` matrix), find the cheapest tour that starts at a city, visits every city exactly once, and returns to the start (a minimum-cost Hamiltonian cycle). Works for directed/asymmetric costs (`dist[i][j]` may differ from `dist[j][i]`).

### Idea
Brute force tries all `(n-1)!` orderings — hopeless past ~12 cities. The **Held-Karp** DP does far better by memoizing over *sets* of visited cities, exploiting that the cost of finishing a tour depends only on which cities remain and where you currently are, not on the order you visited the earlier ones.

State: `dp[mask][i]` = minimum cost of a path that starts at city 0, has visited exactly the set of cities in the bitmask `mask`, and currently sits at city `i` (with bit `i` set in `mask`).

```
base:       dp[{0}][0] = 0                       (start at 0, only 0 visited)
transition: for each state (mask, i) and each city j not in mask:
              dp[mask | (1<<j)][j] = min(that, dp[mask][i] + dist[i][j])
answer:     min over i of dp[fullMask][i] + dist[i][0]   (close the loop back to 0)
```

The bitmask encodes the visited set; fixing the start at city 0 removes the `n`-fold rotational symmetry (every cycle can be written starting at 0).

### Complexity
O(2ⁿ · n²) time — `2ⁿ` masks × `n` current-cities × `n` next-cities — and O(2ⁿ · n) space. Practical up to about n = 18–20. This is still exponential (TSP is NP-hard); Held-Karp is just far better than `(n-1)!` factorial brute force.

### Notes
- **Fix the start at city 0.** A tour is a cycle, so its cost is independent of where you "start" describing it; pinning start = 0 avoids counting each cycle `n` times and anchors the base case.
- **Unreachable states need a sentinel.** Initialize `dp` to a large "infinity," and when combining `dp[mask][i] + dist[i][j]`, skip states still at infinity so they don't leak a bogus finite cost. Pick INF large enough to dominate real tours but small enough that `INF + dist` doesn't overflow `int` — or use `long`.
- **Bit tricks:** `mask & (1<<i)` tests membership; `mask | (1<<j)` adds city `j`; `fullMask = (1<<n) - 1`. Iterate masks in increasing numeric order so every predecessor `mask` is finished before the states it extends.
- **n = 1** is a degenerate tour of cost 0. **n = 2** is just `dist[0][1] + dist[1][0]`.
- **Reconstruction** (the actual tour, not just the cost) works like knapsack's: store parent pointers or re-derive by checking which predecessor achieved each `dp` value, walking back from the closing state.
- Sample checks (independently verified): symmetric 4-city → `80`, asymmetric 3-city → `3`.
