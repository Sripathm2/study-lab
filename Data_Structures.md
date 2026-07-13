# Data Structures — Implementation Notes

Positives, negatives, and algorithm thought-process for each structure built from scratch in the `Data_structures/` package (including the two `Graph` implementations — they are structures, so their notes live here; the algorithms that run on them are in `Algorithms_notes.md`). No code — contracts and reasoning only. Where the implementation deviates from the intended design, a **Code flag** line points at it (details in `01_Mismatch_report.md`).

## Contents
1. [Static_array](#static_arraye)
2. [Dynamic_array](#dynamic_arraye)
3. [Singly_linked_list](#singly_linked_liste)
4. [Doubly_linked_list](#doubly_linked_liste)
5. [Stack](#stacke)
6. [Queue](#queuee)
7. [Heap](#heape-extends-comparablee)
8. [Priority_queue](#priority_queuee)
9. [Union_find](#union_find)
10. [Union_find_compressed](#union_find_compressed)
11. [Binary_search_tree](#binary_search_treee-extends-comparablee)
12. [Hash tables (chaining + double hashing)](#hash-tables-hash_table_chaining-and-hash_table_double_hashing)
13. [Fenwick_tree](#fenwick_tree-binary-indexed-tree)
14. [Suffix_array + LCP array](#suffix_array--lcp-array)
15. [Balanced binary search trees](#balanced-binary-search-trees)
16. [AVL tree](#avl-tree)
17. [Indexed priority queue](#indexed_priority_queue)
18. [Sparse table](#sparse_table)
19. [Graph_adjacency_matrix](#graph_adjacency_matrix)
20. [Graph_adjacency_list](#graph_adjacency_list)

---

## Static_array<E>

Fixed-capacity contiguous backing. Index maps directly to a memory offset.

### Positives
- Access by index O(1) — direct offset, no walking.
- Set into any slot is O(1) (no resize ever happens).
- Lowest memory overhead of any structure — no pointers, no spare capacity.
- Cache-friendly: contiguous layout.

### Negatives
- Fixed size. Can't grow; full means full.
- Search is O(n) (unsorted) — `contains`/`indexOf` scan the whole backing.
- No insert/remove-with-shift here at all — this class is purely get/set over a fixed block; shifting semantics live in the dynamic array.

### Algorithm / thought process
Backing is one `Object[]` of fixed length; `size()` reports the capacity, and every slot exists from the start (holding `null` until set). Index `i` is a direct lookup — that's the whole point: O(1) get and O(1) set, both bounds-checked (negative capacity is also rejected at construction).

`contains`/`indexOf` scan all `capacity` slots with `Objects.equals`, so `null` is a searchable value and `indexOf` returns the first match or −1.

No resizing logic at all — capacity is set once at construction. If you need growth or element shifting, that's the dynamic array.

---

## Dynamic_array<E>

Static array that resizes itself. Tracks `size` (used) separately from `capacity` (allocated).

### Positives
- Access by index O(1) — same direct-offset trick as static.
- Append amortized O(1) — most appends are free, the occasional resize averages out.
- Resizable — no fixed cap; grows on demand (a zero-capacity array bootstraps to 1, then doubles).

### Negatives
- Insert/remove in the middle is O(n) — still has to shift.
- Resize cost is O(n) when it triggers (allocate new array, copy `size` elements over).
- Over-allocates: capacity is usually larger than size, so wasted memory.

### Algorithm / thought process
Two numbers matter: `size` and `capacity`. Append writes at index `size`, then `size++`. When `size == capacity`, double the capacity (0 → 1 → 2 → …), allocate a new array, copy elements, swap it in.

Why doubling and not +1? Amortization. Doubling means resizes happen at 1, 2, 4, 8, … — geometrically rare. Total copy work across n appends sums to ~2n, so per-append cost averages to O(1). That's the "amortized" claim, not worst-case.

The *intended* shrink policy is hysteresis: halve only when size drops to ~¼ of capacity, so grow and shrink thresholds don't sit next to each other and thrash on add/remove at the boundary.

**Code flag (B2):** the implemented condition is inverted — `remove` resizes to `2·size` whenever `size*4 > capacity`, i.e. when the array is *more* than quarter-full. In practice: a remove can *grow* capacity (10 → 16 at size 8), most removes trigger a resize, and a below-¼-full array never shrinks. Also, `remove(int)` nulls `arr[size]` rather than the vacated `arr[size-1]`, leaving a loitering reference at the old last slot.

`add(int index, …)` shifts right from the end to open the slot (accepts `index == size` as an append); `remove(int)` shifts left and returns the removed value; `remove(E)` is `indexOf` + `remove(int)`. `clear()` resets capacity to the default and zeroes `size`.

---

## Singly_linked_list<E>

Chain of nodes, each holding a value and a `next` pointer. Head and tail pointers are maintained.

### Positives
- Insert/remove at head O(1) — just repoint head.
- Append O(1) via the tail pointer.
- Truly dynamic size, node by node — no resize, no copy.
- Less memory than doubly (one pointer per node).

### Negatives
- Access/search by index O(n) — must walk from head; no random access.
- Can't traverse backward — only one direction.
- Remove tail is O(n) even with a tail pointer — you need the *previous* node to fix its `next`, and there's no back-link to find it. The implementation walks from head until `node.next == tail`.

### Algorithm / thought process
Each node points only forward. You hold `head` and `tail`.

Insert at head: new node's `next` = old head, then head = new node. Constant. Remove head: head = head.next. Constant. `add(int index, …)` walks to the node before `index` and splices; index 0 and index `size` route to `addFirst`/`addLast`.

Tail is the asymmetry. Appending is cheap with the tail pointer, but *removing* the tail means the new tail is the second-to-last node — found only by walking the whole list, because no node knows its predecessor. That O(n) tail-removal is exactly what the doubly linked list fixes.

`reverse()` is done in place with a three-pointer walk (prev/current/future), then head and tail are swapped — no rebuild, O(n), O(1) extra space.

**Code flag (B3):** `removeFirst` on a single-element list nulls `head` but leaves `tail` pointing at the removed node — a subsequent `getLast()` on the now-empty list returns the stale value instead of throwing. (`addLast` self-heals via its `head == null` branch, which is why the runners pass.)

---

## Doubly_linked_list<E>

Linked list where each node has both `next` and `prev`. Maintains head and tail.

### Positives
- Traverse in both directions.
- Remove a known node in O(1) — you have its `prev`, so you can splice it out without walking.
- O(1) insert/remove at *both* head and tail (tail removal is cheap here, unlike singly).

### Negatives
- ~2× pointer memory (every node stores `prev` too).
- More bookkeeping — every insert/remove must fix up to four pointers (the node's prev/next and the neighbors' links), easy to get wrong.

### Algorithm / thought process
The `prev` pointer buys two things: backward traversal, and O(1) tail removal. Removing the tail just needs `tail = tail.prev; tail.next = null` — no walk, because the back-link is already there.

The cost is discipline. Any splice touches the node and both neighbors. Insert between A and B: `new.prev=A, new.next=B, A.next=new, B.prev=new`. Miss one and the list corrupts silently in one direction.

This is why `toStringReverse` (walk via `prev` from tail) is a useful test — it catches a broken back-link that a forward-only `toString` would never reveal. The runner exercises it after every mutation.

Stack and Queue are both built on this class: head/tail O(1) access in both directions covers LIFO and FIFO cleanly.

---

## Stack<E>

LIFO (last in, first out). A thin adapter over `Doubly_linked_list`: push/pop/peek all hit the head.

### Positives
- push, pop, peek all O(1).
- Dead simple — one access point (the top).
- Natural fit for DFS, recursion unrolling, undo, expression/bracket matching.

### Negatives
- No random access — only the top is reachable in O(1).
- `contains` exists (delegated to the list) but is an O(n) scan; there's no faster membership test.

### Algorithm / thought process
LIFO means the only interesting end is the top. Mapping onto the doubly linked list: push = `addFirst`, pop = `removeFirst`, peek = `getFirst` — all O(1) because head ops are O(1). Empty pop/peek throw `NoSuchElementException`.

`toString` uses the list's `toStringReverse`, so the printout reads bottom → top (the top of the stack is the last element shown).

Could equally back it with a dynamic array (push = append, pop = remove last) — also amortized O(1). The linked-list version avoids resize cost; the array version is more cache-friendly. Either satisfies the contract.

Why DFS uses it: you push neighbors and pop the most recent, so you dive deep before going wide — depth-first by construction.

---

## Queue<E>

FIFO (first in, first out). A thin adapter over `Doubly_linked_list`: enqueue at tail, dequeue from head.

### Positives
- enqueue, dequeue, peek all O(1); `peekLast` (the back of the line) also O(1).
- Simple — two access points (front and back), each fixed.
- Natural fit for BFS, scheduling, buffering, producer/consumer.

### Negatives
- No random access — only front and back are O(1).
- `contains` exists but is an O(n) scan, same as the stack.

### Algorithm / thought process
FIFO means you add at one end and remove from the other. Mapping onto the doubly linked list: enqueue = `addLast`, dequeue = `removeFirst`. Both O(1). A doubly linked backing makes *both* ends symmetric — though note a singly linked list with a tail pointer would also serve FIFO fine (O(1) `addLast`, O(1) `removeFirst`); it's only tail *removal* that singly can't do cheaply, and a queue never removes at the tail. The doubly backing is used here because it already existed and gives `peekLast` for free.

Why BFS uses it: you enqueue neighbors and dequeue the oldest, so you finish an entire level before descending — breadth-first by construction. Stack→DFS, Queue→BFS is the pairing to remember.

---

## Heap<E extends Comparable<E>>

Binary min-heap stored in an array. Complete binary tree: every level full except the last, which fills left to right, no gaps.

### Positives
- peek min O(1) — it's always at index 0.
- add and poll O(log n) — only one root-to-leaf path moves.
- Array-backed: no node objects, no pointers, cache-friendly. Children are computed, not stored. The backing doubles when full.
- Build from an array (heapify constructor) in O(n), better than n inserts.

### Negatives
- Arbitrary `contains`/search is O(n) — heap order says nothing about left/right siblings, so you can't binary-search it.
- Not sorted — only the min is positioned; the rest is partially ordered.
- Only the extreme is accessible; no efficient access to the k-th element without extracting.
- Arbitrary `remove(value)` is O(n): a linear scan finds the index, then swap-with-last + sift restores the heap. A value→index map would make the *find* O(1); this implementation doesn't keep one.

### Algorithm / thought process
Index arithmetic replaces pointers (0-based):
- parent(i) = (i − 1) / 2  (integer division)
- left(i)   = 2i + 1
- right(i)  = 2i + 2

Heap invariant (min-heap): every parent ≤ its children. Says nothing about left vs right — that's why search isn't logarithmic.

**add:** null is rejected; grow if full; put the new value at the end of the array (next open leaf), then **sift up** — while it's smaller than its parent, swap with the parent. One path up, O(log n).

**poll:** the answer is index 0. Move the last element into the root, shrink size by one, then **sift down** from the root — repeatedly swap with the *smaller* of the two children while it's larger than that child (when only a left child exists, it stands in for both). One path down, O(log n).

**Up or down?** After a replacement you only ever need one direction. A freshly inserted leaf only violates upward → sift up. A new root after poll only violates downward → sift down. Picking the smaller child on the way down is what preserves the min property.

**Heapify constructor:** copies the input, then walks the non-leaf nodes from the last parent (`size/2 − 1`) back to index 0, sifting each. Bottom-up sift-down sums to O(n) because most nodes sit near the bottom with short paths.
**Code flag (C3):** the constructor sifts each node down *and then up*; the sift-up is redundant for bottom-up heapify (correct result, extra work).

**remove(value):** scan for the index (O(n), `Objects.equals`), swap the target with the last element, shrink, then sift the swapped-in stranger both up and down (only one will move it). Returns false for null or absent values. `get(index)` exposes raw slots — used by the priority queue's scans, not part of the heap contract proper.

---

## Priority_queue<E>

A heap ordered by an explicit priority instead of the natural ordering of `E`. Wraps `Heap` via an `Entry` wrapper that carries the value plus the comparator used to rank it.

### Positives
- Order by any priority you define (a `Comparator`), not just `Comparable`; the no-arg constructor defaults to natural order.
- offer and poll O(log n); peek O(1) — inherits the heap's costs. The array constructor heapifies wrapped entries in O(n).
- Decouples "what to store" from "how to rank it."

### Negatives
- Same as the heap underneath: arbitrary search/`contains` O(n), not fully sorted, only the head is cheap.
- Extra wrapper object per element (the `Entry`) — small memory and indirection overhead.
- `remove(value)` scans the entries (O(n)) and then calls the heap's own O(n) remove — two passes.

### Algorithm / thought process
The heap needs a notion of "smaller." A bare `Heap<E>` uses `E`'s `compareTo`. The priority queue instead takes a `Comparator` and ranks by that: each `Entry` holds the value and the comparator, and `Entry.compareTo` delegates to `cmp.compare(this.value, other.value)` — so the untouched heap machinery orders entries exactly as the comparator dictates.

Min vs max is just the comparator's direction — reverse the comparator (or negate priorities) and nothing else changes; sift up/down still picks the "smaller" entry, where "smaller" is whatever the comparator says.

Everything else — offer = add + sift up, poll = swap root with last + sift down — is the underlying heap verbatim.

---

## Union_find

Disjoint-set structure over ids 0..n−1. `find` tells which set an element belongs to (its root); `union` merges two sets. This version uses union by rank, no path compression.

### Positives
- Tracks connected components cheaply; O(log n) union/find with rank.
- Tiny memory: parent array + rank array + a component count.
- The backbone of Kruskal's MST and any "are these connected / how many groups" problem.

### Negatives
- No path compression here, so finds re-walk the tree each time — O(log n) with rank, worse if you skipped rank.
- Can't un-union (no split). Merges are permanent.
- `componentSize` is O(n) — it re-finds every element and counts matches (no size array is maintained).

### Algorithm / thought process
A `parent[]` array. Each element points at its parent; a root points at itself and *is* the representative of its set.

**find(x):** bounds-checked, then follow `parent` links up until you hit a node that points to itself. (The walk starts from `parent[x]`, which is equivalent — a root's parent is itself.)

**union(x, y):** first `connected(x, y)`; if already merged, return false. Otherwise find both roots again and attach one under the other. **Union by rank:** attach the lower-rank root under the higher-rank one; on a rank tie, `y`'s root goes under `x`'s root and `x`'s rank bumps by one. Rank keeps trees shallow → find stays O(log n) instead of degrading to an O(n) chain. (Micro-nit: the `connected` pre-check plus the two explicit finds means four find-walks per union — correct, just redundant work.)

**connected(x, y):** `find(x) == find(y)`. **components:** start at n, decrement on every successful union.

**Negative-size trick (space/bookkeeping optimization — not what this code does):** fold the metadata *into* the parent array. A root stores the **negative of its tree's size** (`parent[root] = -size`) instead of pointing at itself; non-roots store a plain parent index. Then `parent[x] < 0` *is* the root test, `-parent[root]` answers `componentSize` in O(1) after a find (versus the O(n) re-scan this implementation does), and union-by-size falls out naturally: attach the smaller tree under the larger and add the sizes (`parent[bigRoot] += parent[smallRoot]` — negatives add correctly). One array replaces parent + rank + any size array; initialization is all `-1` (each node a root of size 1). Union by size gives the same O(log n) height guarantee as union by rank, and combined with path compression the same α(n). The only subtlety: every "is this a root" check must become `parent[x] < 0` — the `parent[x] == x` idiom no longer holds.

**Why it matters for Kruskal:** sort edges by weight, walk them cheapest-first, and add an edge only if its two endpoints have *different* roots (otherwise it would close a cycle). Each "add" is a union. Union-find is what answers "would this edge close a cycle?" in near-constant time. (MST = minimum-total-weight edge set connecting all vertices; not unique when weights tie.)

---

## Union_find_compressed

Same disjoint-set structure, plus **path compression** in `find`. With rank + compression, operations run in amortized α(n) — inverse Ackermann, effectively constant for any real input.

### Positives
- Amortized α(n) per operation — the fastest practical disjoint-set; α(n) ≤ 4 for any n you'll ever see.
- Trees flatten over time, so repeated finds on the same elements become O(1).
- Same tiny memory footprint as the uncompressed version.

### Negatives
- A `find` now *writes* (rewrites parent pointers), so it's not a pure read.
- Still no un-union; merges are permanent.
- The α(n) bound is amortized, not per-call worst-case — a single early find can still walk a longer path before it flattens.

### Algorithm / thought process
Identical to plain union-find for `union`/`connected`/`componentSize` (including the O(n) `componentSize` — the negative-size trick described in the Union_find note would make it O(1) here too). The change is in `find`.

**find(x), two-pass compression (as implemented):** first walk up from `parent[x]` to locate the root; then walk the same path a second time, repointing each visited node directly at the root. Next time any of them is queried, it's one hop away. (The second walk starts at `parent[x]`, so `x`'s own pointer isn't rewritten — `x` stays one extra hop out; everything above it flattens. Harmless.) The alternative one-liner is the recursive `parent[x] = find(parent[x])`; same flattening, this repo uses the iterative two-pass.

Combined with union by rank, this gives the α(n) amortized guarantee — rank keeps trees shallow on the way up, compression collapses them on the way down. Either alone is good; together they're effectively constant. (One classical footnote: compression makes stored ranks upper bounds on true heights rather than exact — that's expected and fine.)

---

## Binary_search_tree<E extends Comparable<E>>

Ordered tree. At every node: all of left subtree < node < all of right subtree (duplicates are meant to be rejected — but see the code flag). Note "BST" ≠ complete tree — it can be any shape.

### Positives
- Search, insert, delete all O(log n) on average.
- In-order traversal gives sorted output for free.
- Dynamic size; supports min, max, height, and all four standard traversals.

### Negatives
- O(n) worst case — inserting already-sorted data builds a degenerate chain (a linked list).
- Not self-balancing on its own (AVL fixes this — next sections).
- Pointer overhead per node, and no random access by index.

### Algorithm / thought process
Ordering invariant holds at *every* node, not just the root — that's what makes the binary search work at each step.

**Insert (iterative):** null is rejected; an empty tree just takes the root. Otherwise walk down comparing — right when larger, left when smaller — until reaching a node whose relevant child slot is free, then attach there. An equal value found *during the walk* returns false without touching size.

**Code flag (B1):** the descent loop exits when it reaches a leaf (or when the tree is a single node) *before* the equality check runs, so a duplicate of a **leaf** value (or of a lone root) is silently inserted as a left child — violating the ordering invariant and the no-duplicates contract that `remove` relies on. Internal-node duplicates are rejected correctly. (The AVL tree avoids this by running a full `contains` before inserting.)

**Contains (iterative):** the same walk; found on an equal comparison, absent when you fall off.

**min / max:** walk all the way left (min) or all the way right (max); empty tree throws.

**Remove (recursive):** the public method pre-checks `contains` (an extra O(h) pass), then delegates to a helper that finds the node by the same comparison walk and handles it by child count:
- *Leaf:* return null to the parent.
- *One child:* splice — return the single child to the parent.
- *Two children:* don't delete the node — overwrite its value with its in-order **successor** (the leftmost value of the right subtree, via `right_smallest_child`), then recursively remove that successor from the right subtree. The successor has no left child, so its own removal lands in the leaf or one-child case. No orphans, no infinite recursion. (Predecessor — largest in left subtree — works symmetrically.)

The trick that makes remove clean: the recursive helper takes a subtree and **returns the new subtree root**, and the caller reassigns its child pointer. That return-and-reassign handles all the relinking, so you never manually null out parent pointers.

**Traversals** (same recursion skeleton, only the visit position moves):
- **in-order** (left, node, right) → sorted output; the defining BST traversal.
- **pre-order** (node, left, right) → serialize / copy a tree.
- **post-order** (left, right, node) → process children before parent (safe deletion).
- **level order** → breadth-first: a queue seeded with the root; dequeue a node, record it, enqueue its children. Produces the tree top-down, left-to-right per level.

`height()` is computed recursively (empty tree = −1, leaf = 0).

### Interview hooks
- Validate a BST → in-order must be strictly increasing.
- Kth smallest → in-order, stop at k.
- Lowest common ancestor in a BST → walk down comparing both targets to the current node.

---

## Hash tables (`Hash_table_chaining` and `Hash_table_double_hashing`)

Key→value stores backed by an array, with a hash function mapping keys to slots. `H(x)==H(y)` is necessary but not sufficient — `x` *might* equal `y` (collision), but `H(x)!=H(y)` guarantees `x != y`. So after hashing to a slot you must still confirm with `equals`. Keys must be immutable — a key whose hashCode changes after insertion is lost. Both classes hash with `(key.hashCode() & 0x7fffffff) % capacity` (the mask keeps the index non-negative) and reject null keys and null values.

### Positives
- Average O(1) put / get / remove / containsKey.
- Any immutable, hashable key; ideal for frequency counts, dedup, memoization, set membership.

### Negatives
- Worst case O(n) — a degenerate hash or adversarial keys collapse everything into one chain/probe run.
- No ordering: no sorted iteration, no range/min/max queries (that's a tree's job).
- **Chaining (this repo): capacity is fixed** — there is no resize, so chains grow without bound as load rises and operations degrade toward O(chain length). Only the double-hashing table resizes.
- Resize (double hashing) is O(n) — a periodic latency spike.
- Wasted space: open addressing keeps load ≤ 0.5 (half the slots empty); chaining adds a pointer per entry.

### Algorithm / thought process

**Separate chaining (`Hash_table_chaining`).** Each bucket holds the head of a singly linked `Entry{key, value, next}` chain. `put` walks the chain: an existing key gets its value replaced (old value returned); otherwise the new entry is appended at the chain's tail and size bumps. `get`/`containsKey` walk and compare with `equals`. `remove` pre-checks `containsKey`, then unlinks — a head match repoints the bucket, otherwise the walk repairs `prev.next`. Capacity is whatever the constructor got (default 16) forever.

**Open addressing with double hashing (`Hash_table_double_hashing`).** Entries sit directly in the array; each slot is `null` (never used), a shared `TOMBSTONE` marker (deleted), or an `Entry`.

- **Probe sequence:** `position(k, x) = (H1(k) + x·step) mod capacity`, where `step = P − (H1'(k) mod P)` and `P` is the largest prime **below** the capacity (recomputed after every resize; tiny tables fall back to step 1 = linear probing). Because `step ∈ [1, P]`, it can never be 0 — no explicit zero-guard needed. Using a second hash for the step spreads probes and breaks up the primary clustering linear probing suffers from.
- **Coprimality caveat (code flag C6):** a probe sequence covers all slots only when gcd(step, capacity) = 1. Capacity starts prime (11) but *doubles* on resize (22, 44, …), so an even step shares a factor 2 with the capacity and its cycle covers only half the table. The ≤ 0.5 load factor makes a stuck insert unlikely, but the "keep step and size coprime after growth" rule is not actually upheld — keeping capacity prime across resizes would fix it.
- **Deletion → tombstones:** you can't just empty a slot — that severs the probe chain and a later search would stop early and wrongly report "not found." `remove` therefore stamps the slot `TOMBSTONE`. Searches skip tombstones and keep probing; `put` may reuse one; resize purges them all.
- **Relocation-on-probe optimization:** `put`, `get`, and `containsKey` all remember the *first* tombstone met along the probe; when the key then turns up further along, the entry is copied back into that tombstone slot and the original slot is stamped `TOMBSTONE` — shortening future probes for that key. `remove` leans on this: it calls `containsKey` first (which relocates the entry to the earliest tombstone on its probe path), then re-probes knowing the prefix holds only live entries.
  **Code flag (B8):** in `put`'s insert-into-null case, the same swap converts a chain-terminating **null** slot into a `TOMBSTONE` — the entry should simply be placed in the remembered tombstone slot and the null left alone; as written, empty slots erode into tombstones and everyone's probes lengthen until a resize. Separately, `toString` only checks slots against `null`, so it casts a `TOMBSTONE` to `Entry` and throws `ClassCastException` whenever any remove has happened.
- **Resize / rehash:** when `size/capacity ≥ 0.5`, capacity doubles, `P` is recomputed, and every live entry (tombstones skipped) is re-`put` into the fresh table — indices depend on capacity, so they all move.

---

## Fenwick_tree (Binary Indexed Tree)

Array structure for prefix/range sums over a mutable array of `long`s. A plain prefix-sum array gives O(1) queries but O(n) updates; a Fenwick tree balances both at O(log n). Fixed size — you can't add or remove indices after construction.

Each cell is responsible for a block of elements determined by its **least significant bit**, `LSB(i) = i & (−i)`. Cell `i` covers `LSB(i)` elements, the range `(i − LSB(i), i]`:
- `12 = 1100`, LSB = 4 → responsible for 4 cells (indices 9–12).
- `10 = 1010`, LSB = 2 → responsible for 2 cells (9–10).
- `11 = 1011`, LSB = 1 → responsible for itself only (11).

See ![Fenwick walk](./Images/Fenwick.png) for the 0–7 range walk.

### Positives
- Prefix sum, range sum, point update (`update` adds a delta; `set` overwrites by computing the delta from a self-range-sum) all O(log n); construction from values O(n).
- Tiny and simple: a single `long[]`, no nodes or pointers — much lower constant factor than a segment tree for sum-type queries.

### Negatives
- Fixed size — indices can't be added or removed after construction.
- Range query works by subtracting prefixes, so it needs an **invertible** operation (sum, XOR). It can't do range min/max — that's segment-tree territory.
- Less flexible than a segment tree overall (no arbitrary range operations; basic form has no lazy propagation).
- 1-indexed internally; the public API is 0-based and shifts by +1, so the off-by-one lives in exactly one place.

### Algorithm / thought process
Core primitive: `LSB(i) = i & (−i)` — isolates the lowest set bit. Everything below is bit walking (internal 1-based indices).

**Prefix sum `prefixSum(i)`** — walk *down*. `sum = 0; while i > 0: sum += tree[i]; i −= LSB(i)`. Stripping the lowest set bit jumps to the cell covering the block just before this one, so you accumulate a logarithmic number of disjoint blocks — one step per set bit, O(log n).

**Range sum `rangeSum(l, r) = prefixSum(r) − prefixSum(l−1)`** (with `l == 0` short-circuiting to a plain prefix) — this subtraction is exactly why the operation must be invertible. `l > r` throws.

**Point update `update(i, delta)`** — walk *up*. `while i ≤ n: tree[i] += delta; i += LSB(i)`. Adding the LSB moves to the next cell whose responsibility range contains `i`, fixing every cell that includes this index. Also O(log n).

**O(n) construction** — load the raw values into `tree` (shifted to 1-based), then for each `i` push its accumulated value into its parent at `i + LSB(i)` (if the parent is in range). One pass; each cell contributes to exactly one parent, so the whole tree is built in linear time instead of n separate O(log n) updates.

**1-indexing** — the tree must be 1-based: `LSB(0) = 0`, so index 0 would never move and the loops would stall. The constructor allocates `n+1` slots and leaves `tree[0]` unused.

**Variants** (this implementation is the basic point-update + prefix-query form):
- range-update + point-query → store deltas using the difference-array trick on one BIT.
- range-update + range-query → maintain two BITs.

---

## Suffix_array (+ LCP array)

A **suffix** is a non-empty trailing part of a string; an n-character string has n suffixes. The **suffix array** is those suffixes sorted lexicographically, stored as their start indices. The **LCP array** rides alongside: `lcp[i]` is how many leading characters the sorted suffixes at ranks `i` and `i−1` share; `lcp[0] = 0`. Together they turn many naively-quadratic string problems into arithmetic over two arrays.

### Positives
- Simple and transparent: the whole build is "list the suffixes, sort them, compare neighbors" — easy to trust and to debug, which is the point at this stage.
- Unlocks the classic applications directly: distinct-substring counts, longest repeated substring, longest common substring, and pattern search by binary-searching the sorted suffixes.
- Exposes `suffixAt(rank)` for O(1) recovery of any sorted suffix (the strings are retained).

### Negatives
- **This is the naive construction** — it materializes all n suffix strings (O(n²) characters of memory!) and sorts them with the default string comparator: O(n² log n) time, since each comparison can scan O(n) characters. Fine for study-sized inputs; the efficient builds are the upgrade path.
- The LCP array is built by directly comparing each adjacent sorted pair character-by-character — O(n²) worst case (e.g. `"aaaa…"`), not Kasai's O(n).
- Static: built for one fixed string. Any edit means rebuilding.

### Algorithm / thought process
**Build (as implemented):** generate `text.substring(i)` for every `i`, sort the list lexicographically, and recover each rank's start index with `text.lastIndexOf(suffix)` — correct because a suffix's **last** occurrence in the text is the suffix itself (any later match would have to be a longer trailing block, impossible; earlier matches of the same characters are not at the end). `lcp[i]` then comes from a helper that walks the two adjacent sorted suffixes until they diverge, with a first-character fast-path returning 0.

**The upgrade path (not implemented):** **prefix doubling** sorts suffixes by their first 1, 2, 4, …, 2^k characters, reusing the previous round's ranks as sort keys — O(n log n) or O(n log² n) with O(n) integer storage. **Kasai's** builds the LCP in O(n) by walking suffixes in original string order and using the fact that the carried LCP drops by at most 1 per step. Linear-time SA builds (SA-IS, DC3) exist but are intricate. When the notes for `Unique_substrings` and friends quote complexities, they now quote the naive bounds this class actually has.

**Using it:**
- **Distinct substrings** — every substring is a prefix of exactly one suffix. There are `n(n+1)/2` prefixes in total, and adjacent sorted suffixes share `lcp[i]` leading characters counted twice, so `distinct = n(n+1)/2 − sum(lcp)`.
- **Longest repeated substring** — a repeat is a common prefix of two distinct suffixes, so the answer's length is `max(lcp)`.
- **Pattern search** — binary-search the sorted suffixes for the pattern.

---

## Balanced binary search trees

Self-adjusting BSTs that keep the height at O(log n) so every operation stays O(log n). A plain BST can degrade into a long chain (sorted-order inserts), collapsing to O(n). A balanced tree carries a **tree invariant** — a structural rule — and whenever an operation violates it, one or more **rotations** restore it. Different balanced trees differ only in their invariant and when they rotate (AVL: strict height balance; red-black: color rules).

### Positives
- Guaranteed O(log n) search/insert/delete regardless of insertion order — no degeneration to a chain.
- Keeps everything a plain BST offers: in-order traversal is sorted, plus min/max and range-style queries.

### Negatives
- Rotations and their maintenance add constant-factor overhead and real implementation complexity over a plain BST.
- Extra per-node metadata (height, or color/balance) and an update pass on the way back up every operation.
- Pointer-chasing structure; not as cache-friendly as a flat array.

### Algorithm / thought process
A **rotation** is a local, O(1) restructuring that changes a subtree's height while preserving BST ordering. Take two nodes where `B` is a child of `A`:

- **Right rotation** (lift the left child): `B = A.left; A.left = B.right; B.right = A; return B`.
- **Left rotation** is the mirror: lift the right child.

Returning the new subtree root (`B`) is the return-and-reassign pattern — the caller rewires its own child pointer to the returned node, so no parent-pointer bookkeeping is needed (same trick as recursive BST remove). Ordering is preserved because a right rotation only moves `B.right` (all keys between `B` and `A`) from `B`'s right to `A`'s left, which is exactly where they belong.

---

## AVL tree

A balanced BST with a strict per-node balance rule. Each node tracks a **balance factor** `BF = Height(right) − Height(left)`, which must stay in `{−1, 0, 1}`. Height is the number of edges from the node to its furthest leaf (null subtree = −1, leaf = 0). Any insert or delete that pushes a `BF` to ±2 triggers a rebalancing rotation.

### Positives
- Strictly balanced, so it has the tightest height bound of the common balanced trees (height ≤ ~1.44 log n) — the fastest lookups of the balanced family.
- Deterministic O(log n) worst case for search, insert, and delete.

### Negatives
- Rebalances aggressively to stay strictly balanced, so it does more rotations per insert/delete than a red-black tree — write-heavy workloads pay for it.
- Height/BF metadata on every node, plus an `update` + `balance` pass on every node along the path back up.
- Duplicate rejection here is a full `contains` pre-pass before every insert (an extra O(log n) walk — the safe-but-double-work approach; it's also what saves this class from the plain BST's leaf-duplicate bug).

### Algorithm / thought process
**Height and BF:** null height = −1, leaf height = 0. `update(node)` recomputes `node.height = 1 + max(height(left), height(right))` and the BF. (Implementation quirk: the `height` helper also *writes* the height it computes into the child it reads — a recompute-on-read; harmless, slightly redundant.)

**The four cases** (using `BF = right − left`), decided by the node's BF and the offending child's BF:
- `BF = −2`, left child `BF ≤ 0` → **left-left** → single **right** rotation on the node.
- `BF = −2`, left child `BF > 0` → **left-right** → **left** rotate the left child, then **right** rotate the node.
- `BF = +2`, right child `BF ≥ 0` → **right-right** → single **left** rotation on the node.
- `BF = +2`, right child `BF < 0` → **right-left** → **right** rotate the right child, then **left** rotate the node.

Read the child's BF to tell the "straight" case from the "bent" one. The `≤ 0` / `≥ 0` (rather than strict) matters for **deletion**: there the offending child can have `BF = 0`, and treating that as the straight case (single rotation) keeps the tree valid. During insertion the child is never balanced in a violating case, so the boundary is only exercised by deletes.

**Insert:** `contains` pre-check rejects duplicates, then recurse left/right and place the new node at a leaf. On the way back up, `update` then `balance` at each node. Each rotation is O(1); after rotating, the moved nodes are re-`update`d — the lower node first, then the new subtree root — so their heights are correct before the next level up looks at them.

**Delete:** identical to BST deletion (leaf / one-child splice / two-child successor-swap via `right_smallest_child`), then `update` + `balance` on every node along the path back up — the same machinery as insert. A single delete can cascade rotations up multiple levels, unlike insert which needs at most one rebalance.

---

## Indexed_priority_queue

A min-priority queue that also supports fast **update** and **delete** of an arbitrary key's priority — the operations a plain binary heap can't do without an O(n) scan. The trick is a bidirectional mapping: every key gets a stable **key index** `ki` in `[0, N)`, and two inverse maps translate between "which key" and "where it sits in the heap." Priorities are stored keyed by `ki` and never move; only heap *positions* move during swim/sink.

### Positives
- O(log n) insert, delete, and change-priority; O(1) `contains`, `valueOf`, and peek-min.
- Can reprioritize or remove an existing item in O(log n) — exactly what Dijkstra, Prim, and schedulers need (decrease-key on a known node).

### Negatives
- Keys must come from a fixed index domain `[0, N)` set at construction; arbitrary keys need an extra `key → ki` map layer.
- Three parallel arrays to keep in sync (`values`, `pm`, `im`) — more memory and bookkeeping than a plain heap, and easy to desync if a swap doesn't update every map.
- Inserting an already-present `ki` throws (`update` is the reprioritize path); null values are rejected.

### Algorithm / thought process
**The three arrays** (min-heap):
- `values[ki]` — the priority of key `ki`. Keyed by `ki`, so it **never moves**.
- `pm[ki]` (position map) — the heap position where key `ki` currently sits, −1 if absent.
- `im[pos]` (inverse map) — the key index living at heap position `pos`, −1 if empty. The heap itself is really `im`: a heap of key indices, ordered by their `values`.
- Invariant: `pm[im[pos]] == pos` and `im[pm[ki]] == ki` — `pm` and `im` are inverses.

**swap(i, j)** — the reason values don't move. A heap swap only exchanges *positions*: swap `im[i]` with `im[j]` (through a temp), then fix `pm` for both keys so the maps stay inverse. `values` is untouched.

**swim(i)** (move up) — while position `i`'s priority (`values[im[i]]`) is smaller than its parent's at `(i−1)/2`, swap and climb. Used after an insert or when a priority decreases.

**sink(i)** (move down) — pick the smaller-valued child of `2i+1`, `2i+2` (the left child stands alone when the right doesn't exist); while that child is smaller, swap and descend. Used after a delete or when a priority increases.

**insert(ki, value)** — set `values[ki] = value`; place `ki` at the end (`pm[ki] = sz; im[sz] = ki`); `swim` that slot; then bump `sz`. (Swimming before the size bump is fine — swim only looks upward and never consults `sz`.)

**update(ki, value)** — overwrite `values[ki]`, take `i = pm[ki]`, then call **both** `sink(i)` and `swim(i)`. Only one will actually move it, but the new priority could violate the heap in either direction, so you attempt both.

**delete(ki)** — take `i = pm[ki]`, shrink (`sz−−`), swap position `i` with the (new) last position `sz`, then `sink(i)` + `swim(i)` to settle the stranger that landed at `i`; finally clear the removed key's slots (`values[ki] = null`, `pm[ki] = −1`, `im[sz] = −1`). Edge case worth knowing: deleting the last position self-swaps, and the subsequent swim can't drag the dead slot back in — the element there was already ≥ its parent by the heap invariant. Deleting the min is just `delete(im[0])`.

**contains(ki)** — `pm[ki] != −1`. **valueOf(ki)** — `values[ki]`. **peek-min** — the key at `im[0]`.

---

## Sparse_table

Efficient range queries on a **static** array (data never changes). The idea: precompute the answer for every interval whose length is a power of two, then combine a couple of them to answer any query. This implementation is generic over `T` with a pluggable `BinaryOperator<T>` combine function, and additionally maintains an **index table** so min-queries can return the *position* of the minimum, not just its value (that argmin is what the LCA algorithm consumes).

Two properties of the combining function `F` matter:
- **Associative** — `F(F(A,B),C) == F(A,F(B,C))`. Required for a sparse table at all. (Commutativity is a different property; associativity is the one you need.)
- **Idempotent / overlap-friendly** — `F(x,x) == x`, so combining two *overlapping* ranges doesn't double-count. This is what gives **O(1)** queries. `min`, `max`, `gcd`, bitwise and/or have it. `sum`, `product`, `xor` are associative but **not** idempotent — for those, use the disjoint-block query at O(log n) (or just a prefix-sum array for plain sums).

### Positives
- O(1) range queries for idempotent ops after an O(n log n) build; O(log n) disjoint-block queries (`queryDisjoint`) for non-idempotent ops — both are implemented.
- Simple, flat, cache-friendly tables; no pointers, no rebalancing.
- The argmin variant (`queryOverlap_index_min_operation_only`) returns the index of the range minimum with leftmost-wins tie-breaking.

### Negatives
- **Static only.** Any element update invalidates the table; there is no cheap point update — you rebuild in O(n log n). If you need updates, use a segment tree.
- O(n log n) memory — heavier than a prefix-sum array for the ops a prefix sum can already handle.
- The index table is populated only when the stored values are `Integer`s and its comparisons hard-code min semantics (`≤`) — pass a different `T` or a non-min operator and the argmin method is meaningless. A quirk worth cleaning up eventually (e.g. a dedicated min-table subclass).

### Algorithm / thought process
**The table.** Let `P = floor(log2(N))` (computed via `Math.log` — fine at these sizes, though an integer bit-trick avoids any float worry). Build `P+1` rows × `N` columns where `table[i][j]` = `F` over the block `[j, j + 2^i)`. Cells whose block would run off the end are simply never filled or read.

**Construction by doubling.** A block of length `2^i` is two halves of length `2^(i−1)`:

```
table[0][j] = arr[j]                                   // row 0 is the array itself
table[i][j] = F(table[i-1][j], table[i-1][j + 2^(i-1)])
```

Alongside it, `index_table[i][j]` records *where* the winning value came from: copy the left half's index when `left ≤ right`, else the right half's — that `≤` is the leftmost-wins tie-break, and it's filled only for `Integer` values (see negatives).

**The log array.** `log[1] = 0`, `log[i] = log[i/2] + 1` up to N gives `floor(log2(len))` in O(1) per query — no `Math.log` at query time.

**Query `[l, r]` inclusive (`queryOverlap`).** Let `p = log[r − l + 1]`, `k = 2^p`. Cover the range with two length-`k` blocks — one starting at `l`, one ending at `r` (starting at `r − k + 1`). For idempotent `F` their overlap is harmless:

```
answer = F(table[p][l], table[p][r - k + 1])
```

Two lookups and one combine — O(1). The argmin query does the same two lookups and returns the corresponding `index_table` entry (left block on ties).

**Query `[l, r]` for non-idempotent ops (`queryDisjoint`).** Greedily peel power-of-two blocks from the left: for `p` from largest down to 0, if a `2^p` block fits in what remains, combine it and advance `l`. O(log n) combines, no overlaps, so sums/products/xor are safe here.

**The `<<` operator.** `1 << p` is the idiomatic `2^p` (all block sizes here are powers of two), so `r − (1 << p) + 1` is the start of the right-aligned block — faster and exact compared to `Math.pow`.

---

## Graph_adjacency_matrix

A graph stored as a `V × V` grid: cell `(u, v)` records whether edge `u → v` exists and, if weighted, its weight. Implements the common `Graph` interface so algorithms don't care which representation they're handed.

### Positives
- **O(1) edge and weight lookup** — single array reads.
- Simple and cache-friendly; O(1) to add or update an edge.
- Best for **dense** graphs (edges near `V²`), where the matrix is mostly full anyway.

### Negatives
- **O(V²) memory** regardless of how few edges exist.
- **Listing a vertex's neighbors is O(V)** — a full row scan even for one neighbor.
- Iterating all edges is O(V²). At `V = 10⁵` the matrix alone is `10¹⁰` cells — infeasible.

### Algorithm / thought process
Two `n × n` arrays: a boolean `present[u][v]` and an int weight matrix. Splitting presence from weight avoids the "is a 0 cell a weight-0 edge or no edge?" ambiguity.

- **addEdge(u, v[, w]):** bounds-check; bump the edge count **only if the cell was empty** (re-adding overwrites the weight without double-counting); set presence + weight; if **undirected**, mirror into `[v][u]` — both directions, every time, so mirrors can't desync.
- **hasEdge / weight:** direct reads; `weight` throws `NoSuchElementException` when absent. The unweighted `addEdge` stores weight 1.
- **neighbors(v):** scan row `v`, collect present columns — O(V), naturally ascending (this backend genuinely satisfies the interface's ascending-order contract).
- **edgeCount:** maintained on add; undirected edges count once.

The whole trade: memory and neighbor-iteration cost tied to `V`, in exchange for constant-time edge lookup.


---

## Graph_adjacency_list

A graph stored as, per vertex, a list of its outgoing edges (each a small `Edge{to, weight}`). Implements the common `Graph` interface; the default representation for most graph work.

### Positives
- **O(V + E) memory** — exactly the edges that exist. Ideal for **sparse** graphs.
- **Neighbor iteration is O(degree)** — exactly what DFS/BFS/Dijkstra hammer on.
- Scales to graphs where a `V × V` matrix wouldn't fit.

### Negatives
- **Edge/weight lookup is O(degree)** — a scan of `u`'s list (a matrix does it in O(1)).
- Slightly more object overhead per edge than a flat matrix cell.
- Neighbor order is whatever insertion produced — see the code flag.

### Algorithm / thought process
A list-of-lists: `adj.get(u)` is the edges leaving `u`.

- **addEdge(u, v[, w]):** scan `u`'s list for an existing edge to `v` — if found, the weighted overload updates that edge's weight in place and returns (no duplicate, no recount); the unweighted overload just returns. Otherwise append, bump the edge count once, and for **undirected** graphs also append the `v → u` mirror.
- **hasEdge / weight:** the same O(degree) scan; `weight` throws when absent.
- **neighbors(v):** map `v`'s edge list to destinations, **in insertion order**.
- **edgeCount:** incremented once per genuinely new edge (not once per stored direction).

**Code flags (C4):** two contract gaps. (1) The `Graph` interface promises `neighbors` in **ascending order**, but this backend returns insertion order — traversal determinism in the runners currently holds only because tests add edges in ascending order; either sort here or soften the interface comment. (2) The weighted re-add on an **undirected** graph updates only `u`'s copy and returns before touching the `v → u` mirror — the two directions can end up with different weights.

The trade is the mirror image of the matrix: cheap memory and neighbor iteration, at the cost of O(degree) edge lookups. Traversals iterate neighbors far more than they test single edges, so this is the right default for all but the densest graphs.