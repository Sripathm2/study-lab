package Algorithms;
import Data_structures.Graph;
import Data_structures.Graph_adjacency_list;
import java.util.Objects;

// Articulation points (cut vertices) of an UNDIRECTED graph, via one DFS with
// discovery indices and low-links.
// A vertex v is an articulation point if removing it (and its edges) increases
// the number of connected components.
//
// Low-link DFS: ids[v] = discovery index; low[v] = the smallest discovery index
// reachable from v's DFS subtree using any number of tree edges plus AT MOST ONE
// back edge. Rules (note the contrast with Tarjan SCC — here the tree-edge /
// back-edge distinction has real teeth):
//   - tree edge v -> child:  low[v] = min(low[v], low[child]) after recursing,
//       and NON-ROOT v is an articulation point if low[child] >= ids[v]
//       (child's subtree cannot climb above v without using v);
//   - back edge v -> w (visited, w != dfs parent): low[v] = min(low[v], ids[w])
//       — ids, NOT low: one back edge only, or cut vertices get missed;
//   - the DFS ROOT is a special case: it is an articulation point iff it has
//       two or more DFS CHILDREN (neighbor count is irrelevant).
// Parent tracking: skip only the edge you arrived by (in a simple graph, skipping
// the parent id suffices).
public class Articulation_points {

    // Return the ids of all articulation points of g, sorted ascending
    // (empty array if there are none). The graph is treated as undirected and
    // may be disconnected — every component is examined (a fresh DFS root per
    // component, each with its own root rule).
    // Throw NullPointerException if g is null.
    // Throw IllegalArgumentException if g is directed (g.isDirected()).
    // A vertex with no edges is never an articulation point.
    public static int[] findArticulationPoints(Graph g) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // --- suggested private helper (shape only; use, rename, or replace freely) ---

    // DFS from `node` with `parent` = the vertex the edge came from (-1 at a root).
    // Fills ids/low, sets isArticulation flags, and (at roots) counts children.
    private static void dfs(Graph g, int node, int parent) {
        throw new UnsupportedOperationException("not implemented yet");
    }
}

class Articulation_points_Main {
    private static int passed = 0;
    private static int failed = 0;

    private static void checkEquals(String name, Object expected, Object actual) {
        if (Objects.equals(expected, actual)) {
            passed++;
            System.out.println("PASS: " + name);
        } else {
            failed++;
            System.out.println("FAIL: " + name + " — expected <" + expected + ">, got <" + actual + ">");
        }
    }

    private static void checkThrows(String name, Class<? extends Throwable> expected, Runnable r) {
        try {
            r.run();
            failed++;
            System.out.println("FAIL: " + name + " — expected " + expected.getSimpleName() + ", nothing thrown");
        } catch (Throwable t) {
            if (expected.isInstance(t)) {
                passed++;
                System.out.println("PASS: " + name);
            } else {
                failed++;
                System.out.println("FAIL: " + name + " — expected " + expected.getSimpleName()
                        + ", got " + t.getClass().getSimpleName());
            }
        }
    }

    private static Graph build(int n, int[][] edges) {
        Graph_adjacency_list g = new Graph_adjacency_list(n, false);
        for (int[] e : edges) g.addEdge(e[0], e[1]);
        return g;
    }

    // Independent reference: v is an articulation point iff deleting it increases
    // the component count (isolated vertices excluded). O(V * (V + E)) brute force.
    private static int[] brute(int n, int[][] edges) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (int v = 0; v < n; v++) {
            boolean hasEdge = false;
            for (int[] e : edges) if (e[0] == v || e[1] == v) { hasEdge = true; break; }
            if (hasEdge && components(n, edges, v) > components(n, edges, -1)) out.add(v);
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int components(int n, int[][] edges, int skip) {
        boolean[] seen = new boolean[n];
        int count = 0;
        for (int s = 0; s < n; s++) {
            if (s == skip || seen[s]) continue;
            count++;
            java.util.ArrayDeque<Integer> stack = new java.util.ArrayDeque<>();
            stack.push(s);
            while (!stack.isEmpty()) {
                int x = stack.pop();
                if (seen[x]) continue;
                seen[x] = true;
                for (int[] e : edges) {
                    int y = e[0] == x ? e[1] : (e[1] == x ? e[0] : -1);
                    if (y != -1 && y != skip && !seen[y]) stack.push(y);
                }
            }
        }
        return count;
    }

    private static void checkCase(String name, int n, int[][] edges, int[] expected) {
        // expected constants below were derived from (and re-verified here against)
        // the independent delete-and-count brute force
        checkEquals(name + " (vs constant)", java.util.Arrays.toString(expected),
                java.util.Arrays.toString(Articulation_points.findArticulationPoints(build(n, edges))));
        checkEquals(name + " (vs brute force)", java.util.Arrays.toString(brute(n, edges)),
                java.util.Arrays.toString(Articulation_points.findArticulationPoints(build(n, edges))));
    }

    public static void main(String[] args) {
        // --- Path 0-1-2-3-4: every interior vertex cuts it ---
        checkCase("path", 5, new int[][]{{0,1},{1,2},{2,3},{3,4}}, new int[]{1, 2, 3});

        // --- Cycle: no single vertex disconnects a ring ---
        checkCase("cycle", 5, new int[][]{{0,1},{1,2},{2,3},{3,4},{4,0}}, new int[]{});

        // --- Star: the hub is the only cut vertex (root-with->=2-children case
        //     when the DFS starts at 0) ---
        checkCase("star", 5, new int[][]{{0,1},{0,2},{0,3},{0,4}}, new int[]{0});

        // --- Two triangles sharing vertex 2: the shared vertex cuts ---
        checkCase("two triangles", 5, new int[][]{{0,1},{1,2},{0,2},{2,3},{3,4},{2,4}}, new int[]{2});

        // --- Two triangles joined by a bridge 2-3: both bridge endpoints cut ---
        checkCase("bridged triangles", 6,
                new int[][]{{0,1},{1,2},{0,2},{2,3},{3,4},{4,5},{3,5}}, new int[]{2, 3});

        // --- Disconnected: two paths, each with its own interior cut vertex ---
        checkCase("disconnected", 6, new int[][]{{0,1},{1,2},{3,4},{4,5}}, new int[]{1, 4});

        // --- Single vertex / no edges / single edge: never articulation points ---
        checkCase("single vertex", 1, new int[][]{}, new int[]{});
        checkCase("edgeless", 4, new int[][]{}, new int[]{});
        checkCase("single edge", 2, new int[][]{{0,1}}, new int[]{});

        // --- Root special case the wrong-root-rule misses: DFS root inside a
        //     cycle has 1 DFS child but degree 2 -> NOT an articulation point ---
        checkCase("triangle (root not AP)", 3, new int[][]{{0,1},{1,2},{0,2}}, new int[]{});

        // --- Randomized cross-check against the brute force ---
        java.util.Random rng = new java.util.Random(7);
        for (int t = 0; t < 5; t++) {
            int n = 8;
            java.util.Set<Long> used = new java.util.HashSet<>();
            java.util.List<int[]> edges = new java.util.ArrayList<>();
            int m = 6 + rng.nextInt(6);
            while (edges.size() < m) {
                int u = rng.nextInt(n), v = rng.nextInt(n);
                if (u == v) continue;
                long key = (long) Math.min(u, v) * 100 + Math.max(u, v);
                if (used.add(key)) edges.add(new int[]{u, v});
            }
            int[][] es = edges.toArray(new int[0][]);
            checkEquals("random graph " + t + " vs brute force",
                    java.util.Arrays.toString(brute(n, es)),
                    java.util.Arrays.toString(Articulation_points.findArticulationPoints(build(n, es))));
        }

        // --- Validation ---
        checkThrows("null graph", NullPointerException.class,
                () -> Articulation_points.findArticulationPoints(null));
        checkThrows("directed graph rejected", IllegalArgumentException.class,
                () -> Articulation_points.findArticulationPoints(new Graph_adjacency_list(3, true)));

        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}
