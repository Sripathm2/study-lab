package Algorithms;
import Data_structures.Graph;
import Data_structures.Queue;

import java.util.Objects;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

// Breadth-first search on any Graph. BFS explores level by level (a queue, FIFO),
// so in an UNWEIGHTED graph the first time it reaches a vertex is via a shortest
// path. Recording where each vertex was first reached (a parent array) lets us
// reconstruct that path. Neighbors are visited in ascending order, so results are
// deterministic.
public class Breadth_first_search {

    private static int[] parent;

    // Vertices in BFS visit order starting from `start` (only reachable vertices).
    // Throw NullPointerException if g is null.
    // Throw IndexOutOfBoundsException if start is not in [0, vertexCount).
    public static int[] order(Graph g, int start) {
        if(g == null){
            throw new NullPointerException();
        } 
        if(start <0 || start >=g.vertexCount()){
            throw new IndexOutOfBoundsException();
        }
        
        parent = new int[g.vertexCount()];
        for(int i =0;i<parent.length;i++){
            parent[i] = -1;
        }
        parent[start] = -2;
        ArrayList<Integer> order = new ArrayList<Integer>();
        boolean [] visit = new boolean[g.vertexCount()];
        Queue<Integer> queue = new Queue<Integer>();
        queue.enqueue(start);
        while(!queue.isEmpty()){
            int node = (int)queue.dequeue();
            if(!visit[node]){
                order.add(node);
                visit[node] = true;
                int [] neighbors = g.neighbors(node);
                for(int i = 0; i < neighbors.length; i++){
                    if(parent[neighbors[i]] == -1)
                        parent[neighbors[i]] = node;
                    queue.enqueue(neighbors[i]);
                }
            }
        }
        return order.stream().mapToInt(i -> i).toArray();
    }

    // dist[v] = number of edges on a shortest path from start to v, or -1 if v is
    // unreachable. dist[start] == 0. Length == vertexCount().
    // Throw NullPointerException if g is null.
    // Throw IndexOutOfBoundsException if start is not in [0, vertexCount).
    public static int[] distances(Graph g, int start) {
        int[] returnv = new int[g.vertexCount()];
        order(g,start);
        for(int i=0;i<returnv.length; i++){
            returnv[i] = shortestPath_compute_once(g, start, i).length;
            returnv[i] -= 1;
        }
        return returnv;
    }

    public static int[] shortestPath_compute_once(Graph g, int start, int target) {
        if(target <0 || target >=g.vertexCount()){
            throw new IndexOutOfBoundsException();
        }
        if(start == target){
            return new int[]{start};
        }
        ArrayList<Integer> path = new ArrayList<Integer>();
        int index = target;
        if(parent[index] == -1){
            return new int[0];
        }
        while(true){
            path.add(index);
            index = parent[index];
            if(index == -2){
                break;
            }
        }
        Collections.reverse(path);
        return path.stream().mapToInt(i -> i).toArray();
    }

    // The vertices along a shortest path from start to target, inclusive of both.
    // Return {start} if start == target; return an empty array if target is
    // unreachable from start.
    // Throw NullPointerException if g is null.
    // Throw IndexOutOfBoundsException if start or target is not in [0, vertexCount).
    public static int[] shortestPath(Graph g, int start, int target) {
        if(target <0 || target >=g.vertexCount()){
            throw new IndexOutOfBoundsException();
        }
        if(start == target){
            return new int[]{start};
        }
        order(g,start);
        ArrayList<Integer> path = new ArrayList<Integer>();
        int index = target;
        if(parent[index] == -1){
            return new int[0];
        }
        while(true){
            path.add(index);
            index = parent[index];
            if(index == -2){
                break;
            }
        }
        Collections.reverse(path);
        return path.stream().mapToInt(i -> i).toArray();
    }
}

class Breadth_first_search_Main {
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

    private static void checkTrue(String name, boolean cond) {
        if (cond) { passed++; System.out.println("PASS: " + name); }
        else      { failed++; System.out.println("FAIL: " + name); }
    }

    private static void checkThrows(String name, Class<? extends Throwable> expected, Runnable r) {
        try {
            r.run();
            failed++;
            System.out.println("FAIL: " + name + " — expected " + expected.getSimpleName() + ", none thrown");
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

    // Self-contained Graph so BFS is testable without the representation classes.
    private static final class TestGraph implements Graph {
        private final int n;
        private final boolean directed;
        private final List<List<Integer>> adj;
        TestGraph(int n, boolean directed) {
            this.n = n; this.directed = directed;
            adj = new ArrayList<>();
            for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        }
        void addEdge(int u, int v) { adj.get(u).add(v); if (!directed) adj.get(v).add(u); }
        public int vertexCount() { return n; }
        public boolean isDirected() { return directed; }
        public boolean hasEdge(int u, int v) { return adj.get(u).contains(v); }
        public int weight(int u, int v) {
            if (!hasEdge(u, v)) throw new java.util.NoSuchElementException();
            return 1;
        }
        public int[] neighbors(int v) {
            List<Integer> ns = new ArrayList<>(adj.get(v));
            Collections.sort(ns);
            int[] out = new int[ns.size()];
            for (int i = 0; i < out.length; i++) out[i] = ns.get(i);
            return out;
        }
    }

    // Validate that a returned path is a genuine walk of the claimed shortest length.
    private static boolean validPath(Graph g, int start, int target, int[] path, int[] dist) {
        if (dist[target] == -1) return path.length == 0;          // unreachable => empty
        if (path.length == 0) return false;
        if (path[0] != start || path[path.length - 1] != target) return false;
        for (int i = 0; i + 1 < path.length; i++)
            if (!g.hasEdge(path[i], path[i + 1])) return false;   // each step is a real edge
        return path.length - 1 == dist[target];                   // edge count == distance
    }

    public static void main(String[] args) {
        // Undirected:
        //   0-1, 0-2, 1-3, 2-3, 3-4, 4-5   ; 6 isolated
        TestGraph u = new TestGraph(7, false);
        u.addEdge(0,1); u.addEdge(0,2); u.addEdge(1,3); u.addEdge(2,3); u.addEdge(3,4); u.addEdge(4,5);

        checkEquals("order from 0", Arrays.toString(new int[]{0,1,2,3,4,5}),
                Arrays.toString(Breadth_first_search.order(u, 0)));

        checkEquals("distances from 0",
                Arrays.toString(new int[]{0,1,1,2,3,4,-1}),
                Arrays.toString(Breadth_first_search.distances(u, 0)));

        checkEquals("path 0->5", Arrays.toString(new int[]{0,1,3,4,5}),
                Arrays.toString(Breadth_first_search.shortestPath(u, 0, 5)));
        checkEquals("path 0->3", Arrays.toString(new int[]{0,1,3}),
                Arrays.toString(Breadth_first_search.shortestPath(u, 0, 3)));
        checkEquals("path start==target", Arrays.toString(new int[]{0}),
                Arrays.toString(Breadth_first_search.shortestPath(u, 0, 0)));
        checkEquals("path unreachable", Arrays.toString(new int[]{}),
                Arrays.toString(Breadth_first_search.shortestPath(u, 0, 6)));

        // Directed: 0->1, 1->2, 2->3, 0->3  (direct 0->3 is shorter than 0->1->2->3)
        TestGraph d = new TestGraph(4, true);
        d.addEdge(0,1); d.addEdge(1,2); d.addEdge(2,3); d.addEdge(0,3);
        checkEquals("dir distances from 0",
                Arrays.toString(new int[]{0,1,2,1}),
                Arrays.toString(Breadth_first_search.distances(d, 0)));
        checkEquals("dir path 0->3 (uses direct edge)", Arrays.toString(new int[]{0,3}),
                Arrays.toString(Breadth_first_search.shortestPath(d, 0, 3)));

        // Path validity over random graphs (independent of the exact route chosen)
        java.util.Random rng = new java.util.Random(111);
        boolean ok = true;
        for (int t = 0; t < 300 && ok; t++) {
            int n = 1 + rng.nextInt(8);
            boolean dir = rng.nextBoolean();
            TestGraph g = new TestGraph(n, dir);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (i != j && rng.nextInt(3) == 0) g.addEdge(i, j);
            int s = rng.nextInt(n);
            int[] dist = Breadth_first_search.distances(g, s);
            if (dist.length != n) { ok = false; break; }
            for (int tgt = 0; tgt < n; tgt++)
                if (!validPath(g, s, tgt, Breadth_first_search.shortestPath(g, s, tgt), dist)) { ok = false; break; }
        }
        checkTrue("random paths valid & shortest", ok);

        // --- Errors ---
        checkThrows("null graph", NullPointerException.class,
                () -> Breadth_first_search.order(null, 0));
        checkThrows("start oob", IndexOutOfBoundsException.class,
                () -> Breadth_first_search.distances(u, 7));
        checkThrows("target oob", IndexOutOfBoundsException.class,
                () -> Breadth_first_search.shortestPath(u, 0, 7));

        System.out.println();
        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}