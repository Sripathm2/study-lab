package Problems;
import java.util.Objects;
import java.util.ArrayList;
import java.util.Arrays;

// 0/1 Knapsack with reconstruction: return both the maximum achievable value
// and the indices of a chosen optimal item subset.
//
// The value is computed with the usual dp[i][w] table; the chosen items are
// recovered by walking that table backward from (n, W): whenever
// dp[i][w] != dp[i-1][w], item i-1 was taken, so record it and drop its weight.
public class Knapsack_01_items {

    // Result of a knapsack solve: the optimal value and the indices taken (ascending).
    public static final class Result {
        public final int value;
        public final int[] items;   // chosen item indices, ascending
        public Result(int value, int[] items) { this.value = value; this.items = items; }
        @Override public boolean equals(Object o) {
            if (!(o instanceof Result)) return false;
            Result r = (Result) o;
            return value == r.value && Arrays.equals(items, r.items);
        }
        @Override public int hashCode() { return 31 * value + Arrays.hashCode(items); }
        @Override public String toString() { return "value=" + value + ", items=" + Arrays.toString(items); }
    }

    // Return the maximum total value and the indices of an optimal subset (ascending).
    // If several subsets are optimal, returning any one of them is acceptable.
    // Throw NullPointerException if weights or values is null.
    // Throw IllegalArgumentException if the lengths differ, capacity < 0,
    //   or any weight or value is negative.
    public static Result maxValueWithItems(int[] weights, int[] values, int capacity) {
        if(weights == null || values == null){
            throw new NullPointerException();
        } 
        if(weights.length != values.length || capacity < 0){
            throw new IllegalArgumentException();
        }

        for(int i=0;i<weights.length; i++){
            if(weights[i] < 0 || values[i] < 0){
                throw new IllegalArgumentException();
            }
        }

        if(weights.length == 0){
            return new Result(0,new int[0]);
        }

        int [][] dp = new int[weights.length+1][capacity+1];
        
        for(int i = 1; i <= weights.length; i++){
            for(int w = 0; w < dp[0].length; w++){
                if(weights[i-1] > w){
                    dp[i][w] = dp[i-1][w];
                }else{
                    dp[i][w] = Math.max(dp[i-1][w] , dp[i-1][w-weights[i-1]] + values[i-1]);
                }
            }
            
        }
        ArrayList<Integer> items = new ArrayList<Integer>();

        int i = dp.length-1;
        int j = dp[0].length-1;

        while(i > 0){
            if(dp[i][j] != dp[i-1][j]){
                items.add(i-1);
                j = j-weights[i-1];
                i = i-1;
            }else{
                i = i-1;
            }
        }

        int[] items_arr = items.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(items_arr);
        Result result = new Result(dp[dp.length-1][dp[0].length-1], items_arr);
        return result;
    }
}

class Knapsack_01_items_Main {
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

    private static int bruteMax(int[] w, int[] v, int cap, int i) {
        if (i == w.length) return 0;
        int skip = bruteMax(w, v, cap, i + 1);
        int take = (w[i] <= cap) ? v[i] + bruteMax(w, v, cap - w[i], i + 1) : Integer.MIN_VALUE;
        return Math.max(skip, take);
    }

    // A returned Result is "good" iff its items are distinct, in range, fit the
    // capacity, sum to the reported value, and that value equals the true optimum.
    private static boolean good(int[] w, int[] v, int cap, Knapsack_01_items.Result r) {
        if (r == null || r.items == null) return false;
        boolean[] seen = new boolean[w.length];
        int tw = 0, tv = 0;
        for (int idx : r.items) {
            if (idx < 0 || idx >= w.length || seen[idx]) return false;
            seen[idx] = true;
            tw += w[idx]; tv += v[idx];
        }
        int best = bruteMax(w, v, cap, 0);
        boolean ascending = true;
        for (int i = 1; i < r.items.length; i++) if (r.items[i] <= r.items[i - 1]) ascending = false;
        return tw <= cap && tv == r.value && r.value == best && ascending;
    }

    public static void main(String[] args) {
        // --- Unique optimum: value 9 comes only from items {1,2} = (3,4)+(4,5) ---
        int[] w1 = {1,3,4,5}, v1 = {1,4,5,7};
        Knapsack_01_items.Result r1 = Knapsack_01_items.maxValueWithItems(w1, v1, 7);
        checkEquals("classic exact", new Knapsack_01_items.Result(9, new int[]{1,2}), r1);
        checkTrue("classic valid", good(w1, v1, 7, r1));

        // --- All items fit -> take everything ---
        int[] w2 = {1,2,3}, v2 = {10,20,30};
        checkEquals("all fit exact",
                new Knapsack_01_items.Result(60, new int[]{0,1,2}),
                Knapsack_01_items.maxValueWithItems(w2, v2, 100));

        // --- Nothing fits -> empty selection ---
        int[] w3 = {5,6,7}, v3 = {10,20,30};
        checkEquals("nothing fits exact",
                new Knapsack_01_items.Result(0, new int[]{}),
                Knapsack_01_items.maxValueWithItems(w3, v3, 4));

        // --- Single item ---
        checkEquals("single fits",
                new Knapsack_01_items.Result(5, new int[]{0}),
                Knapsack_01_items.maxValueWithItems(new int[]{3}, new int[]{5}, 3));
        checkEquals("single too heavy",
                new Knapsack_01_items.Result(0, new int[]{}),
                Knapsack_01_items.maxValueWithItems(new int[]{3}, new int[]{5}, 2));

        // --- Empty item set ---
        checkEquals("no items",
                new Knapsack_01_items.Result(0, new int[]{}),
                Knapsack_01_items.maxValueWithItems(new int[]{}, new int[]{}, 10));

        // --- General instances: validate the returned selection (ties allowed) ---
        java.util.Random rng = new java.util.Random(91);
        boolean ok = true;
        for (int t = 0; t < 500 && ok; t++) {
            int n = rng.nextInt(10);
            int[] w = new int[n], v = new int[n];
            for (int i = 0; i < n; i++) { w[i] = rng.nextInt(12); v[i] = rng.nextInt(20); }
            int cap = rng.nextInt(25);
            if (!good(w, v, cap, Knapsack_01_items.maxValueWithItems(w, v, cap))) ok = false;
        }
        if (ok) { passed++; System.out.println("PASS: random selections are valid & optimal"); }
        else    { failed++; System.out.println("FAIL: random selections are valid & optimal"); }

        // --- Validation ---
        checkThrows("null weights", NullPointerException.class,
                () -> Knapsack_01_items.maxValueWithItems(null, new int[]{1}, 5));
        checkThrows("null values", NullPointerException.class,
                () -> Knapsack_01_items.maxValueWithItems(new int[]{1}, null, 5));
        checkThrows("length mismatch", IllegalArgumentException.class,
                () -> Knapsack_01_items.maxValueWithItems(new int[]{1,2}, new int[]{1}, 5));
        checkThrows("negative capacity", IllegalArgumentException.class,
                () -> Knapsack_01_items.maxValueWithItems(new int[]{1}, new int[]{1}, -1));
        checkThrows("negative weight", IllegalArgumentException.class,
                () -> Knapsack_01_items.maxValueWithItems(new int[]{-1}, new int[]{1}, 5));

        System.out.println();
        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}