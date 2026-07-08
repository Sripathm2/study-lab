package Problems;
import java.util.Objects;

// 0/1 Knapsack (dynamic programming).
// Given n items, each with a weight and a value, and a knapsack capacity, pick a
// subset of items whose total weight is at most the capacity and whose total
// value is as large as possible. Each item may be taken at most once (0/1).
//
// dp[i][w] = best value using the first i items within capacity w:
//   dp[0][w] = 0
//   dp[i][w] = dp[i-1][w]                                    if weights[i-1] > w
//            = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])  otherwise
public class Knapsack_01 {

    // Return the maximum total value achievable without exceeding capacity.
    // weights[i] and values[i] describe item i; the arrays must be the same length.
    // Throw NullPointerException if weights or values is null.
    // Throw IllegalArgumentException if the lengths differ, capacity < 0,
    //   or any weight or value is negative.
    public static int maxValue(int[] weights, int[] values, int capacity) {
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
            return 0;
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
        return dp[dp.length-1][dp[0].length-1];
    }



}

class Knapsack_01_Main {
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

    // Independent brute force: try every subset (include/exclude each item).
    private static int brute(int[] w, int[] v, int cap) {
        return rec(w, v, cap, 0);
    }
    private static int rec(int[] w, int[] v, int cap, int i) {
        if (i == w.length) return 0;
        int skip = rec(w, v, cap, i + 1);
        int take = (w[i] <= cap) ? v[i] + rec(w, v, cap - w[i], i + 1) : Integer.MIN_VALUE;
        return Math.max(skip, take);
    }

    public static void main(String[] args) {
        // --- Classic example: best is items (3,4)+(4,5) = weight 7, value 9 ---
        checkEquals("classic w{1,3,4,5} v{1,4,5,7} cap7", 9,
                Knapsack_01.maxValue(new int[]{1,3,4,5}, new int[]{1,4,5,7}, 7));

        // --- Another: (2,3)+(3,4) = weight 5, value 7 ---
        checkEquals("w{2,3,4,5} v{3,4,5,6} cap5", 7,
                Knapsack_01.maxValue(new int[]{2,3,4,5}, new int[]{3,4,5,6}, 5));

        // --- Nothing fits ---
        checkEquals("all too heavy", 0,
                Knapsack_01.maxValue(new int[]{5,6,7}, new int[]{10,20,30}, 4));

        // --- Everything fits -> sum of values ---
        checkEquals("all fit -> sum", 60,
                Knapsack_01.maxValue(new int[]{1,2,3}, new int[]{10,20,30}, 100));

        // --- Single item ---
        checkEquals("single fits", 5, Knapsack_01.maxValue(new int[]{3}, new int[]{5}, 3));
        checkEquals("single too heavy", 0, Knapsack_01.maxValue(new int[]{3}, new int[]{5}, 2));

        // --- Empty item set ---
        checkEquals("no items", 0, Knapsack_01.maxValue(new int[]{}, new int[]{}, 10));

        // --- Capacity 0 ---
        checkEquals("cap 0", 0, Knapsack_01.maxValue(new int[]{1,2}, new int[]{5,6}, 0));

        // --- Each item usable at most once (0/1, not unbounded) ---
        // one item (w2,v3), cap 6: unbounded would give 9, 0/1 gives 3.
        checkEquals("0/1 not unbounded", 3, Knapsack_01.maxValue(new int[]{2}, new int[]{3}, 6));

        // --- Brute-force cross-check on random small instances ---
        java.util.Random rng = new java.util.Random(81);
        boolean ok = true;
        for (int t = 0; t < 500 && ok; t++) {
            int n = rng.nextInt(10);
            int[] w = new int[n], v = new int[n];
            for (int i = 0; i < n; i++) { w[i] = rng.nextInt(12); v[i] = rng.nextInt(20); }
            int cap = rng.nextInt(25);
            if (Knapsack_01.maxValue(w, v, cap) != brute(w, v, cap)) ok = false;
        }
        if (ok) { passed++; System.out.println("PASS: brute-force cross-check"); }
        else    { failed++; System.out.println("FAIL: brute-force cross-check"); }

        // --- Validation ---
        checkThrows("null weights", NullPointerException.class,
                () -> Knapsack_01.maxValue(null, new int[]{1}, 5));
        checkThrows("null values", NullPointerException.class,
                () -> Knapsack_01.maxValue(new int[]{1}, null, 5));
        checkThrows("length mismatch", IllegalArgumentException.class,
                () -> Knapsack_01.maxValue(new int[]{1,2}, new int[]{1}, 5));
        checkThrows("negative capacity", IllegalArgumentException.class,
                () -> Knapsack_01.maxValue(new int[]{1}, new int[]{1}, -1));
        checkThrows("negative weight", IllegalArgumentException.class,
                () -> Knapsack_01.maxValue(new int[]{-1}, new int[]{1}, 5));
        checkThrows("negative value", IllegalArgumentException.class,
                () -> Knapsack_01.maxValue(new int[]{1}, new int[]{-1}, 5));

        System.out.println();
        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}