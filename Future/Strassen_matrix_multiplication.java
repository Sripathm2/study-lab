package Algorithms;
import java.util.Objects;
import java.util.Random;

// Strassen matrix multiplication (divide and conquer).
// Multiply two n x n matrices using 7 recursive sub-multiplications instead of the
// naive 8: split each matrix into four n/2 x n/2 quadrants, form Strassen's seven
// products M1..M7 from sums/differences of quadrants, and combine them into the
// four quadrants of the result. T(n) = 7 T(n/2) + O(n^2) -> O(n^log2(7)) ~ O(n^2.81).
//
// Matrices that are not a power-of-two size are padded with zeros up to the next
// power of two, multiplied, then cropped back — padding never changes the product.
public class Strassen_matrix_multiplication {

    // Return the matrix product a * b (standard row-by-column semantics).
    // Both inputs must be square, non-empty, and the same size n x n (any n >= 1,
    //   not just powers of two — pad internally).
    // The inputs must not be modified. The returned matrix is freshly allocated.
    // Throw NullPointerException if a or b (or any of their rows) is null.
    // Throw IllegalArgumentException if either matrix is empty, non-square
    //   (any row length != row count), or the two sizes differ.
    public static long[][] multiply(long[][] a, long[][] b) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // --- suggested private helpers (shape only; use, rename, or replace freely) ---

    // Return a copy of m padded with zero rows/columns up to size p x p (p >= n).
    private static long[][] pad(long[][] m, int p) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // Entrywise sum / difference of two same-size square matrices.
    private static long[][] add(long[][] x, long[][] y) {
        throw new UnsupportedOperationException("not implemented yet");
    }
    private static long[][] sub(long[][] x, long[][] y) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // Recursive core: multiply two p x p matrices where p is a power of two.
    // Base case: fall back to direct multiplication at small sizes (p <= some
    // threshold, e.g. 1 or a small cutoff — Strassen's constant factors lose
    // below the cutoff anyway).
    private static long[][] strassen(long[][] x, long[][] y) {
        throw new UnsupportedOperationException("not implemented yet");
    }
}

class Strassen_matrix_multiplication_Main {
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

    private static void checkTrue(String name, boolean condition) {
        if (condition) {
            passed++;
            System.out.println("PASS: " + name);
        } else {
            failed++;
            System.out.println("FAIL: " + name);
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

    // Independent reference: naive O(n^3) row-by-column multiplication.
    private static long[][] naive(long[][] a, long[][] b) {
        int n = a.length;
        long[][] c = new long[n][n];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    private static String grid(long[][] m) {
        StringBuilder sb = new StringBuilder();
        for (long[] row : m) sb.append(java.util.Arrays.toString(row));
        return sb.toString();
    }

    public static void main(String[] args) {
        // --- 1x1 ---
        checkEquals("1x1", grid(new long[][]{{42}}),
                grid(Strassen_matrix_multiplication.multiply(new long[][]{{6}}, new long[][]{{7}})));

        // --- 2x2 known product: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]] ---
        // (verified by hand and against numpy)
        checkEquals("2x2 known", grid(new long[][]{{19, 22}, {43, 50}}),
                grid(Strassen_matrix_multiplication.multiply(
                        new long[][]{{1, 2}, {3, 4}}, new long[][]{{5, 6}, {7, 8}})));

        // --- 3x3 (non-power-of-two -> exercises padding), negatives included ---
        // [[1,-2,3],[4,5,-6],[-7,8,9]] * [[2,0,1],[1,3,-1],[0,-2,4]]
        //   = [[0,-12,15],[13,27,-25],[-6,6,21]]   (verified against numpy)
        checkEquals("3x3 with negatives (padding path)",
                grid(new long[][]{{0, -12, 15}, {13, 27, -25}, {-6, 6, 21}}),
                grid(Strassen_matrix_multiplication.multiply(
                        new long[][]{{1, -2, 3}, {4, 5, -6}, {-7, 8, 9}},
                        new long[][]{{2, 0, 1}, {1, 3, -1}, {0, -2, 4}})));

        // --- Identity and zero ---
        long[][] a = {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}};
        long[][] id = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        checkEquals("A * I = A", grid(a), grid(Strassen_matrix_multiplication.multiply(a, id)));
        checkEquals("I * A = A", grid(a), grid(Strassen_matrix_multiplication.multiply(id, a)));
        long[][] zero = new long[3][3];
        checkEquals("A * 0 = 0", grid(zero), grid(Strassen_matrix_multiplication.multiply(a, zero)));

        // --- Inputs unchanged ---
        long[][] aCopy = {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}};
        Strassen_matrix_multiplication.multiply(a, id);
        checkEquals("input not mutated", grid(aCopy), grid(a));

        // --- Randomized cross-check against the independent naive multiplier ---
        Random rng = new Random(42);
        for (int n : new int[]{4, 5, 7, 8}) {
            long[][] x = new long[n][n];
            long[][] y = new long[n][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    x[i][j] = rng.nextInt(21) - 10;
                    y[i][j] = rng.nextInt(21) - 10;
                }
            checkEquals("random " + n + "x" + n + " vs naive", grid(naive(x, y)),
                    grid(Strassen_matrix_multiplication.multiply(x, y)));
        }

        // --- Validation ---
        checkThrows("null a", NullPointerException.class,
                () -> Strassen_matrix_multiplication.multiply(null, id));
        checkThrows("null b", NullPointerException.class,
                () -> Strassen_matrix_multiplication.multiply(id, null));
        checkThrows("empty", IllegalArgumentException.class,
                () -> Strassen_matrix_multiplication.multiply(new long[0][0], new long[0][0]));
        checkThrows("non-square", IllegalArgumentException.class,
                () -> Strassen_matrix_multiplication.multiply(new long[][]{{1, 2}}, new long[][]{{1, 2}}));
        checkThrows("size mismatch", IllegalArgumentException.class,
                () -> Strassen_matrix_multiplication.multiply(id, new long[][]{{1, 2}, {3, 4}}));

        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}
