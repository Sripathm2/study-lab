package Problems;
import java.util.Objects;

// N-Queens (backtracking).
// Place n queens on an n x n board so that no two attack each other: no shared
// row, column, or diagonal. Two classic questions: how many distinct placements
// exist, and produce one.
//
// Backtracking, one row at a time: since each row holds exactly one queen, the
// state is "which column in each row so far." At row r, try each column c that is
// not attacked, place, recurse to r+1, then UNDO and try the next c. Reaching row
// n = one complete solution.
//
// O(1) attack checks with three occupancy sets (boolean arrays):
//   cols[c]            — column already used
//   diag1[r + c]       — "/" anti-diagonals: r + c constant along each (2n-1 of them)
//   diag2[r - c + n-1] — "\" main diagonals: r - c constant, shifted non-negative
// Place = set all three + record; undo = clear all three. The undo is the essence
// of backtracking — state must be IDENTICAL before and after trying a column.
public class N_queens {

    // Return the number of distinct solutions for an n x n board.
    // Distinct = different column assignments; symmetric/rotated boards count
    // separately (the standard convention: n=8 -> 92).
    // Throw IllegalArgumentException if n < 1.
    public static long countSolutions(int n) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // Return one valid placement as an array q of length n, where q[r] is the
    // column of the queen in row r — or an EMPTY array if no solution exists
    // (n = 2 and n = 3 are the only unsolvable sizes >= 1).
    // Any valid placement is acceptable; tests validate the board, not one answer.
    // Throw IllegalArgumentException if n < 1.
    public static int[] oneSolution(int n) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // --- suggested private helper (shape only; use, rename, or replace freely) ---

    // Backtrack from row r; return the count below this state (for counting), or
    // a boolean "found" (for oneSolution) — or unify both with a small flag.
    private static long backtrack(int r, int n, boolean[] cols, boolean[] diag1,
                                  boolean[] diag2, int[] placement, boolean stopAtFirst) {
        throw new UnsupportedOperationException("not implemented yet");
    }
}

class N_queens_Main {
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

    // Independent validity check for a returned placement.
    private static boolean validBoard(int[] q, int n) {
        if (q.length != n) return false;
        for (int r = 0; r < n; r++) {
            if (q[r] < 0 || q[r] >= n) return false;
            for (int r2 = r + 1; r2 < n; r2++) {
                if (q[r] == q[r2]) return false;                       // same column
                if (Math.abs(q[r] - q[r2]) == Math.abs(r - r2)) return false; // diagonal
            }
        }
        return true;
    }

    public static void main(String[] args) {
        // --- Solution counts: OEIS A000170 (independently established sequence) ---
        long[] expected = {1, 0, 0, 2, 10, 4, 40, 92, 352};
        for (int n = 1; n <= 9; n++) {
            checkEquals("count n=" + n, expected[n - 1], N_queens.countSolutions(n));
        }

        // --- oneSolution: validate the board wherever a solution exists ---
        for (int n : new int[]{1, 4, 5, 6, 7, 8}) {
            int[] q = N_queens.oneSolution(n);
            checkTrue("oneSolution n=" + n + " is a valid board", validBoard(q, n));
        }

        // --- The unsolvable sizes return empty ---
        checkEquals("n=2 unsolvable", 0, N_queens.oneSolution(2).length);
        checkEquals("n=3 unsolvable", 0, N_queens.oneSolution(3).length);

        // --- n=1 trivial board ---
        checkEquals("n=1 single queen at col 0", "[0]",
                java.util.Arrays.toString(N_queens.oneSolution(1)));

        // --- Count/oneSolution consistency: zero count <=> empty board ---
        for (int n = 1; n <= 6; n++) {
            checkEquals("consistency n=" + n,
                    N_queens.countSolutions(n) == 0,
                    N_queens.oneSolution(n).length == 0);
        }

        // --- Validation ---
        checkThrows("n=0 rejected", IllegalArgumentException.class, () -> N_queens.countSolutions(0));
        checkThrows("negative n rejected", IllegalArgumentException.class, () -> N_queens.oneSolution(-3));

        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}
