package Problems;
import java.util.Objects;

// Job sequencing with deadlines (greedy).
// n jobs, each taking exactly 1 unit of time, with a deadline d[i] >= 1 and a
// profit p[i] >= 0. The machine runs one job per time slot 1, 2, 3, ...; a job
// earns its profit only if it completes by its deadline (i.e. it is scheduled in
// some slot <= d[i]). Choose which jobs to run, and when, to maximize total profit.
//
// Greedy that is provably optimal: consider jobs in decreasing profit order and
// place each into the LATEST still-free slot <= its deadline (skip the job if no
// such slot is free). Latest-possible placement keeps earlier slots open for
// tighter deadlines. (Exchange argument: any optimal schedule can be transformed
// into the greedy one without losing profit.)
public class Job_sequencing_deadlines {

    // Return the maximum total profit achievable.
    // deadlines[i] and profits[i] describe job i; the arrays must be the same length.
    // Throw NullPointerException if deadlines or profits is null.
    // Throw IllegalArgumentException if the lengths differ, any deadline < 1,
    //   or any profit < 0.
    // Zero jobs -> profit 0.
    public static long maxProfit(int[] deadlines, int[] profits) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // Return the number of jobs the optimal greedy schedule actually runs.
    // Same validation rules as maxProfit.
    // (Note: the job COUNT of a maximum-profit schedule is not unique in general —
    // but the greedy above yields a canonical schedule; return that schedule's size.
    // Tests only assert counts on inputs where the count is provably forced.)
    public static int scheduledCount(int[] deadlines, int[] profits) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // --- suggested private helper (shape only; use, rename, or replace freely) ---

    // Sort job indices by profit descending (ties: lower index first, for
    // determinism), e.g. into an Integer[] sorted with a comparator.
    private static Integer[] byProfitDesc(int[] profits) {
        throw new UnsupportedOperationException("not implemented yet");
    }
}

class Job_sequencing_deadlines_Main {
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

    public static void main(String[] args) {
        // Expected profits below were verified against an independent exhaustive
        // subset search (a set of jobs is schedulable iff, after sorting its
        // deadlines ascending, the i-th deadline is >= i; maximize profit over
        // all schedulable subsets).

        // --- Classic example (GfG / Abdul Bari): d{2,1,2,1,3} p{100,19,27,25,15} -> 142
        //     (jobs 100 + 27 + 15) ---
        checkEquals("classic -> 142", 142L,
                Job_sequencing_deadlines.maxProfit(new int[]{2, 1, 2, 1, 3}, new int[]{100, 19, 27, 25, 15}));

        // --- d{2,2,1,3,3} p{20,15,10,5,1} -> 40 (20 + 15 + 5) ---
        checkEquals("second example -> 40", 40L,
                Job_sequencing_deadlines.maxProfit(new int[]{2, 2, 1, 3, 3}, new int[]{20, 15, 10, 5, 1}));

        // --- Deadline-1 pileup: d{1,1,1} p{5,9,3} -> only one job fits -> 9 ---
        checkEquals("all deadline 1 -> best single", 9L,
                Job_sequencing_deadlines.maxProfit(new int[]{1, 1, 1}, new int[]{5, 9, 3}));
        checkEquals("all deadline 1 -> one scheduled", 1,
                Job_sequencing_deadlines.scheduledCount(new int[]{1, 1, 1}, new int[]{5, 9, 3}));

        // --- Latest-slot placement matters: d{4,1,1,1} p{50,10,40,30} -> 90 (50 + 40) ---
        // Greedy places 50 at slot 4 (not slot 1), leaving slot 1 free for 40.
        checkEquals("latest-slot placement -> 90", 90L,
                Job_sequencing_deadlines.maxProfit(new int[]{4, 1, 1, 1}, new int[]{50, 10, 40, 30}));
        checkEquals("latest-slot placement -> two scheduled", 2,
                Job_sequencing_deadlines.scheduledCount(new int[]{4, 1, 1, 1}, new int[]{50, 10, 40, 30}));

        // --- Everything fits: d{1,2,3} p{10,20,30} -> 60, all three run ---
        checkEquals("all fit -> sum", 60L,
                Job_sequencing_deadlines.maxProfit(new int[]{1, 2, 3}, new int[]{10, 20, 30}));
        checkEquals("all fit -> count 3", 3,
                Job_sequencing_deadlines.scheduledCount(new int[]{1, 2, 3}, new int[]{10, 20, 30}));

        // --- Single job / zero jobs ---
        checkEquals("single job", 7L, Job_sequencing_deadlines.maxProfit(new int[]{3}, new int[]{7}));
        checkEquals("no jobs", 0L, Job_sequencing_deadlines.maxProfit(new int[]{}, new int[]{}));
        checkEquals("no jobs count", 0, Job_sequencing_deadlines.scheduledCount(new int[]{}, new int[]{}));

        // --- Zero-profit jobs are legal (may or may not be scheduled; profit unaffected) ---
        checkEquals("zero profits", 0L,
                Job_sequencing_deadlines.maxProfit(new int[]{1, 2}, new int[]{0, 0}));

        // --- Deadlines beyond n don't break slot bookkeeping ---
        checkEquals("huge deadline", 11L,
                Job_sequencing_deadlines.maxProfit(new int[]{1000000, 1}, new int[]{5, 6}));

        // --- Validation ---
        checkThrows("null deadlines", NullPointerException.class,
                () -> Job_sequencing_deadlines.maxProfit(null, new int[]{1}));
        checkThrows("null profits", NullPointerException.class,
                () -> Job_sequencing_deadlines.maxProfit(new int[]{1}, null));
        checkThrows("length mismatch", IllegalArgumentException.class,
                () -> Job_sequencing_deadlines.maxProfit(new int[]{1, 2}, new int[]{5}));
        checkThrows("deadline < 1", IllegalArgumentException.class,
                () -> Job_sequencing_deadlines.maxProfit(new int[]{0}, new int[]{5}));
        checkThrows("negative profit", IllegalArgumentException.class,
                () -> Job_sequencing_deadlines.maxProfit(new int[]{1}, new int[]{-5}));

        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}
