package Algorithms;
import java.util.Map;
import java.util.Objects;

// Huffman encoding (greedy).
// Build an optimal prefix-free binary code for the characters of a text: frequent
// characters get short codes, rare ones get long codes, and no code is a prefix of
// another (so decoding needs no separators). Greedy construction: put every
// character in a min-heap keyed by frequency; repeatedly pop the two lightest
// nodes, join them under a new internal node whose weight is their sum, and push
// it back; the last remaining node is the root. Left edge = '0', right edge = '1';
// a character's code is the root-to-leaf path.
//
// Huffman's greedy is provably optimal: it minimizes sum(freq(c) * len(code(c))),
// and that minimum total is unique even though the codes themselves are not
// (tie-breaks flip siblings / equal-weight merges). Tests therefore assert the
// TOTAL encoded length and code properties, never specific bit strings.
//
// The repo's Data_structures.Priority_queue (comparator-ordered) fits the heap role.
public class Huffman_encoding {

    // Build a Huffman code table for the given text: a map from each distinct
    // character in the text to its binary code string (e.g. 'a' -> "10").
    // Special case: a text with exactly ONE distinct character maps it to "0"
    //   (a zero-length code would make the encoding empty and undecodable).
    // Throw NullPointerException if text is null.
    // Throw IllegalArgumentException if text is empty (no frequencies to build from).
    public static Map<Character, String> buildCodes(String text) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // Encode the text with the given code table: the concatenation of each
    // character's code, as a string of '0'/'1'.
    // Throw NullPointerException if text or codes is null.
    // Throw IllegalArgumentException if the text contains a character absent
    //   from the table.
    public static String encode(String text, Map<Character, String> codes) {
        throw new UnsupportedOperationException("not implemented yet");
    }

    // Decode a '0'/'1' string produced by encode back into the original text,
    // using the same code table. Prefix-freeness makes this unambiguous: walk the
    // bits, and whenever the accumulated bits match some character's code, emit it
    // and reset. (Equivalently: rebuild/keep the tree and walk it bit by bit.)
    // Throw NullPointerException if bits or codes is null.
    // Throw IllegalArgumentException if bits contains a character other than
    //   '0'/'1', or is not a valid encoding under the table (leftover bits that
    //   match no code).
    public static String decode(String bits, Map<Character, String> codes) {
        throw new UnsupportedOperationException("not implemented yet");
    }
}

class Huffman_encoding_Main {
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

    // Prefix-free property: no code is a prefix of a different character's code.
    private static boolean prefixFree(Map<Character, String> codes) {
        for (Map.Entry<Character, String> a : codes.entrySet())
            for (Map.Entry<Character, String> b : codes.entrySet())
                if (!a.getKey().equals(b.getKey()) && b.getValue().startsWith(a.getValue()))
                    return false;
        return true;
    }

    private static void roundTripAndLength(String label, String text, int expectedBits) {
        Map<Character, String> codes = Huffman_encoding.buildCodes(text);
        checkTrue(label + ": prefix-free", prefixFree(codes));
        checkTrue(label + ": covers all chars",
                text.chars().allMatch(c -> codes.containsKey((char) c)));
        String bits = Huffman_encoding.encode(text, codes);
        checkTrue(label + ": bits are 0/1 only", bits.chars().allMatch(c -> c == '0' || c == '1'));
        // Optimal total length is unique across all valid Huffman codes for the text.
        checkEquals(label + ": optimal total bits", expectedBits, bits.length());
        checkEquals(label + ": decode(encode(x)) == x", text, Huffman_encoding.decode(bits, codes));
    }

    public static void main(String[] args) {
        // Expected bit totals verified against an independent heap-merge computation
        // (sum of the two popped weights at every merge = total weighted code length).

        // --- freq a:4 b:3 c:2 d:1 -> merges (1+2)+(3+3)+(4+6) -> 19 bits total ---
        roundTripAndLength("aaaabbbccd", "aaaabbbccd", 19);

        // --- "beep boop beer!" -> 40 bits ---
        roundTripAndLength("beep boop beer!", "beep boop beer!", 40);

        // --- "mississippi" -> 21 bits ---
        roundTripAndLength("mississippi", "mississippi", 21);

        // --- Two distinct chars: each code must be 1 bit -> length == text length ---
        roundTripAndLength("ababab", "ababab", 6);

        // --- Single distinct character: code "0", 1 bit per char ---
        Map<Character, String> single = Huffman_encoding.buildCodes("zzzz");
        checkEquals("single distinct char -> code \"0\"", "0", single.get('z'));
        checkEquals("single distinct char -> table size 1", 1, single.size());
        checkEquals("single distinct char round trip", "zzzz",
                Huffman_encoding.decode(Huffman_encoding.encode("zzzz", single), single));

        // --- Encoding never beats the entropy floor / never exceeds fixed-width ---
        // For k distinct chars, fixed-width needs ceil(log2 k) bits per char; Huffman
        // must be <= that in total (here: 4 distinct -> 2 bits * 10 chars = 20 >= 19).
        checkTrue("beats fixed width on skewed input",
                Huffman_encoding.encode("aaaabbbccd",
                        Huffman_encoding.buildCodes("aaaabbbccd")).length() <= 20);

        // --- Validation ---
        checkThrows("null text (build)", NullPointerException.class,
                () -> Huffman_encoding.buildCodes(null));
        checkThrows("empty text (build)", IllegalArgumentException.class,
                () -> Huffman_encoding.buildCodes(""));
        checkThrows("null text (encode)", NullPointerException.class,
                () -> Huffman_encoding.encode(null, java.util.Map.of('a', "0")));
        checkThrows("char missing from table", IllegalArgumentException.class,
                () -> Huffman_encoding.encode("ab", java.util.Map.of('a', "0")));
        checkThrows("null bits (decode)", NullPointerException.class,
                () -> Huffman_encoding.decode(null, java.util.Map.of('a', "0")));
        checkThrows("non-bit character", IllegalArgumentException.class,
                () -> Huffman_encoding.decode("012", java.util.Map.of('a', "0")));
        // "010" -> "01" = b, leftover "0" matches no code -> invalid encoding
        checkThrows("dangling bits", IllegalArgumentException.class,
                () -> Huffman_encoding.decode("010",
                        java.util.Map.of('a', "00", 'b', "01", 'c', "1")));

        System.out.println("=== " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }
}
