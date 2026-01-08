package de.softvare.ddnnife;

import java.math.BigInteger;
import java.util.List;
import org.junit.jupiter.api.Test;

import static de.softvare.ddnnife.JavaUtils.*;
import static java.util.Collections.emptyList;
import static org.junit.jupiter.api.Assertions.*;

class JavaTest {
    private final Ddnnf ddnnf = ddnnfFromFile("../../example_input/busybox-1.18.0_c2d.nnf", null);
    private final Integer features = 854;

    @Test
    void sat() {
        DdnnfMut ddnnfMut = ddnnf.asMut();
        assertTrue(ddnnfMut.isSat(emptyList()));
        assertTrue(ddnnfMut.isSat(List.of(-2)));
        assertFalse(ddnnfMut.isSat(List.of(1)));
    }

    @Test
    void count() {
        // Only the root count.
        BigInteger count = ddnnf.rc();
        BigInteger expected =
                new BigInteger("2061138519356781760670618805653750167349287991336595876373542198990734653489713239449032049664199494301454199336000050382457451123894821886472278234849758979132037884598159833615564800000000000000000000");
        assertEquals(count, expected);

        // The count for a given assumption.
        DdnnfMut ddnnfMut = ddnnf.asMut();
        assertEquals(ddnnfMut.count(emptyList()), count);
        assertEquals(new BigInteger("0"), ddnnfMut.count(List.of(1)));
    }

    @Test
    void coreAndDead() {
        List<Integer> both = ddnnf.getCore();
        assertEquals(41, both.size());

        DdnnfMut ddnnfMut = ddnnf.asMut();
        List<Integer> core = ddnnfMut.core(emptyList());
        assertEquals(23, core.size());

        List<Integer> dead = ddnnfMut.dead(emptyList());
        assertEquals(18, dead.size());
    }

    @Test
    void enumerateTest() {
        List<List<Integer>> configs = enumerate(ddnnf.asMut(), emptyList(), 1);
        assertEquals(1, configs.size());
        assertEquals(features, configs.getFirst().size());
    }

    @Test
    void randomTest() {
        List<List<Integer>> configs = random(ddnnf.asMut(), emptyList(), 2, 42);
        assertEquals(2, configs.size());
        assertEquals(features, configs.getFirst().size());
    }

    @Test
    void atomicSetsTest() {
        List<List<Short>> atomicSets = atomicSets(ddnnf.asMut(), null, List.of(1), true);
        assertEquals(features, atomicSets.getFirst().size());
    }

    @Test
    void tWise() {
        SamplingResult result = sampleTWise(ddnnf, 1);
        assertTrue(isSample(result));

        Sample sample = getSample(result);
        assertEquals(features, sample.getVars().size());
    }

    @Test
    void cnf() {
        Cnf cnf = ddnnf.toCnf();
        String serialized = cnf.serialize();
        // TODO: `numVariables` is not reachable
        //assertEquals(2830u, cnf.numVariables());
        assertEquals(7481, cnf.clauses().size());
        assertEquals(110642, serialized.length());
    }

    @Test
    void toUIntTest() {
        assertEquals(1, toUInt(1));
    }

    @Test
    void trivial() {
      Ddnnf trivialDdnnf = ddnnfFromFile("../../ddnnife/tests/data/stub_true.nnf", null);
      assert(trivialDdnnf.isTrivial());
    }

    @Test
    void nonTrivial() {
      assert(!ddnnf.isTrivial());
    }
}
