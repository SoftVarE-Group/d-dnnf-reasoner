import de.softvare.ddnnife.Ddnnf
import de.softvare.ddnnife.SamplingResult
import java.math.BigInteger
import kotlin.test.assertEquals
import kotlin.test.Test
import kotlin.test.fail

internal class Ddnnf {
    private val ddnnf = Ddnnf.fromFile("../../example_input/busybox-1.18.0_c2d.nnf", null)
    private val features = 854

    @Test
    fun sat() {
        val ddnnf = ddnnf.asMut()
        assert(ddnnf.isSat(emptyList()))
        assert(ddnnf.isSat(listOf(-2)))
        assert(!ddnnf.isSat(listOf(1)))
    }

    @Test
    fun count() {
        // Only the root count.
        val count = ddnnf.rc()
        val expected = BigInteger("2061138519356781760670618805653750167349287991336595876373542198990734653489713239449032049664199494301454199336000050382457451123894821886472278234849758979132037884598159833615564800000000000000000000")
        assertEquals(count, expected)

        // The count for a given assumption.
        val ddnnf = ddnnf.asMut()
        assertEquals(ddnnf.count(emptyList()), count)
        assertEquals(ddnnf.count(listOf(1)), BigInteger("0"))
    }

    @Test
    fun coreAndDead() {
        val both = ddnnf.getCore()
        assertEquals(41, both.size)

        val ddnnf = ddnnf.asMut()
        val core = ddnnf.core(emptyList())
        assertEquals(23, core.size)

        val dead = ddnnf.dead(emptyList())
        assertEquals(18, dead.size)
    }

    @Test
    fun enumerate() {
        val configs = ddnnf.asMut().enumerate(emptyList(), 1u)
        assertEquals(1, configs.size)
        assertEquals(features, configs[0].size)
    }

    @Test
    fun random() {
        val configs = ddnnf.asMut().random(emptyList(), 2u, 42u)
        assertEquals(2, configs.size)
        assertEquals(features, configs[0].size)
    }

    @Test
    fun atomicSets() {
        val atomicSets = ddnnf.asMut().atomicSets(null, listOf(1), true)
        assertEquals(features, atomicSets[0].size)
    }

    @Test
    fun tWise() {
        when(val sample = ddnnf.sampleTWise(1u)) {
            is SamplingResult.ResultWithSample -> {
                assertEquals(features, sample.v1.vars.size)
            }
            else -> {
                fail("T-wise sample is invalid.")
            }
        }
    }
}
