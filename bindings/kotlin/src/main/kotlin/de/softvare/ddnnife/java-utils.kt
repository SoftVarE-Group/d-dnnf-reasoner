@file:JvmName("JavaUtils")

// This files contains code for the Kotlin -> Java interop.

package de.softvare.ddnnife

/**
 * Loads a d-DNNF from file.
 *
 * @param path Where to load the d-DNNF from.
 * @param features How many features the corresponding model has.
 * Can be `null` in which case this will be determined by building the d-DNNF.
 * */
fun ddnnfFromFile(path: String, features: Int?): Ddnnf {
    if (features != null) {
        require(features >= 0) { "Features amount must be positive." }
    }

    return Ddnnf.fromFile(path, features?.toUInt())
}

fun toUInt(i: Int): UInt {
    require(i >= 0) { "Integer must be positive." }
    return i.toUInt()
}

fun enumerate(ddnnf: DdnnfMut, assumptions: List<Int>, amount: Int): List<List<Int>> {
    require(amount >= 0) { "Amount must be positive." }
    return ddnnf.enumerate(assumptions, amount.toULong())
}

fun random(ddnnf: DdnnfMut, assumptions: List<Int>, amount: Int, seed: Int): List<List<Int>> {
    require(amount >= 0) { "Amount must be positive." }
    return ddnnf.random(assumptions, amount.toULong(), seed.toULong())
}

fun atomicSets(ddnnf: DdnnfMut, candidates: List<Int>?, assumptions: List<Int>, cross: Boolean): List<List<Short>> {
    val candidatesUInt = candidates?.map { candidate ->
        require(candidate >= 0) { "Candidates must be positive." }
        candidate.toUInt()
    }

    return ddnnf.atomicSets(candidatesUInt, assumptions, cross)
}

fun sampleTWise(ddnnf: Ddnnf, t: Int): SamplingResult {
    require(t >= 0) { "t must be positive." }
    return ddnnf.sampleTWise(t.toULong())
}

fun isSample(result: SamplingResult): Boolean = when(result) {
    is SamplingResult.ResultWithSample -> true
    else -> false
}

fun isEmpty(result: SamplingResult): Boolean = when(result) {
    is SamplingResult.Empty -> true
    else -> false
}

fun isVoid(result: SamplingResult): Boolean = when(result) {
    is SamplingResult.Void -> true
    else -> false
}

@Throws(IllegalArgumentException::class)
fun getSample(result: SamplingResult): Sample {
    when(result) {
        is SamplingResult.ResultWithSample -> {
            return result.v1
        }
        else -> {
            throw IllegalArgumentException("Invalid sampling result type: ${result::class}")
        }
    }
}
