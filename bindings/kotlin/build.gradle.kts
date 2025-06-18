import com.sun.jna.Platform
import org.gradle.internal.os.OperatingSystem

group = "de.softvare"
version = "0.9.0"

plugins {
    java
    kotlin("jvm") version "2.1.+"
    id("com.github.johnrengelman.shadow") version "8.+"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.+")
    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.+")
}

buildscript {
    dependencies {
        classpath("net.java.dev.jna:jna:5.+")
    }
}

// Directory structure for generated files.
val generatedResources = "${layout.buildDirectory.get()}/main/resources"

val singleLibrary = hasProperty("library")
val multipleLibraries = hasProperty("libraries")

// When no library oder library folder is passed, we only target the current system and directly use Rust to build the library.
val onlyCurrent = !singleLibrary && !multipleLibraries

val resourcePrefix: String = if (hasProperty("generatePrefix")) {
    property("generatePrefix").toString()
} else {
    Platform.RESOURCE_PREFIX
}

val libraryName: String = if (hasProperty("generateLib")) {
    property("generateLib").toString()
} else {
    OperatingSystem.current().getSharedLibraryName("ddnnife")
}

val libraryDest = "${generatedResources}/${resourcePrefix}"

var librariesPath = ""

if (onlyCurrent) {
    librariesPath = "../../target/release/${libraryName}"
}

if (singleLibrary) {
    librariesPath = property("library").toString()
}

if (multipleLibraries) {
    librariesPath = property("libraries").toString()
}

// The bindgen tool can be passed via the `bindgen` property, otherwise we invoke it via cargo.
val bindgen = if (hasProperty("bindgen")) {
    listOf(property("bindgen").toString())
} else {
    listOf("cargo", "run", "--bin", "uniffi-bindgen")
}

tasks.register<Copy>("nativeLibrary") {
    group = "Build"
    description = "Copies the native library."

    if (onlyCurrent || singleLibrary) {
        from(librariesPath)
        into(libraryDest)
    }

    if (multipleLibraries) {
        from(librariesPath)
        into(generatedResources)
    }
}

tasks.processResources {
    dependsOn("nativeLibrary")
}

// Skip building the Rust library in case its path was given.
if (onlyCurrent) {
    tasks.register<Exec>("buildRust") {
        group = "Build"
        description = "Compiles the Rust crate."
        commandLine("cargo", "build", "--release", "--package", "ddnnife_ffi")
    }

    tasks.named("nativeLibrary") {
        dependsOn("buildRust")
    }
}

tasks.register<Exec>("generateBindings") {
    group = "Build"
    description = "Generates the Kotlin uniffi bindings for the Rust crate."
    commandLine(bindgen)
    args("generate", "--language", "kotlin", "--out-dir", "${layout.projectDirectory}/src/main/kotlin", "--library", "${libraryDest}/${libraryName}", "--metadata-no-deps", "--no-format")

    dependsOn("nativeLibrary")
}

tasks.compileKotlin {
    dependsOn("generateBindings")
}

tasks.test {
    useJUnitPlatform()

    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

sourceSets {
    main {
        resources {
            srcDir(generatedResources)
        }
    }
}
