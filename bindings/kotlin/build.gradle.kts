import com.sun.jna.Platform
import org.gradle.internal.os.OperatingSystem

group = "de.softvare"
version = "0.7.0"

plugins {
    java
    kotlin("jvm") version "2.0.0"
    id("org.jetbrains.dokka") version "1.9.20"
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

kotlin {
    jvmToolchain(17)
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.14.0")
    testImplementation(kotlin("test"))
}

buildscript {
    dependencies {
        classpath("net.java.dev.jna:jna:5.14.0")
    }
}

// Directory structure for generated files.
val generatedResources = "${layout.buildDirectory.get()}/main/resources"

// When no library folder is passed, we only target the current system and directly use Rust to build the library.
val onlyCurrent = !hasProperty("libraries")

// Otherwise, extract the path to the pre-built libraries.
var librariesPath = ""
if (!onlyCurrent) {
    librariesPath = property("libraries").toString()
}

// OS specific directories for the native library.
// This only covers the current platform used for generation.
val os: OperatingSystem = OperatingSystem.current()
val resourcePrefix = Platform.RESOURCE_PREFIX;
val libraryDest = "${generatedResources}/${resourcePrefix}"
val libraryName: String = os.getSharedLibraryName("ddnnife")

// The bindgen tool can be passed via the `bindgen` property, otherwise we invoke it via cargo.
val bindgen = if (hasProperty("bindgen")) {
    listOf(property("bindgen").toString())
} else {
    listOf("cargo", "run", "--bin", "uniffi-bindgen")
}

tasks.register<Copy>("nativeLibrary") {
    group = "Build"
    description = "Copies the native library."

    if (onlyCurrent) {
        from("../../target/release/${libraryName}")
        into(libraryDest)
    } else {
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
        commandLine("cargo", "build", "--release", "--package", "ddnnife", "--features", "uniffi")
    }

    tasks.named("nativeLibrary") {
        dependsOn("buildRust")
    }
}

tasks.register<Exec>("generateBindings") {
    group = "Build"
    description = "Generates the Kotlin uniffi bindings for the Rust crate."
    commandLine(bindgen)
    args("generate", "--language", "kotlin", "--out-dir", "${layout.projectDirectory}/src/main/kotlin", "--library", "${libraryDest}/${libraryName}", "--no-format")

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
