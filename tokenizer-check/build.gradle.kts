plugins {
    kotlin("jvm")
    application
}

dependencies {
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
}

application {
    mainClass.set("TokenizerCheckKt")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}
