// ─────────────────────────────────────────────────────────────────────────
// 이 모듈도 안드로이드 앱이 아니에요. 컴퓨터(깃허브 서버)에서 지금 앱이
// 쓰는 M2M100-418M 모델과, 더 큰 후보인 NLLB-200-distilled-600M 모델을
// 둘 다 불러와서 (폰과 똑같이 스레드 1개로 제한한 가벼운 설정으로) 같은
// 문장들을 번역해보고, 다운로드 용량 / 로딩 시간 / 문장당 번역 시간 /
// 번역 결과를 나란히 비교하기 위한 테스트용 모듈이에요.
// ─────────────────────────────────────────────────────────────────────────
plugins {
    kotlin("jvm")
    application
}

dependencies {
    // 앱(app 모듈)이 쓰는 것과 정확히 같은 버전이어야 실제 상황과 똑같이 비교돼요.
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    // onnxruntime-android이 아니라 일반 자바용 onnxruntime이에요 — 컴퓨터(리눅스)에서
    // 그대로 실행하기 위한 버전이고, 안드로이드 버전과 계산 결과는 동일해요.
    implementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
}

application {
    mainClass.set("ModelSpeedCheckKt")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

tasks.named<JavaExec>("run") {
    standardOutput = System.out
    errorOutput = System.err
    // NLLB-200(600M)은 M2M100(418M)보다 다운로드 용량이 커서 힙을 더 넉넉히 줘요.
    jvmArgs("-Xmx6g")
}
