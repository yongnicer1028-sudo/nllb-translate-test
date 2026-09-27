// ─────────────────────────────────────────────────────────────────────────
// 이 모듈도 안드로이드 앱이 아니에요. 컴퓨터(깃허브 서버)에서 실제 번역
// 모델(인코더+디코더 onnx)을 통째로 불러와서, "그리디(현재 방식)" 와
// "빔서치(개선 후보)" 두 가지 방식으로 같은 문장들을 번역해보고 결과를
// 비교하기 위한 테스트용 모듈이에요. 폰 설치 없이, 번역 품질을 먼저
// 확인해보기 위한 용도예요.
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
    // 허깅페이스 저장소의 파일 목록(JSON)을 읽어오기 위한 JSON 처리 라이브러리
    // (안드로이드에는 org.json이 기본 내장이지만, 일반 자바에는 없어서 따로 추가해요).
    implementation("com.google.code.gson:gson:2.10.1")
}

application {
    mainClass.set("TranslateQualityCheckKt")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

tasks.named<JavaExec>("run") {
    standardOutput = System.out
    errorOutput = System.err
    // 모델 파일이 수백MB라 기본 힙 메모리로는 부족할 수 있어요.
    jvmArgs("-Xmx4g")
}
