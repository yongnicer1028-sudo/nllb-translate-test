// ─────────────────────────────────────────────────────────────────────────
// 이 모듈은 안드로이드 앱이 아니에요. 컴퓨터(깃허브 서버)에서 일반 자바 프로그램
// 처럼 실행해서, "폰 앱이 쓰는 것과 똑같은 토크나이저 라이브러리 버전"이 실제
// tokenizer.json 파일을 잘 읽는지 직접 확인해보기 위한 테스트용 모듈이에요.
// (폰으로 매번 테스트 안 해도, 여기서 먼저 재현/확인해볼 수 있어요)
// ─────────────────────────────────────────────────────────────────────────
plugins {
    kotlin("jvm")
    application
}

dependencies {
    // 앱(app 모듈)이 쓰는 것과 정확히 같은 버전이어야 같은 문제가 재현돼요.
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    // tokenizer.json 안의 merges 목록을 직접 잘라보면서(이분 탐색) 정확히 몇 번째
    // merge 항목이 문제인지 찾아내기 위한 JSON 처리용 라이브러리예요.
    implementation("com.google.code.gson:gson:2.10.1")
}

application {
    mainClass.set("TokenizerCheckKt")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

tasks.named<JavaExec>("run") {
    standardOutput = System.out
    errorOutput = System.err
}
