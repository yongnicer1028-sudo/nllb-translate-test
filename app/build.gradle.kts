plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.yongyong.nllbtest"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.yongyong.nllbtest"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1-phase1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        // onnxruntime / tokenizers native libraries + Java service files can collide; keep the first one.
        resources.pickFirsts.add("META-INF/*")
        jniLibs.useLegacyPackaging = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    // 실제 번역 모델(NLLB)을 폰 안에서 돌리는 엔진 (마이크로소프트 공식 ONNX Runtime)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // NLLB가 쓰는 토크나이저(tokenizer.json)를 그대로 읽어서 문장<->숫자 변환을 해주는 라이브러리.
    // 언어 코드(jpn_Jpan, kor_Hang 등)의 내부 숫자값을 우리가 직접 하드코딩하지 않고
    // 이 라이브러리가 tokenizer.json에서 그대로 읽어오게 해서 실수를 줄인다.
    implementation("ai.djl.huggingface:tokenizers:0.31.1")

    // 모델 파일(수백MB)을 앱 첫 실행 시 다운로드하기 위한 네트워크 라이브러리
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // 비동기 처리
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
