package com.yongyong.nllbtest

// ─────────────────────────────────────────────────────────────────────────
// 1단계 테스트판: 음성인식(위스퍼)은 아직 없고, 번역 엔진만 따로 떼서
// "폰 안에서 실제로 잘 돌아가는지" 확인하는 화면이에요.
//
// 구성:
// - ModelManager: 번역에 필요한 파일 3개(인코더, 디코더, 토크나이저)를
// 첫 실행 시 인터넷에서 받아오는 역할
// - NllbTranslator: 실제로 문장을 번역하는 엔진 (ONNX Runtime 사용)
// - MainActivity: 화면 (다운로드 버튼, 입력창, 번역 버튼, 결과창)
//
// ⚠️ 중요: 일본어(jpn_Jpan), 한국어(kor_Hang) 같은 "언어 코드"의 내부 숫자값을
// 이 코드가 직접 하드코딩하지 않아요. 그 값이 정확히 몇 번인지 사람이 손으로
// 계산하면 실수하기 쉬운 부분이라서, 대신 번역 모델과 같이 배포되는
// tokenizer.json 파일에서 "그 코드에 해당하는 숫자가 몇 번인지"를 앱이
// 실행될 때 직접 읽어오게 만들었어요. (langTokenId 함수 참고)
//
// ⚠️ 이번 버전은 "같은 태블릿에서 두 모델의 실제 속도를 직접 비교"하기 위한
// 임시 비교판이에요. 방금 전 버전(NLLB-200-distilled-600M)을 태블릿에서
// 테스트했더니 번역은 잘 됐지만 문장 하나에 7.6초가 걸려서 생각보다 느렸고,
// 깃허브 서버(x86)에서의 비교 테스트 수치(NLLB-200이 M2M100보다 약 30%
// 느림)가 실제 폰/태블릿(ARM 칩)에서도 똑같이 적용되는지는 직접 재봐야
// 확실하다고 판단해서, 잠깐 M2M100-418M(양자화 기준 약 603MB, 더 가벼운
// 모델)으로 되돌려서 같은 문장으로 다시 시간을 재보는 버전이에요. 그리디
// 반복 버그를 막는 안전장치(no-repeat-ngram)는 그대로 유지해요 — 이건
// 모델과 상관없이 계속 필요한 고침이에요.
//
// ⚠️ NLLB-200과 M2M100은 둘 다 내부적으로 같은 아키텍처(M2M100ForConditionalGeneration,
// model_type: m2m_100)라서, EOS/디코더 시작 토큰 번호(둘 다 2번)를 포함한
// 입출력 방식이 완전히 동일해요. 다른 건 언어 코드 표기법 정도예요
// (M2M100은 "__ja__" 같은 형태, NLLB는 "jpn_Jpan" 같은 FLORES-200 코드).
//
// ⚠️ 그리디(매번 제일 확률 높은 토큰만 고르는) 방식은 가끔 "아니, 아니, 아니,
// ..." 처럼 같은 표현을 끝없이 반복하는 유명한 버그가 있어요. 비교 테스트
// 중 NLLB-200에서 실제로 이 버그가 나서(한 문장이 19초 넘게 걸리고 결과도
// 깨짐) "최근 나온 표현이 이미 나왔으면 그 다음 토큰은 후보에서 제외"하는
// 안전장치(no-repeat-ngram)를 추가해서 고쳤어요. (NllbTranslator.translate
// 함수의 bannedNextTokens 참고)
//
// ⚠️ 예전에 huggingface.co의 Xenova/m2m100_418M 저장소 tokenizer.json 파일
// 자체에 버그가 있어서(merge 규칙 1,053개가 vocab에 없는 결과를 만듦) 고친
// 버전을 우리 저장소에 따로 올려서 썼었어요. NLLB-200 저장소의 tokenizer.json
// 은 비교 테스트에서 6문장 모두 정상적으로 처리돼서, 같은 문제는 없는 걸로
// 확인했어요 — 그래서 NLLB는 huggingface 원본 tokenizer.json을 그대로 받아요.
// ─────────────────────────────────────────────────────────────────────────

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.yongyong.nllbtest.databinding.ActivityMainBinding
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

private const val TAG = "NllbTest"

// ═════════════════════════════════════════════════════════════════════════
// 모델 파일 정보 + 다운로드 담당
// ═════════════════════════════════════════════════════════════════════════

/** 다운로드가 끝난 뒤, 실제로 받은 파일들의 로컬 경로를 담아요. */
data class ModelPaths(
    val encoderPath: String,
    val decoderPath: String,
    val tokenizerPath: String
)

object ModelManager {

    // 직전 버전(NLLB-200)의 태블릿 실측 속도가 생각보다 느려서, 같은
    // 태블릿에서 직접 비교하려고 잠깐 M2M100-418M(약 603MB, 더 가벼운
    // 모델)으로 되돌렸어요. (자세한 이유는 파일 맨 위 주석 참고)
    private const val REPO = "Xenova/m2m100_418M"
    private const val API_URL = "https://huggingface.co/api/models/$REPO"
    private const val FILE_BASE_URL = "https://huggingface.co/$REPO/resolve/main/"

    // 어떤 파일 "종류"를 골라서 받았는지 나타내는 버전 번호예요. 나중에 파일
    // 선택 로직(resolveFileNames)을 고치면 이 숫자를 1씩 올려주세요 — 그래야
    // 예전 로직으로 잘못 받아둔 파일(예: 용량이 훨씬 큰 fp32/int8 디코더)이
    // 기기에 남아있어도, 앱이 그걸 재사용하지 않고 새 로직으로 다시 받아요.
    // (이 값이 없으면: 디코더 선택 로직을 고쳐도, 이미 큰 파일을 받아버린
    // 태블릿에서는 새 코드가 적용 안 되고 계속 옛날의 큰 파일을 불러오려다
    // 메모리 부족으로 앱이 꺼지는 걸 반복하게 돼요.)
    // 5로 올려서, 기기에 남아있는 NLLB-200 파일을 무시하고 M2M100을 새로 받게 해요.
    private const val MODEL_SCHEMA_VERSION = 5

    // M2M100 저장소 원본 tokenizer.json에는 merge 규칙 1,053개가 vocab에 없는
    // 결과를 만드는 버그가 있어서, 고친 버전을 우리 저장소에 따로 올려서 써요.
    // (파일 맨 위 주석 참고)
    private const val TOKENIZER_URL =
        "https://github.com/yongnicer1028-sudo/nllb-translate-test/releases/download/tokenizer-fixed-m2m100/tokenizer.json"

    // 지금 기기에 받아져 있는 tokenizer.json이 "몇 번째 버전"인지 표시하는
    // 값이에요. 이 값을 바꾸면(모델을 바꾸거나, 토크나이저 파일 자체가
    // 나중에 또 바뀌면) 이미 받아둔 인코더/디코더(수백MB)는 그대로 두고
    // 토크나이저 파일(수십MB)만 다시 받아요. NLLB-200 → M2M100으로 모델
    // 자체를 다시 바꿨으니 이번엔 3으로 올려요.
    private const val TOKENIZER_VERSION = 3

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private fun modelsDir(context: android.content.Context): File {
        val dir = File(context.filesDir, "nllb_model")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * 이미 파일 3개가 전부 정상적으로 다운로드되어 있는지 확인.
     * (1단계라 정확한 용량 검증까지는 안 하고, 파일 존재+용량>0 만 확인해요.
     * 나중에 문제가 생기면 이 부분에 체크섬 검증을 추가하면 좋아요.)
     */
    fun isDownloaded(context: android.content.Context): ModelPaths? {
        val dir = modelsDir(context)
        val versionFile = File(dir, "schema_version.txt")
        val savedVersion = if (versionFile.exists()) {
            versionFile.readText().trim().toIntOrNull()
        } else null
        if (savedVersion != MODEL_SCHEMA_VERSION) {
            // 옛날 로직/모델로 받아둔 파일일 수 있어요. 안전하게 새로
            // 받도록 "없음" 취급해요.
            return null
        }

        val enc = File(dir, "encoder.onnx")
        val dec = File(dir, "decoder.onnx")
        val tok = File(dir, "tokenizer.json")
        if (enc.length() <= 0 || dec.length() <= 0 || tok.length() <= 0) return null

        // 인코더/디코더는 위에서 이미 확인했으니, 토크나이저 파일만 "지금
        // 버전"인지 따로 확인해요. 이 버전이 다르면 download()가 알아서
        // 토크나이저 파일만 다시 받고, 이미 받아둔 인코더/디코더는 그대로
        // 재사용해요 (수백MB를 또 받을 필요가 없어요).
        val tokVersionFile = File(dir, "tokenizer_fix_version.txt")
        val savedTokVersion = if (tokVersionFile.exists()) {
            tokVersionFile.readText().trim().toIntOrNull()
        } else null
        if (savedTokVersion != TOKENIZER_VERSION) return null

        return ModelPaths(enc.absolutePath, dec.absolutePath, tok.absolutePath)
    }

    /**
     * huggingface 저장소의 실제 파일 목록을 조회해서, 인코더/디코더 onnx 파일
     * 이름이 정확히 뭔지 찾아내요. (버전에 따라 파일명이 다를 수 있어서,
     * 이름을 하드코딩하는 대신 "이런 패턴을 포함하는 파일"을 직접 찾아요)
     */
    private fun resolveFileNames(): Pair<String, String> {
        val request = Request.Builder().url(API_URL).build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string() ?: throw IllegalStateException("모델 파일 목록을 못 가져왔어요.")
            val root = org.json.JSONObject(body)
            val siblings: JSONArray = root.getJSONArray("siblings")
            val names = mutableListOf<String>()
            for (i in 0 until siblings.length()) {
                names.add(siblings.getJSONObject(i).getString("rfilename"))
            }

            // 파일 이름 우선순위: "_quantized.onnx" 로 끝나는 게 제일 작고
            // 가벼운 표준 8비트 양자화 버전이에요. "int8"/"uint8" 이라는
            // 이름만 보면 더 작을 것 같지만, 실제로는(NLLB-200 기준)
            // quantized 파일보다 3배 넘게 큰 경우가 있어서 이름만 보고
            // 고르면 안 돼요 — 정확한 접미사로 구분해요.
            fun priority(name: String): Int = when {
                name.endsWith("_quantized.onnx") -> 0
                name.contains("int8") || name.contains("uint8") -> 1
                else -> 2
            }

            // 인코더: onnx/ 폴더 안에서 encoder_model 이 들어간 파일 중 제일 가벼운 것
            val encoderName = names
                .filter { it.startsWith("onnx/") && it.contains("encoder_model") && it.endsWith(".onnx") }
                .sortedBy { priority(it) }
                .firstOrNull()
                ?: throw IllegalStateException("인코더(.onnx) 파일을 저장소에서 못 찾았어요. 전체 목록: $names")

            // 디코더: 캐시 없이 통째로 계산하는 "decoder_model.onnx" 형태를 최우선으로 찾고,
            // 없으면 "decoder_model_merged"(캐시 지원 버전)로 대체 — 이 경우는 지금 버전 코드가
            // 아직 처리 못 하니 나중에 에러 메시지로 알려줘요.
            val plainDecoder = names
                .filter {
                    it.startsWith("onnx/") && it.contains("decoder_model") &&
                        !it.contains("merged") && !it.contains("with_past") && it.endsWith(".onnx")
                }
                .sortedBy { priority(it) }
                .firstOrNull()
            val mergedDecoder = names
                .filter { it.startsWith("onnx/") && it.contains("decoder_model_merged") && it.endsWith(".onnx") }
                .sortedBy { priority(it) }
                .firstOrNull()
            val decoderName = plainDecoder ?: mergedDecoder
                ?: throw IllegalStateException("디코더(.onnx) 파일을 저장소에서 못 찾았어요. 전체 목록: $names")

            Log.i(TAG, "선택된 파일 -> encoder: $encoderName, decoder: $decoderName")
            return Pair(encoderName, decoderName)
        }
    }

    /**
     * 필요한 파일만 골라서 다운로드해요 — 이미 정상적으로 받아둔 파일은 다시
     * 안 받아요. 예를 들어 토크나이저 파일만 바뀌었을 땐 인코더/디코더
     * (수백MB)는 그대로 두고 토크나이저 파일만 다시 받아요.
     * [onProgress] 는 0~100 사이 값으로, 이번에 실제로 받는 파일들 기준
     * 진행률을 알려줘요.
     */
    suspend fun download(context: android.content.Context, onProgress: (Int) -> Unit): ModelPaths {
        val dir = modelsDir(context)
        val encFile = File(dir, "encoder.onnx")
        val decFile = File(dir, "decoder.onnx")
        val tokFile = File(dir, "tokenizer.json")

        val schemaOk = File(dir, "schema_version.txt").let {
            it.exists() && it.readText().trim().toIntOrNull() == MODEL_SCHEMA_VERSION
        }
        val encDecReady = schemaOk && encFile.length() > 0 && decFile.length() > 0

        val targets = mutableListOf<Pair<String, File>>()
        if (!encDecReady) {
            val (encoderName, decoderName) = resolveFileNames()
            targets.add((FILE_BASE_URL + encoderName) to encFile)
            targets.add((FILE_BASE_URL + decoderName) to decFile)
        }

        val tokVersionFile = File(dir, "tokenizer_fix_version.txt")
        val tokReady = tokFile.length() > 0 &&
            tokVersionFile.exists() &&
            tokVersionFile.readText().trim().toIntOrNull() == TOKENIZER_VERSION
        if (!tokReady) {
            targets.add(TOKENIZER_URL to tokFile)
        }

        // 파일마다 크기가 크게 달라서(모델 수백MB vs 토크나이저 수십MB), 미리 각 파일 크기를 안 뒤
        // 그냥 "몇 번째 파일"로 대충 진행률을 나누면 부정확해요. 그래서 실제 바이트 수 기준으로 계산해요.
        if (targets.isNotEmpty()) {
            val sizes = LongArray(targets.size)
            for (i in targets.indices) {
                sizes[i] = headContentLength(targets[i].first)
            }
            val totalBytes = sizes.sum().coerceAtLeast(1L)
            var doneBytesBase = 0L
            // 진행률 콜백이 매 64KB마다 불려서 (큰 파일은 수천 번) UI 쪽에 너무 자주 업데이트를
            // 요청하지 않도록, 정수 퍼센트 값이 실제로 바뀔 때만 onProgress를 호출해요.
            var lastReportedPercent = -1

            for (i in targets.indices) {
                val (url, localFile) = targets[i]
                downloadOne(url, localFile) { bytesSoFarThisFile ->
                    val overall = ((doneBytesBase + bytesSoFarThisFile).toDouble() / totalBytes * 100).toInt()
                        .coerceIn(0, 100)
                    if (overall != lastReportedPercent) {
                        lastReportedPercent = overall
                        onProgress(overall)
                    }
                }
                doneBytesBase += sizes[i]
            }
        }

        onProgress(100)
        // 새로 받은 파일들 기준으로, 이번에 어떤 버전 로직/토크나이저로 받았는지 기록해둡요.
        // (다음에 앱을 켰을 때 isDownloaded()가 이 값을 보고 재사용 여부를 판단해요)
        File(dir, "schema_version.txt").writeText(MODEL_SCHEMA_VERSION.toString())
        tokVersionFile.writeText(TOKENIZER_VERSION.toString())
        return ModelPaths(
            encFile.absolutePath,
            decFile.absolutePath,
            tokFile.absolutePath
        )
    }

    private fun headContentLength(url: String): Long {
        return try {
            val request = Request.Builder().url(url).head().build()
            client.newCall(request).execute().use { it.header("Content-Length")?.toLongOrNull() ?: 0L }
        } catch (e: Exception) {
            0L
        }
    }

    private fun downloadOne(url: String, dest: File, onBytes: (Long) -> Unit) {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("다운로드 실패 ($url): HTTP ${response.code}")
            }
            val body = response.body ?: throw IllegalStateException("다운로드 실패 ($url): 내용 없음")
            body.byteStream().use { input ->
                FileOutputStream(dest).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var bytesRead: Int
                    var total = 0L
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        total += bytesRead
                        onBytes(total)
                    }
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 실제 번역 엔진
// ═════════════════════════════════════════════════════════════════════════

data class TranslateResult(val text: String, val elapsedMs: Long)

class NllbTranslator(paths: ModelPaths) : AutoCloseable {

    companion object {
        // NLLB-200과 M2M100은 같은 아키텍처(model_type: m2m_100)라서 config.json
        // 값이 동일해요: eos_token_id=2, decoder_start_token_id=2
        // (두 값이 같아요 — EOS를 디코더 시작 신호로도 같이 써요).
        private const val EOS_ID = 2L
        private const val DECODER_START_TOKEN_ID = 2L
        private const val MAX_NEW_TOKENS = 80

        // 그리디 방식은 가끔 같은 표현을 끝없이 반복하는 버그가 있어서(예:
        // "아니, 아니, 아니, ..."), 최근에 나온 표현이 이미 나온 적 있으면
        // 그 다음 토큰을 후보에서 제외하는 안전장치예요.
        private const val NO_REPEAT_NGRAM_SIZE = 3
    }

    private val env = OrtEnvironment.getEnvironment()

    /**
     * 모델을 불러올 때 쓰는 옵션이에요. 기본값 그대로 두면 그래프 최적화 작업과
     * 여러 스레드용 계산 버퍼 때문에 "불러오는 바로 그 순간"에 메모리를 더 많이
     * 써요 — 이게 태블릿에서 메모리 부족으로 앱이 조용히 꺼지는 원인 중 하나로
     * 보여서, 메모리를 아끼는 쪽으로 설정을 낮췄어요. (그 대신 번역 속도는
     * 아주 약간 느려질 수 있어요.)
     */
    private fun lightweightSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(1)
        }

    private val encoderSession: OrtSession = env.createSession(paths.encoderPath, lightweightSessionOptions())
    private val decoderSession: OrtSession = env.createSession(paths.decoderPath, lightweightSessionOptions())
    private val tokenizer: HuggingFaceTokenizer = HuggingFaceTokenizer.newInstance(File(paths.tokenizerPath).toPath())

    init {
        // 지금 코드는 "캐시 없이 매번 통째로 계산하는" 단순한 디코더만 지원해요.
        // 혹시 다운로드된 파일이 "캐시 지원(merged)" 형태라면, 여기서 미리 명확한
        // 에러를 던져서 나중에 원인 모를 이상한 결과가 나오는 걸 막아요.
        val decoderInputs = decoderSession.inputNames
        if (decoderInputs.any { it.contains("past_key_values") || it.contains("use_cache_branch") }) {
            throw UnsupportedOperationException(
                "다운로드된 번역모델의 디코더가 '캐시 지원(merged)' 형식이라 지금 코드가 아직 처리 못 해요.\n" +
                    "디코더 입력 목록: $decoderInputs\n" +
                    "이 메시지를 그대로 캡처해서 알려주시면 코드를 그 형식에 맞게 고칠게요."
            )
        }
    }

    /** 문자열로 된 언어 코드(예: "jpn_Jpan")의 실제 내부 숫자 id를 tokenizer.json에서 읽어와요. */
    private fun langTokenId(langCode: String): Long {
        val encoding = tokenizer.encode(langCode, false, false)
        val ids = encoding.ids
        if (ids.size != 1) {
            throw IllegalStateException(
                "언어 코드 '$langCode' 를 토큰 1개로 인식하지 못했어요 (결과: ${ids.toList()}). " +
                    "이 메시지를 캡처해서 알려주세요."
            )
        }
        return ids[0]
    }

    /** 최근에 나온 (NO_REPEAT_NGRAM_SIZE - 1)개 토큰 패턴이 예전에도 나온 적 있다면,
     *  그 뒤에 이어졌던 토큰을 이번에는 후보에서 빼서 같은 구절이 무한 반복되는 걸 막아요. */
    private fun bannedNextTokens(generated: List<Long>): Set<Int> {
        val prefixLen = NO_REPEAT_NGRAM_SIZE - 1
        if (generated.size < prefixLen) return emptySet()
        val prefix = generated.takeLast(prefixLen)
        val banned = mutableSetOf<Int>()
        for (i in 0..generated.size - prefixLen - 1) {
            if (generated.subList(i, i + prefixLen) == prefix) {
                banned.add(generated[i + prefixLen].toInt())
            }
        }
        return banned
    }

    fun translate(text: String, srcLang: String, tgtLang: String): TranslateResult {
        val startTime = System.currentTimeMillis()

        val srcId = langTokenId(srcLang)
        val tgtId = langTokenId(tgtLang)
        val bodyIds = tokenizer.encode(text, false, false).ids

        // 입력 형식: [소스 언어 코드] + 실제 문장 토큰들 + [문장끝 표시]
        val inputIds = LongArray(bodyIds.size + 2)
        inputIds[0] = srcId
        for (i in bodyIds.indices) inputIds[i + 1] = bodyIds[i]
        inputIds[inputIds.size - 1] = EOS_ID
        val attnMask = LongArray(inputIds.size) { 1L }

        OnnxTensor.createTensor(env, arrayOf(inputIds)).use { inputTensor ->
            OnnxTensor.createTensor(env, arrayOf(attnMask)).use { maskTensor ->
                encoderSession.run(mapOf("input_ids" to inputTensor, "attention_mask" to maskTensor)).use { encoderResult ->
                    val encoderHidden = encoderResult.get("last_hidden_state")
                        .orElseThrow {
                            IllegalStateException(
                                "인코더 출력에서 last_hidden_state를 못 찾았어요. " +
                                    "실제 출력 이름: ${encoderSession.outputNames}"
                            )
                        } as OnnxTensor

                    // 한 글자(토큰)씩 순서대로 만들어나가요 (제일 확률 높은 걸 그대로 고르되,
                    // 같은 표현이 반복되는 건 막아요).
                    val generated = mutableListOf(DECODER_START_TOKEN_ID, tgtId)
                    var steps = 0
                    while (steps < MAX_NEW_TOKENS) {
                        val decInputIds = generated.toLongArray()
                        val banned = bannedNextTokens(generated)
                        val nextId = OnnxTensor.createTensor(env, arrayOf(decInputIds)).use { decInputTensor ->
                            val decoderInputsMap = mutableMapOf<String, OnnxTensor>(
                                "input_ids" to decInputTensor,
                                "encoder_hidden_states" to encoderHidden
                            )
                            if (decoderSession.inputNames.contains("encoder_attention_mask")) {
                                decoderInputsMap["encoder_attention_mask"] = maskTensor
                            }
                            decoderSession.run(decoderInputsMap).use { decoderResult ->
                                val logitsTensor = decoderResult.get("logits")
                                    .orElseThrow {
                                        IllegalStateException(
                                            "디코더 출력에서 logits를 못 찾았어요. " +
                                                "실제 출력 이름: ${decoderSession.outputNames}"
                                        )
                                    } as OnnxTensor

                                @Suppress("UNCHECKED_CAST")
                                val logitsArr = logitsTensor.value as Array<Array<FloatArray>>
                                val lastPosition = logitsArr[0][logitsArr[0].size - 1]
                                var bestId = -1
                                var bestScore = Float.NEGATIVE_INFINITY
                                for (i in lastPosition.indices) {
                                    if (i in banned) continue
                                    if (lastPosition[i] > bestScore) {
                                        bestScore = lastPosition[i]
                                        bestId = i
                                    }
                                }
                                // 이론상 모든 후보가 다 막히는 일은 없지만(벡터 크기가
                                // 훨씬 크니까), 혹시 모를 안전장치로 막힌 것도 포함해서
                                // 다시 한번 최댓값을 찾아요.
                                if (bestId == -1) {
                                    for (i in lastPosition.indices) {
                                        if (lastPosition[i] > bestScore) {
                                            bestScore = lastPosition[i]
                                            bestId = i
                                        }
                                    }
                                }
                                bestId.toLong()
                            }
                        }
                        generated.add(nextId)
                        steps++
                        if (nextId == EOS_ID) break
                    }

                    // 맨 앞의 "목표 언어 코드"랑 마지막 EOS는 실제 번역 문장이 아니니 빼고 글자로 되돌려요.
                    val outputIds = generated.drop(2).filter { it != EOS_ID }.toLongArray()
                    val resultText = tokenizer.decode(outputIds, true)
                    val elapsed = System.currentTimeMillis() - startTime
                    return TranslateResult(resultText, elapsed)
                }
            }
        }
    }

    override fun close() {
        encoderSession.close()
        decoderSession.close()
        tokenizer.close()
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 화면
// ═════════════════════════════════════════════════════════════════════════

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val activityJob = Job()
    private val activityScope = CoroutineScope(Dispatchers.Main + activityJob)

    private var translator: NllbTranslator? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnDownloadModel.setOnClickListener { startDownloadOrLoad() }
        binding.btnTranslate.setOnClickListener { runTranslate() }

        binding.textBuildInfo.text = try {
            "빌드 버전: " + packageManager.getPackageInfo(packageName, 0).versionName
        } catch (e: Exception) {
            ""
        }

        checkExistingModel()
    }

    private fun checkExistingModel() {
        val existing = ModelManager.isDownloaded(this)
        if (existing != null) {
            binding.textModelStatus.text = "모델 상태: 이미 받아둔 파일이 있어요. 불러오는 중…"
            loadTranslator(existing)
        } else {
            binding.textModelStatus.text = "모델 상태: 아직 다운로드 안 됨"
        }
    }

    private fun startDownloadOrLoad() {
        val existing = ModelManager.isDownloaded(this)
        if (existing != null) {
            loadTranslator(existing)
            return
        }

        binding.btnDownloadModel.isEnabled = false
        binding.progressDownload.visibility = android.view.View.VISIBLE
        binding.textModelStatus.text = "모델 상태: 다운로드 중… (와이파이 권장, 시간이 좀 걸려요)"

        activityScope.launch {
            try {
                val paths = withContext(Dispatchers.IO) {
                    ModelManager.download(this@MainActivity) { percent ->
                        activityScope.launch {
                            binding.progressDownload.progress = percent
                            binding.textModelStatus.text = "모델 상태: 다운로드 중… ($percent%)"
                        }
                    }
                }
                loadTranslator(paths)
            } catch (e: Exception) {
                Log.e(TAG, "다운로드 실패", e)
                binding.textModelStatus.text = "모델 상태: 다운로드 실패 - ${e.message}"
                binding.btnDownloadModel.isEnabled = true
            }
        }
    }

    private fun loadTranslator(paths: ModelPaths) {
        binding.textModelStatus.text = "모델 상태: 불러오는 중…"
        activityScope.launch {
            try {
                val loaded = withContext(Dispatchers.IO) { NllbTranslator(paths) }
                translator = loaded
                binding.textModelStatus.text = "모델 상태: 준비 완료 ✅"
                binding.progressDownload.visibility = android.view.View.GONE
                binding.btnDownloadModel.isEnabled = true
                binding.btnDownloadModel.text = "모델 다시 불러오기"
                binding.btnTranslate.isEnabled = true
            } catch (e: CancellationException) {
                // 화면을 나가서 정상적으로 취소된 경우예요. 이건 진짜 오류가 아니니까
                // 그대로 다시 던져서 코루틴이 원래 하던 대로 정리되게 둬요.
                throw e
            } catch (e: Throwable) {
                // 원래는 Exception만 잡았는데, 메모리가 부족해서 나는 OutOfMemoryError는
                // Exception이 아니라 Error라서 그동안 여기서 안 잡히고 앱이 통째로
                // 조용히 꺼져버렸어요 (에러 메시지도 없이 그냥 화면이 홈으로 내려가는
                // 것처럼 보였던 이유가 이거예요). Throwable로 바꿔서 이제는 화면에
                // 원인을 보여주고 앱은 안 죽게 만들었어요.
                Log.e(TAG, "모델 불러오기 실패", e)
                val reason = if (e is OutOfMemoryError) {
                    "메모리 부족 (이 기기에서 번역 모델을 불러오기엔 램이 부족해요)"
                } else {
                    e.message ?: e.toString()
                }
                binding.textModelStatus.text = "모델 상태: 불러오기 실패 - $reason"
                binding.btnDownloadModel.isEnabled = true
                Toast.makeText(this@MainActivity, "실패 내용을 캡처해서 알려주세요", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun runTranslate() {
        val engine = translator
        if (engine == null) {
            Toast.makeText(this, "먼저 모델을 다운로드/불러와주세요.", Toast.LENGTH_SHORT).show()
            return
        }
        val text = binding.editInput.text.toString().trim()
        if (text.isEmpty()) {
            Toast.makeText(this, "일본어 문장을 입력해주세요.", Toast.LENGTH_SHORT).show()
            return
        }

        binding.btnTranslate.isEnabled = false
        binding.textOutput.text = "번역 중…"
        binding.textTiming.text = ""

        activityScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    engine.translate(text, "__ja__", "__ko__")
                }
                binding.textOutput.text = result.text
                binding.textTiming.text = "걸린 시간: ${result.elapsedMs} ms"
            } catch (e: Exception) {
                Log.e(TAG, "번역 실패", e)
                binding.textOutput.text = "오류: ${e.message}"
                Toast.makeText(this@MainActivity, "번역 중 오류가 났어요. 화면을 캡처해서 알려주세요", Toast.LENGTH_LONG).show()
            } finally {
                binding.btnTranslate.isEnabled = true
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        activityJob.cancel()
        translator?.close()
    }
}
