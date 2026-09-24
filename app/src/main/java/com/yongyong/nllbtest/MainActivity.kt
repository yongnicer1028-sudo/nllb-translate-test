package com.yongyong.nllbtest

// ─────────────────────────────────────────────────────────────────────────
// 1단계 테스트판: 음성인식(위스퍼)은 아직 없고, 번역 엔진(NLLB)만 따로 떼서
// "폰 안에서 실제로 잘 돌아가는지" 확인하는 화면이에요.
//
// 구성:
//  - ModelManager: 번역에 필요한 파일 3개(인코더, 디코더, 토크나이저)를
//    첫 실행 시 인터넷에서 받아오는 역할
//  - NllbTranslator: 실제로 문장을 번역하는 엔진 (ONNX Runtime 사용)
//  - MainActivity: 화면 (다운로드 버튼, 입력창, 번역 버튼, 결과창)
//
// ⚠️ 중요: 일본어(jpn_Jpan), 한국어(kor_Hang) 같은 "언어 코드"의 내부 숫자값을
// 이 코드가 직접 하드코딩하지 않아요. 그 값이 정확히 몇 번인지 사람이 손으로
// 계산하면 실수하기 쉬운 부분이라서, 대신 번역 모델과 같이 배포되는
// tokenizer.json 파일에서 "그 코드에 해당하는 숫자가 몇 번인지"를 앱이
// 실행될 때 직접 읽어오게 만들었어요. (langTokenId 함수 참고)
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

    private const val REPO = "Xenova/nllb-200-distilled-600M"
    private const val API_URL = "https://huggingface.co/api/models/$REPO"
    private const val FILE_BASE_URL = "https://huggingface.co/$REPO/resolve/main/"

    // 어떤 파일 "종류"를 골라서 받았는지 나타내는 버전 번호예요. 나중에 파일
    // 선택 로직(resolveFileNames)을 고치면 이 숫자를 1씩 올려주세요 — 그래야
    // 예전 로직으로 잘못 받아둔 파일(예: 용량이 훨씬 큰 fp32 디코더)이 기기에
    // 남아있어도, 앱이 그걸 재사용하지 않고 새 로직으로 다시 받아요.
    // (이 값이 없으면: 디코더 선택 로직을 고쳐도, 이미 큰 파일을 받아버린
    //  태블릿에서는 새 코드가 적용 안 되고 계속 옛날의 큰 파일을 불러오려다
    //  메모리 부족으로 앱이 꺼지는 걸 반복하게 돼요.)
    private const val MODEL_SCHEMA_VERSION = 2

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
     * 이미 파일 3개가 전부 다운로드되어 있는지 확인.
     * (1단계라 정확한 용량 검증까지는 안 하고, 파일 존재+용량>0 만 확인해요.
     *  나중에 문제가 생기면 이 부분에 체크섬 검증을 추가하면 좋아요.)
     */
    fun isDownloaded(context: android.content.Context): ModelPaths? {
        val dir = modelsDir(context)
        val versionFile = File(dir, "schema_version.txt")
        val savedVersion = if (versionFile.exists()) {
            versionFile.readText().trim().toIntOrNull()
        } else null
        if (savedVersion != MODEL_SCHEMA_VERSION) {
            // 옷날 로직으로 받아둔 파일일 수 있어요 (예: 메모리 부족을 일으키는
            // 큰 디코더). 안전하게 새로 받도록 "없음" 취급해요.
            return null
        }

        val enc = File(dir, "encoder.onnx")
        val dec = File(dir, "decoder.onnx")
        val tok = File(dir, "tokenizer.json")
        return if (enc.length() > 0 && dec.length() > 0 && tok.length() > 0) {
            ModelPaths(enc.absolutePath, dec.absolutePath, tok.absolutePath)
        } else null
    }

    /**
     * huggingface 저장소의 실제 파일 목록을 조회해서, 인코더/디코더 onnx 파일
     * 이름이 정확히 뭔지 찾아내요. (버전에 따라 파일명이 다를 수 있어서,
     * 이름을 하드코딩하는 대신 "이런 패턴을 포함하는 파일"을 직접 찾아요)
     */
    private fun resolveFileNames(): Triple<String, String, String> {
        val request = Request.Builder().url(API_URL).build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string() ?: throw IllegalStateException("모델 파일 목록을 못 가져왔어요.")
            val root = org.json.JSONObject(body)
            val siblings: JSONArray = root.getJSONArray("siblings")
            val names = mutableListOf<String>()
            for (i in 0 until siblings.length()) {
                names.add(siblings.getJSONObject(i).getString("rfilename"))
            }

            // 인코더: onnx/ 폴더 안에서 encoder_model 이 들어간 것 중, quantized(가벼운 버전) 우선
            val encoderName = names
                .filter { it.startsWith("onnx/") && it.contains("encoder_model") && it.endsWith(".onnx") }
                .sortedBy { name -> if (name.contains("quantized") || name.contains("int8")) 0 else 1 }
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
                .sortedBy { name -> if (name.contains("quantized") || name.contains("int8")) 0 else 1 }
                .firstOrNull()
val mergedDecoder = names
                .filter { it.startsWith("onnx/") && it.contains("decoder_model_merged") && it.endsWith(".onnx") }
                .sortedBy { name -> if (name.contains("quantized") || name.contains("int8")) 0 else 1 }
                .firstOrNull()
            val decoderName = plainDecoder ?: mergedDecoder
                ?: throw IllegalStateException("디코더(.onnx) 파일을 저장소에서 못 찾았어요. 전체 목록: $names")

            Log.i(TAG, "선택된 파일 -> encoder: $encoderName, decoder: $decoderName")
            return Triple(encoderName, decoderName, "tokenizer.json")
        }
    }

    /**
     * 파일 3개를 순서대로 다운로드. [onProgress] 는 0~100 사이 값으로 전체 진행률을 알려줘요.
     */
    suspend fun download(context: android.content.Context, onProgress: (Int) -> Unit): ModelPaths {
        val (encoderName, decoderName, tokenizerName) = resolveFileNames()
        val dir = modelsDir(context)

        val targets = listOf(
            encoderName to File(dir, "encoder.onnx"),
            decoderName to File(dir, "decoder.onnx"),
            tokenizerName to File(dir, "tokenizer.json")
        )

        // 파일마다 크기가 크게 달라서(모델 수백MB vs 토크나이저 수십MB), 미리 각 파일 크기를 안 뒤
        // 그냥 "몇 번째 파일"로 대충 진행률을 나누면 부정확해요. 그래서 실제 바이트 수 기준으로 계산해요.
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
            val (remoteName, localFile) = targets[i]
            downloadOne(remoteName, localFile) { bytesSoFarThisFile ->
                val overall = ((doneBytesBase + bytesSoFarThisFile).toDouble() / totalBytes * 100).toInt()
                    .coerceIn(0, 100)
                if (overall != lastReportedPercent) {
                    lastReportedPercent = overall
                    onProgress(overall)
                }
            }
            doneBytesBase += sizes[i]
        }

        onProgress(100)
        // 새로 다 받았으니, 이번에 어떤 버전 로직으로 받았는지 기록해둡요.
        // (다음에 앱을 켰을 때 isDownloaded()가 이 값을 보고 재사용 여부를 판단해요)
        File(dir, "schema_version.txt").writeText(MODEL_SCHEMA_VERSION.toString())
        return ModelPaths(
            targets[0].second.absolutePath,
            targets[1].second.absolutePath,
            targets[2].second.absolutePath
        )
    }

    private fun headContentLength(remoteName: String): Long {
        return try {
            val request = Request.Builder().url(FILE_BASE_URL + remoteName).head().build()
            client.newCall(request).execute().use { it.header("Content-Length")?.toLongOrNull() ?: 0L }
        } catch (e: Exception) {
            0L
        }
    }

    private fun downloadOne(remoteName: String, dest: File, onBytes: (Long) -> Unit) {
        val request = Request.Builder().url(FILE_BASE_URL + remoteName).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("다운로드 실패 ($remoteName): HTTP ${response.code}")
            }
            val body = response.body ?: throw IllegalStateException("다운로드 실패 ($remoteName): 내용 없음")
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
        private const val EOS_ID = 2L
        private const val MAX_NEW_TOKENS = 80
    }

    private val env = OrtEnvironment.getEnvironment()
    private val encoderSession: OrtSession = env.createSession(paths.encoderPath)
    private val decoderSession: OrtSession = env.createSession(paths.decoderPath)
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

    fun translate(text: String, srcLang: String, tgtLang: String): TranslateResult {
        val startTime = System.currentTimeMillis()

        val srcId = langTokenId(srcLang)
        val tgtId = langTokenId(tgtLang)
        val bodyIds = tokenizer.encode(text, false, false).ids

        // NLLB 입력 형식: [소스 언어 코드] + 실제 문장 토큰들 + [문장끝 표시]
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

                    // 한 글자(토큰)씩 순서대로 만들어나가요 (제일 확률 높은 걸 그대로 고르는 방식).
                    val generated = mutableListOf(tgtId)
                    var steps = 0
                    while (steps < MAX_NEW_TOKENS) {
                        val decInputIds = generated.toLongArray()
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
                                var bestId = 0
                                var bestScore = Float.NEGATIVE_INFINITY
                                for (i in lastPosition.indices) {
                                    if (lastPosition[i] > bestScore) {
                                        bestScore = lastPosition[i]
                                        bestId = i
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
                    val outputIds = generated.drop(1).filter { it != EOS_ID }.toLongArray()
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
            } catch (e: Exception) {
                Log.e(TAG, "모델 불러오기 실패", e)
                binding.textModelStatus.text = "모델 상태: 불러오기 실패 - ${e.message}"
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
                    engine.translate(text, "jpn_Jpan", "kor_Hang")
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
