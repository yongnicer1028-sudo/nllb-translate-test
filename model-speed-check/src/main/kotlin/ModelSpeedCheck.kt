// ─────────────────────────────────────────────────────────────────────────
// 지금 앱이 쓰고 있는 "M2M100-418M" 과, 품질이 더 좋을 수도 있는 더 큰 후보
// 모델 "NLLB-200-distilled-600M" 을 같은 조건(폰과 똑같이 스레드 1개로
// 제한한 가벼운 설정, ai.djl 토크나이저, onnxruntime 버전 모두 app 모듈과
// 동일)에서 같은 6개 문장으로 번역해보고,
//   - 다운로드 용량 (폰에 처음 설치할 때 받아야 하는 크기)
//   - 모델 로딩 시간 (앱 켤 때 한 번)
//   - 문장 한 개당 번역 시간 (자막 한 줄 번역할 때마다)
//   - 실제 번역 결과 품질
// 을 나란히 비교해요. (빔서치는 이미 다른 테스트에서 "품질은 조금 좋아지지만
// 40배 넘게 느려짐"으로 확인되어서, 여기서는 그리디 방식만 비교해요 — 두
// 모델 다 실제로 앱에 넣는다면 그리디로 쓸 것이기 때문이에요.)
//
// 두 모델 다 폰 안에서 도는 로컬 모델이라 API 사용료는 없어요(둘 다 무료).
// ─────────────────────────────────────────────────────────────────────────
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

private const val EOS_ID = 2L
private const val DECODER_START_TOKEN_ID = 2L
private const val MAX_NEW_TOKENS = 80

// 그리디 방식은 가끔 "아니, 아니, 아니, ..." 처럼 같은 표현을 끝없이 반복하는
// 유명한 버그가 있어요 (1차 테스트에서 NLLB-200이 실제로 이 버그에 걸려서
// 19초 넘게 걸리고 결과도 깨졌었어요). 최근 3개 토큰이 이미 나왔던 패턴과
// 겹치면 그 다음 토큰을 후보에서 빼는 방식(no-repeat-ngram)으로 막아요.
private const val NO_REPEAT_NGRAM_SIZE = 3

// 예전에 기본 tokenizer.json 에서 발견됐던 병합(merge) 오류를 고쳐서 우리
// 저장소에 올려둔 버전이에요. M2M100 은 반드시 이 고쳐진 버전을 써야 해요.
private const val FIXED_M2M100_TOKENIZER_URL =
    "https://github.com/yongnicer1028-sudo/nllb-translate-test/releases/download/tokenizer-fixed-m2m100/tokenizer.json"

private data class ModelConfig(
    val label: String,
    val folderName: String,
    val tokenizerUrl: String,
    val encoderUrl: String,
    val decoderUrl: String,
    val srcLangCode: String,
    val tgtLangCode: String
)

private val MODELS = listOf(
    ModelConfig(
        label = "M2M100-418M (현재 앱이 쓰는 모델)",
        folderName = "m2m100",
        tokenizerUrl = FIXED_M2M100_TOKENIZER_URL,
        encoderUrl = "https://huggingface.co/Xenova/m2m100_418M/resolve/main/onnx/encoder_model_quantized.onnx",
        decoderUrl = "https://huggingface.co/Xenova/m2m100_418M/resolve/main/onnx/decoder_model_quantized.onnx",
        srcLangCode = "__ja__",
        tgtLangCode = "__ko__"
    ),
    ModelConfig(
        label = "NLLB-200-distilled-600M (더 큰 후보 모델)",
        folderName = "nllb200",
        tokenizerUrl = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/tokenizer.json",
        encoderUrl = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/encoder_model_quantized.onnx",
        decoderUrl = "https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/decoder_model_quantized.onnx",
        srcLangCode = "jpn_Jpan",
        tgtLangCode = "kor_Hang"
    )
)

// translate-quality-check 와 동일한 6개 테스트 문장 (직접 비교 가능하도록 통일)
private val TEST_SENTENCES = listOf(
    "いや、別にそういうつもりじゃなかったんだけど。",
    "マジで無理、これ以上耐えられない。",
    "ちょっと待って、それってどういう意味?",
    "今日はほんとにありがとうございました、助かりました。",
    "え、まさかそんなことになるとは思わなかった。",
    "早くしないと遅刻しちゃうよ!"
)

private val client: HttpClient = HttpClient.newBuilder()
    .followRedirects(HttpClient.Redirect.NORMAL)
    .build()

fun main() {
    val overallSummary = StringBuilder()
    overallSummary.appendLine("| 모델 | 다운로드 용량 | 모델 로딩 시간 | 문장당 평균 번역 시간 |")
    overallSummary.appendLine("|---|---|---|---|")

    val perSentenceSummary = StringBuilder()

    for ((index, model) in MODELS.withIndex()) {
        println()
        println("=== ${index + 1}) ${model.label} ===")
        runModel(model, overallSummary, perSentenceSummary)
    }

    val summaryFile = System.getenv("GITHUB_STEP_SUMMARY")
    if (summaryFile != null) {
        File(summaryFile).appendText(
            "\n## 모델 속도/용량 비교 (M2M100-418M vs NLLB-200-distilled-600M, 둘 다 그리디 + 폰과 동일한 스레드 1개 설정)\n\n" +
                overallSummary.toString() + "\n" + perSentenceSummary.toString() + "\n"
        )
    }

    println()
    println("=== RESULT_DONE ===")
}

private fun runModel(model: ModelConfig, overallSummary: StringBuilder, perSentenceSummary: StringBuilder) {
    val dir = File("models", model.folderName)
    dir.mkdirs()

    val tokFile = File(dir, "tokenizer.json")
    val encFile = File(dir, "encoder.onnx")
    val decFile = File(dir, "decoder.onnx")

    println("파일 다운로드 중...")
    download(model.tokenizerUrl, tokFile)
    download(model.encoderUrl, encFile)
    download(model.decoderUrl, decFile)
    val totalBytes = tokFile.length() + encFile.length() + decFile.length()
    val totalMb = totalBytes / 1024 / 1024
    println("다운로드 완료: 총 ${totalMb}MB (tokenizer ${tokFile.length()} bytes, encoder ${encFile.length()} bytes, decoder ${decFile.length()} bytes)")

    val env = OrtEnvironment.getEnvironment()
    val loadStart = System.currentTimeMillis()
    val encoderSession = env.createSession(encFile.absolutePath, lightweightSessionOptions())
    val decoderSession = env.createSession(decFile.absolutePath, lightweightSessionOptions())
    val tokenizer = HuggingFaceTokenizer.newInstance(tokFile.toPath())
    val loadMs = System.currentTimeMillis() - loadStart
    println("모델 로딩 완료: ${loadMs}ms")

    var totalTranslateMs = 0L
    perSentenceSummary.appendLine()
    perSentenceSummary.appendLine("### ${model.label}")
    perSentenceSummary.appendLine()
    perSentenceSummary.appendLine("| 원문(일본어) | 번역 결과 | 걸린 시간 |")
    perSentenceSummary.appendLine("|---|---|---|")

    try {
        val srcId = langTokenId(tokenizer, model.srcLangCode)
        val tgtId = langTokenId(tokenizer, model.tgtLangCode)

        for (sentence in TEST_SENTENCES) {
            val bodyIds = tokenizer.encode(sentence, false, false).ids
            val inputIds = LongArray(bodyIds.size + 2)
            inputIds[0] = srcId
            for (i in bodyIds.indices) inputIds[i + 1] = bodyIds[i]
            inputIds[inputIds.size - 1] = EOS_ID
            val attnMask = LongArray(inputIds.size) { 1L }

            OnnxTensor.createTensor(env, arrayOf(inputIds)).use { inputTensor ->
                OnnxTensor.createTensor(env, arrayOf(attnMask)).use { maskTensor ->
                    encoderSession.run(mapOf("input_ids" to inputTensor, "attention_mask" to maskTensor)).use { encoderResult ->
                        val encoderHidden = encoderResult.get("last_hidden_state").get() as OnnxTensor

                        val start = System.currentTimeMillis()
                        val ids = greedyDecode(env, decoderSession, encoderHidden, maskTensor, tgtId)
                        val ms = System.currentTimeMillis() - start
                        totalTranslateMs += ms
                        val text = tokenizer.decode(ids, true)

                        println("  원문: $sentence")
                        println("  번역(${ms}ms): $text")
                        perSentenceSummary.appendLine("| $sentence | $text | ${ms}ms |")
                    }
                }
            }
        }
    } finally {
        tokenizer.close()
        encoderSession.close()
        decoderSession.close()
    }

    val avgMs = totalTranslateMs / TEST_SENTENCES.size
    overallSummary.appendLine("| ${model.label} | ${totalMb}MB | ${loadMs}ms | ${avgMs}ms |")
}

private fun lightweightSessionOptions(): OrtSession.SessionOptions {
    val options = OrtSession.SessionOptions()
    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
    options.setIntraOpNumThreads(1)
    return options
}

/** 언어 코드(예: "__ja__", "jpn_Jpan")의 실제 내부 숫자 id를 tokenizer.json에서 읽어와요. */
private fun langTokenId(tokenizer: HuggingFaceTokenizer, langCode: String): Long {
    val ids = tokenizer.encode(langCode, false, false).ids
    check(ids.size == 1) { "언어 코드 '$langCode' 를 토큰 1개로 인식하지 못했어요 (결과: ${ids.toList()})" }
    return ids[0]
}

/** app 모듈의 NllbTranslator.translate와 완전히 똑같은 방식 (그리디: 매번 제일 확률 높은 토큰 1개만 선택). */
private fun greedyDecode(
    env: OrtEnvironment,
    decoderSession: OrtSession,
    encoderHidden: OnnxTensor,
    maskTensor: OnnxTensor,
    tgtId: Long
): LongArray {
    val generated = mutableListOf(DECODER_START_TOKEN_ID, tgtId)
    var steps = 0
    while (steps < MAX_NEW_TOKENS) {
        val decInputIds = generated.toLongArray()
        val banned = bannedNextTokens(generated)
        val nextId = OnnxTensor.createTensor(env, arrayOf(decInputIds)).use { decInputTensor ->
            val inputs = buildDecoderInputs(decoderSession, decInputTensor, encoderHidden, maskTensor)
            decoderSession.run(inputs).use { result ->
                val logits = lastPositionLogits(result)
                var bestId = -1
                var bestScore = Float.NEGATIVE_INFINITY
                for (i in logits.indices) {
                    if (i in banned) continue
                    if (logits[i] > bestScore) {
                        bestScore = logits[i]
                        bestId = i
                    }
                }
                // 이론상 모든 후보가 다 막히는 일은 없지만(벡터 크기가 훨씬 크니까),
                // 혹시 모를 안전장치로 막힌 것도 포함해서 다시 한번 최댓값을 찾아요.
                if (bestId == -1) {
                    for (i in logits.indices) {
                        if (logits[i] > bestScore) {
                            bestScore = logits[i]
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
    return generated.drop(2).filter { it != EOS_ID }.toLongArray()
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

private fun buildDecoderInputs(
    decoderSession: OrtSession,
    decInputTensor: OnnxTensor,
    encoderHidden: OnnxTensor,
    maskTensor: OnnxTensor
): Map<String, OnnxTensor> {
    val inputs = mutableMapOf(
        "input_ids" to decInputTensor,
        "encoder_hidden_states" to encoderHidden
    )
    if (decoderSession.inputNames.contains("encoder_attention_mask")) {
        inputs["encoder_attention_mask"] = maskTensor
    }
    return inputs
}

private fun lastPositionLogits(result: OrtSession.Result): FloatArray {
    val logitsTensor = result.get("logits").get() as OnnxTensor
    @Suppress("UNCHECKED_CAST")
    val logitsArr = logitsTensor.value as Array<Array<FloatArray>>
    return logitsArr[0][logitsArr[0].size - 1]
}

private fun download(url: String, dest: File) {
    val request = HttpRequest.newBuilder(URI.create(url)).GET().build()
    val response = client.send(request, HttpResponse.BodyHandlers.ofFile(dest.toPath()))
    check(response.statusCode() in 200..299) { "다운로드 실패 ($url): HTTP ${response.statusCode()}" }
}
