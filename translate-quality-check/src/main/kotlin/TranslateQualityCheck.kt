// ─────────────────────────────────────────────────────────────────────────
// 폰 설치 없이, 깃허브 서버에서 실제 번역 모델(인코더+디코더)을 통째로
// 불러와서 몇 문장을 번역해봐요. 지금 앱이 쓰는 "그리디"(매 순간 제일
// 그럴듯한 단어 하나만 고르고 다시 안 돌아보는 방식)와, 개선 후보인
// "빔서치"(여러 후보 문장을 동시에 따져보다가 제일 자연스러운 걸 고르는
// 방식) 두 가지로 같은 문장을 번역해서 결과를 나란히 비교해요.
//
// 여기 쓰는 인코더/디코더/토크나이저 로딩·번역 로직은 app 모듈의
// NllbTranslator와 최대한 똑같이 맞췄어요 — 그래야 여기서 나온 결과가
// 실제 폰에서 나올 결과랑 같다고 믿을 수 있어요.
// ─────────────────────────────────────────────────────────────────────────
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.gson.JsonParser
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

private const val REPO = "Xenova/m2m100_418M"
private const val API_URL = "https://huggingface.co/api/models/$REPO"
private const val FILE_BASE_URL = "https://huggingface.co/$REPO/resolve/main/"
private const val FIXED_TOKENIZER_URL =
    "https://github.com/yongnicer1028-sudo/nllb-translate-test/releases/download/tokenizer-fixed-m2m100/tokenizer.json"

private const val EOS_ID = 2L
private const val DECODER_START_TOKEN_ID = 2L
private const val MAX_NEW_TOKENS = 80
private const val BEAM_WIDTH = 4
private const val LENGTH_PENALTY = 0.6

// 실제 영상 자막에 나올 법한, 격식체/반말/감탄/질문이 섞인 일본어 문장들이에요.
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
    val dir = File("models")
    dir.mkdirs()

    val tokFile = File(dir, "tokenizer.json")
    println("=== 1) 파일 다운로드 ===")
    download(FIXED_TOKENIZER_URL, tokFile)
    println("tokenizer.json: ${tokFile.length()} bytes")

    val (encoderName, decoderName) = resolveFileNames()
    val encFile = File(dir, "encoder.onnx")
    val decFile = File(dir, "decoder.onnx")
    download(FILE_BASE_URL + encoderName, encFile)
    println("encoder.onnx: ${encFile.length()} bytes")
    download(FILE_BASE_URL + decoderName, decFile)
    println("decoder.onnx: ${decFile.length()} bytes")

    println()
    println("=== 2) 모델 불러오기 ===")
    val env = OrtEnvironment.getEnvironment()
    val encoderSession = env.createSession(encFile.absolutePath, OrtSession.SessionOptions())
    val decoderSession = env.createSession(decFile.absolutePath, OrtSession.SessionOptions())
    val tokenizer = HuggingFaceTokenizer.newInstance(tokFile.toPath())
    println("불러오기 완료")

    val srcId = langTokenId(tokenizer, "__ja__")
    val tgtId = langTokenId(tokenizer, "__ko__")

    val summary = StringBuilder()
    summary.appendLine("| 원문(일본어) | 그리디(현재 방식) | 빔서치(개선 후보) |")
    summary.appendLine("|---|---|---|")

    println()
    println("=== 3) 문장별 번역 비교 ===")
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

                    val greedyStart = System.currentTimeMillis()
                    val greedyIds = greedyDecode(env, decoderSession, encoderHidden, maskTensor, tgtId)
                    val greedyMs = System.currentTimeMillis() - greedyStart
                    val greedyText = tokenizer.decode(greedyIds, true)

                    val beamStart = System.currentTimeMillis()
                    val beamIds = beamSearchDecode(env, decoderSession, encoderHidden, maskTensor, tgtId)
                    val beamMs = System.currentTimeMillis() - beamStart
                    val beamText = tokenizer.decode(beamIds, true)

                    println("원문: $sentence")
                    println("  그리디(${greedyMs}ms): $greedyText")
                    println("  빔서치(${beamMs}ms): $beamText")
                    println()

                    summary.appendLine(
                        "| $sentence | $greedyText (${greedyMs}ms) | $beamText (${beamMs}ms) |"
                    )
                }
            }
        }
    }

    tokenizer.close()
    encoderSession.close()
    decoderSession.close()

    val summaryFile = System.getenv("GITHUB_STEP_SUMMARY")
    if (summaryFile != null) {
        File(summaryFile).appendText(
            "\n## 번역 품질 비교 (그리디 vs 빔서치)\n\n" + summary.toString() + "\n"
        )
    }

    println("=== RESULT_DONE ===")
}

/** 언어 코드(예: "__ja__")의 실제 내부 숫자 id를 tokenizer.json에서 읽어와요. */
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
        val nextId = OnnxTensor.createTensor(env, arrayOf(decInputIds)).use { decInputTensor ->
            val inputs = buildDecoderInputs(decoderSession, decInputTensor, encoderHidden, maskTensor)
            decoderSession.run(inputs).use { result ->
                val logits = lastPositionLogits(result)
                var bestId = 0
                var bestScore = Float.NEGATIVE_INFINITY
                for (i in logits.indices) {
                    if (logits[i] > bestScore) {
                        bestScore = logits[i]
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
    return generated.drop(2).filter { it != EOS_ID }.toLongArray()
}

private data class Beam(val ids: List<Long>, val score: Double, val finished: Boolean)

private fun beamScore(beam: Beam): Double = beam.score / beam.ids.size.toDouble().pow(LENGTH_PENALTY)

/** 여러 후보를 동시에 유지하면서, 매 단계마다 전체적으로 제일 그럴듯한 조합을 찾아나가요. */
private fun beamSearchDecode(
    env: OrtEnvironment,
    decoderSession: OrtSession,
    encoderHidden: OnnxTensor,
    maskTensor: OnnxTensor,
    tgtId: Long
): LongArray {
    var beams = listOf(Beam(listOf(DECODER_START_TOKEN_ID, tgtId), 0.0, false))
    val completed = mutableListOf<Beam>()
    var step = 0

    while (step < MAX_NEW_TOKENS && beams.isNotEmpty()) {
        val candidates = mutableListOf<Beam>()
        for (beam in beams) {
            if (beam.finished) {
                completed.add(beam)
                continue
            }
            val decInputIds = beam.ids.toLongArray()
            val logProbs = OnnxTensor.createTensor(env, arrayOf(decInputIds)).use { decInputTensor ->
                val inputs = buildDecoderInputs(decoderSession, decInputTensor, encoderHidden, maskTensor)
                decoderSession.run(inputs).use { result -> logSoftmax(lastPositionLogits(result)) }
            }
            for (idx in topKIndices(logProbs, BEAM_WIDTH)) {
                val newIds = beam.ids + idx.toLong()
                candidates.add(Beam(newIds, beam.score + logProbs[idx], idx.toLong() == EOS_ID))
            }
        }
        beams = candidates.sortedByDescending { beamScore(it) }.take(BEAM_WIDTH)
        step++
    }
    completed.addAll(beams)

    val best = completed.maxByOrNull { beamScore(it) } ?: Beam(listOf(DECODER_START_TOKEN_ID, tgtId), 0.0, true)
    return best.ids.drop(2).filter { it != EOS_ID }.toLongArray()
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

private fun logSoftmax(logits: FloatArray): DoubleArray {
    val maxLogit = logits.max().toDouble()
    var sumExp = 0.0
    val shifted = DoubleArray(logits.size)
    for (i in logits.indices) {
        shifted[i] = logits[i].toDouble() - maxLogit
        sumExp += exp(shifted[i])
    }
    val logSumExp = ln(sumExp)
    return DoubleArray(logits.size) { shifted[it] - logSumExp }
}

private fun topKIndices(scores: DoubleArray, k: Int): List<Int> =
    scores.indices.sortedByDescending { scores[it] }.take(k)

/** app 모듈의 ModelManager.resolveFileNames와 같은 로직 (org.json 대신 gson 사용). */
private fun resolveFileNames(): Pair<String, String> {
    val request = HttpRequest.newBuilder(URI.create(API_URL)).GET().build()
    val response = client.send(request, HttpResponse.BodyHandlers.ofString())
    val root = JsonParser.parseString(response.body()).asJsonObject
    val siblings = root.getAsJsonArray("siblings")
    val names = mutableListOf<String>()
    for (el in siblings) names.add(el.asJsonObject.get("rfilename").asString)

    val encoderName = names
        .filter { it.startsWith("onnx/") && it.contains("encoder_model") && it.endsWith(".onnx") }
        .sortedBy { if (it.contains("quantized") || it.contains("int8")) 0 else 1 }
        .firstOrNull() ?: error("인코더(.onnx) 파일을 못 찾았어요. 전체 목록: $names")

    val plainDecoder = names
        .filter { it.startsWith("onnx/") && it.contains("decoder_model") && !it.contains("merged") && !it.contains("with_past") && it.endsWith(".onnx") }
        .sortedBy { if (it.contains("quantized") || it.contains("int8")) 0 else 1 }
        .firstOrNull()
    val mergedDecoder = names
        .filter { it.startsWith("onnx/") && it.contains("decoder_model_merged") && it.endsWith(".onnx") }
        .sortedBy { if (it.contains("quantized") || it.contains("int8")) 0 else 1 }
        .firstOrNull()
    val decoderName = plainDecoder ?: mergedDecoder ?: error("디코더(.onnx) 파일을 못 찾았어요. 전체 목록: $names")

    println("선택된 파일 -> encoder: $encoderName, decoder: $decoderName")
    return Pair(encoderName, decoderName)
}

private fun download(url: String, dest: File) {
    val request = HttpRequest.newBuilder(URI.create(url)).GET().build()
    val response = client.send(request, HttpResponse.BodyHandlers.ofFile(dest.toPath()))
    check(response.statusCode() in 200..299) { "다운로드 실패 ($url): HTTP ${response.statusCode()}" }
}
