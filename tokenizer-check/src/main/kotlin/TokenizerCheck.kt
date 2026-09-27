// ─────────────────────────────────────────────────────────────────────────
// 실제 폰 앱이 다운로드해서 쓰는 tokenizer.json 파일을 여기(깃허브 서버)에서도
// 똑같이 받아서, 똑같은 토크나이저 라이브러리로 불러와봐요.
//
// 앞서 이분 탐색으로 확인한 결과: merges(합치기 규칙) 목록 안에 "합쳤을 때
// 결과 토큰이 vocab(단어 사전)에 없는" 이상한 항목이 최소 2개 이상 있었어요.
// 하나씩 이분 탐색으로 찾는 건 너무 느리니까, 이번엔 231,277개 merges 전체를
// 한 번에 쭉 검사해서 "문제 있는 항목을 전부" 찾아내고, 그것들만 빼고
// 새로운(고쳐진) tokenizer.json을 만들어서 그게 진짜로 잘 불러와지는지까지
// 확인해요.
// ─────────────────────────────────────────────────────────────────────────
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import com.google.gson.JsonParser
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

private const val TOKENIZER_URL = "https://huggingface.co/Xenova/m2m100_418M/resolve/main/tokenizer.json"

fun main() {
    val dest = File("tokenizer.json")
    println("=== 1) tokenizer.json download start ===")
    println("URL: $TOKENIZER_URL")

    val client = HttpClient.newBuilder()
        .followRedirects(HttpClient.Redirect.NORMAL)
        .build()
    val request = HttpRequest.newBuilder(URI.create(TOKENIZER_URL)).GET().build()
    val response = client.send(request, HttpResponse.BodyHandlers.ofFile(dest.toPath()))
    println("HTTP status: ${response.statusCode()}")
    println("downloaded bytes: ${dest.length()}")

    if (response.statusCode() !in 200..299) {
        println("!!! download itself failed, check status code")
        return
    }

    println()
    println("=== 2) loading with HuggingFaceTokenizer (original file, no changes) ===")
    if (tryLoadFile(dest)) {
        println()
        println("=== RESULT_OK ===")
        return
    }

    println()
    println("=== 3) scanning ALL merge entries for ones whose result is not in vocab ===")
    scanAndFix(dest)
}

private fun tryLoadFile(file: File): Boolean {
    return try {
        val tokenizer = HuggingFaceTokenizer.newInstance(file.toPath())
        println("SUCCESS: tokenizer loaded")
        val ja = tokenizer.encode("こんにちは、元気ですか？", false, false)
        println("sanity encode (Japanese) ids: ${ja.ids.toList()}")
        val ko = tokenizer.decode(ja.ids, true)
        println("sanity decode back: $ko")
        true
    } catch (e: Throwable) {
        println("FAIL: ${e.javaClass.name}: ${e.message}")
        false
    }
}

private fun scanAndFix(originalFile: File) {
    val root = JsonParser.parseReader(originalFile.bufferedReader(Charsets.UTF_8)).asJsonObject
    val model = root.getAsJsonObject("model")
    val merges = model.getAsJsonArray("merges")
    val total = merges.size()
    println("total merges: $total")

    val vocabObj = model.getAsJsonObject("vocab")
    val vocab = HashSet<String>(vocabObj.size() * 2)
    for (key in vocabObj.keySet()) vocab.add(key)
    println("vocab size: ${vocab.size}")

    data class BadEntry(val index: Int, val text: String, val reason: String)
    val bad = ArrayList<BadEntry>()

    for (i in 0 until total) {
        val el = merges[i]
        if (!el.isJsonPrimitive) {
            bad.add(BadEntry(i, el.toString(), "not a string entry"))
            continue
        }
        val s = el.asString
        val parts = s.split(" ")
        if (parts.size != 2 || parts[0].isEmpty() || parts[1].isEmpty()) {
            bad.add(BadEntry(i, s, "does not split into exactly 2 non-empty parts (got ${parts.size})"))
            continue
        }
        val (a, b) = parts
        if (!vocab.contains(a)) {
            bad.add(BadEntry(i, s, "left part '$a' not in vocab"))
            continue
        }
        if (!vocab.contains(b)) {
            bad.add(BadEntry(i, s, "right part '$b' not in vocab"))
            continue
        }
        val merged = a + b
        if (!vocab.contains(merged)) {
            bad.add(BadEntry(i, s, "merged result '$merged' not in vocab"))
            continue
        }
    }

    println()
    println("=== scan complete: ${bad.size} bad merge entries out of $total (${"%.4f".format(bad.size * 100.0 / total)}%) ===")
    println("showing up to 30 examples:")
    for (b in bad.take(30)) {
        println("  [${b.index}] \"${b.text}\" -> ${b.reason}")
    }

    if (bad.isEmpty()) {
        println("no bad entries found by this check, but the original file still failed to load.")
        println("the problem must be somewhere else (not a simple merge-vs-vocab mismatch). stopping here.")
        return
    }

    println()
    println("=== 4) building a FIXED tokenizer.json with all bad merge entries removed ===")
    val badIndices = bad.map { it.index }.toHashSet()
    val fixedMerges = com.google.gson.JsonArray()
    for (i in 0 until total) {
        if (i !in badIndices) fixedMerges.add(merges[i])
    }
    println("fixed merges count: ${fixedMerges.size()} (removed ${total - fixedMerges.size()})")
    model.add("merges", fixedMerges)

    val fixedFile = File("tokenizer.fixed.json")
    fixedFile.writeText(root.toString(), Charsets.UTF_8)
    println("wrote fixed file: ${fixedFile.absolutePath} (${fixedFile.length()} bytes)")

    println()
    println("=== 5) loading the FIXED tokenizer.json to confirm it actually works ===")
    val fixedOk = tryLoadFile(fixedFile)
    println()
    if (fixedOk) {
        println("=== RESULT_FIXED_OK: removing ${bad.size} bad merge entries (out of $total) makes the tokenizer load correctly! ===")
    } else {
        println("=== RESULT_STILL_FAILS: even after removing all ${bad.size} flagged entries, loading still fails. more investigation needed. ===")
    }
}
