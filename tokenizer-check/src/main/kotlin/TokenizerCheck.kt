// ─────────────────────────────────────────────────────────────────────────
// 실제 폰 앱이 다운로드해서 쓰는 tokenizer.json 파일을 여기(깃허브 서버)에서도
// 똑같이 받아서, 똑같은 토크나이저 라이브러리로 불러와봐요.
// 여기서 성공하면 -> 문제는 "안드로이드에서만" 생기는 문제라는 뜻.
// 여기서도 똑같이 실패하면 -> tokenizer.json 파일 자체(또는 이 라이브러리
//   버전과의 궁합) 문제라는 뜻이라서, 폰으로 안 옮겨봐도 원인을 알 수 있어요.
//
// 추가로: 만약 실패한다면, merges(합치기 규칙) 목록을 반으로 자르고 또 자르고
// 하면서(이분 탐색) "정확히 몇 번째 merge 줄이 문제인지"까지 찾아내요.
// ─────────────────────────────────────────────────────────────────────────
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import com.google.gson.JsonArray
import com.google.gson.JsonObject
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
    val originalOk = tryLoadFile(dest)
    if (originalOk) {
        println()
        println("=== RESULT_OK ===")
        return
    }

    println()
    println("=== 3) bisecting the merges list to find the exact bad entry ===")
    findBadMergeEntry(dest)
}

private fun tryLoadFile(file: File): Boolean {
    return try {
        val tokenizer = HuggingFaceTokenizer.newInstance(file.toPath())
        println("SUCCESS: tokenizer loaded")
        val enc = tokenizer.encode("こんにちは", false, false)
        println("sanity encode ids: ${enc.ids.toList()}")
        true
    } catch (e: Throwable) {
        println("FAIL: ${e.javaClass.name}: ${e.message}")
        false
    }
}

private fun findBadMergeEntry(originalFile: File) {
    val root = JsonParser.parseReader(originalFile.bufferedReader(Charsets.UTF_8)).asJsonObject
    val model = root.getAsJsonObject("model")
    val merges = model.getAsJsonArray("merges")
    val total = merges.size()
    println("total merges: $total")

    val tempFile = File("tokenizer_bisect.json")

    // prefixLen 개의 merge만 남겼을 때 성공하는지 확인하는 함수
    fun tryPrefix(prefixLen: Int): Boolean {
        val truncated = JsonArray()
        for (i in 0 until prefixLen) {
            truncated.add(merges[i])
        }
        model.add("merges", truncated)
        tempFile.writeText(root.toString(), Charsets.UTF_8)
        return tryLoadFileQuiet(tempFile)
    }

    // 이분 탐색: prefixLen이 lo면 성공, hi면 실패하는 상태를 유지하면서 좁혀나가요.
    var lo = 0
    var hi = total
    // 우선 0개(성공해야 정상)와 전체(실패해야 정상)를 확인
    println("checking prefix length 0 (should succeed)...")
    if (!tryPrefix(0)) {
        println("!!! even 0 merges fails -> problem is NOT in merges list itself (maybe vocab/added_tokens/other section). Stopping bisection.")
        return
    }
    println("checking prefix length $total (should fail, same as original)...")
    if (tryPrefix(total)) {
        println("!!! full merges list actually succeeded here (unstable?). Stopping bisection.")
        return
    }

    var steps = 0
    while (hi - lo > 1) {
        val mid = (lo + hi) / 2
        val ok = tryPrefix(mid)
        steps++
        println("step $steps: prefix length $mid -> ${if (ok) "OK" else "FAIL"}")
        if (ok) lo = mid else hi = mid
    }

    val badIndex = hi - 1 // 0-based index of the first bad merge entry
    println()
    println("=== FOUND: first bad merge entry is at 0-based index $badIndex (1-based #${badIndex + 1} of $total) ===")

    val context = 5
    val start = maxOf(0, badIndex - context)
    val end = minOf(total - 1, badIndex + context)
    println("context around the bad entry:")
    for (i in start..end) {
        val marker = if (i == badIndex) " <-- BAD" else ""
        println("  [$i] ${merges[i]}$marker")
    }

    val badEntry = merges[badIndex]
    println()
    println("bad entry raw JSON: $badEntry")
    if (badEntry.isJsonPrimitive) {
        val s = badEntry.asString
        println("bad entry as string: ${s.let { "\"" + it + "\"" }}")
        println("bad entry char count: ${s.length}")
        println("bad entry chars (codepoints): ${s.map { "'${it}' (U+%04X)".format(it.code) }}")
        val parts = s.split(" ")
        println("split by single space -> ${parts.size} parts: $parts")
    }

    // 이 하나의 항목만 빼면 전체 나머지가 정상 로딩되는지 최종 확인
    println()
    println("=== 4) double-check: removing ONLY this one bad entry, does the rest load fine? ===")
    val withoutBad = JsonArray()
    for (i in 0 until total) {
        if (i != badIndex) withoutBad.add(merges[i])
    }
    model.add("merges", withoutBad)
    tempFile.writeText(root.toString(), Charsets.UTF_8)
    val fixedOk = tryLoadFile(tempFile)
    println()
    if (fixedOk) {
        println("=== RESULT: removing just entry #$badIndex fixes loading! ===")
    } else {
        println("=== RESULT: still fails even after removing entry #$badIndex, more than one bad entry may exist. ===")
    }
}

private fun tryLoadFileQuiet(file: File): Boolean {
    return try {
        HuggingFaceTokenizer.newInstance(file.toPath())
        true
    } catch (e: Throwable) {
        false
    }
}
