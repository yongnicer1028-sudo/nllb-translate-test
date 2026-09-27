import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
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
                                                              println("=== 2) loading with HuggingFaceTokenizer ===")
                                                                  try {
                                                                            val tokenizer = HuggingFaceTokenizer.newInstance(dest.toPath())
                                                                                    println("SUCCESS: tokenizer loaded")

                                                                                            println()
                                                                                                    println("=== 3) lang token id check (__ja__, __ko__) ===")
                                                                                                            for (lang in listOf("__ja__", "__ko__")) {
                                                                                                                          val enc = tokenizer.encode(lang, false, false)
                                                                                                                                      println("$lang -> ids=${enc.ids.toList()}")
                                                                                                            }
                                                                                                            
                                                                                                                    println()
                                                                                                                            println("=== 4) sample encode/decode ===")
                                                                                                                                    val sample = "こんにちは"
                                                                            val enc = tokenizer.encode(sample, false, false)
                                                                                    println("input: $sample")
                                                                                            println("ids: ${enc.ids.toList()}")
                                                                                                    val decoded = tokenizer.decode(enc.ids, true)
                                                                                                            println("decoded back: $decoded")
                                                                                                            
                                                                                                                    println()
                                                                                                                            println("=== RESULT_OK ===")
                                                                  } catch (e: Throwable) {
                                                                            println()
                                                                                    println("=== RESULT_FAIL ===")
                                                                                            println("error type: ${e.javaClass.name}")
                                                                                                    println("error message: ${e.message}")
                                                                                                            println()
                                                                                                                    println("--- stacktrace ---")
                                                                                                                            e.printStackTrace(System.out)
                                                                  }
}
