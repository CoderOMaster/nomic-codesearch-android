package com.example.codesearcb

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Loads model.onnx from assets and returns L2-normalised 768-dim embeddings.
 *
 * Usage:
 *   val embedder = OnnxEmbedder(context)
 *
 *   // embed code or queries
 *   val emb: FloatArray = embedder.embed("def add(a, b): return a + b")
 *
 *   // semantic code search
 *   val results = embedder.search("add two numbers", listOf(...), topK = 3)
 *
 *   embedder.close()   // release native resources when done
 */
class OnnxEmbedder(context: Context) : AutoCloseable {

    private val tokenizer = BertTokenizer(context)
    private val env       = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // Copy asset to cache file if it doesn't exist, then load by path.
        // This avoids loading the entire ~130MB model into the JVM heap.
        val modelFile = java.io.File(context.cacheDir, "model.onnx")
        if (!modelFile.exists()) {
            context.assets.open("model.onnx").use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        session = env.createSession(modelFile.absolutePath)
    }

    // ── public API ────────────────────────────────────────────────────────────

    /** Embed a single string; returns a normalised FloatArray of size 768. */
    fun embed(text: String, maxLength: Int = 512): FloatArray =
        embedBatch(listOf(text), maxLength)[0]

    /**
     * Embed a list of strings in one ONNX call (more efficient than calling
     * embed() in a loop when you have many snippets).
     */
    fun embedBatch(texts: List<String>, maxLength: Int = 512): Array<FloatArray> {
        require(texts.isNotEmpty()) { "texts must not be empty" }
        val encodings = texts.map { tokenizer.encode(it, maxLength) }
        val batchSize = texts.size
        val seqLen    = maxLength.toLong()

        // Fill flat LongBuffers in row-major order
        val idsBuf  = LongBuffer.allocate(batchSize * maxLength)
        val maskBuf = LongBuffer.allocate(batchSize * maxLength)
        for (enc in encodings) {
            idsBuf.put(enc.inputIds)
            maskBuf.put(enc.attentionMask)
        }
        idsBuf.rewind(); maskBuf.rewind()

        val shape      = longArrayOf(batchSize.toLong(), seqLen)
        val inputIds   = OnnxTensor.createTensor(env, idsBuf,  shape)
        val attMask    = OnnxTensor.createTensor(env, maskBuf, shape)

        val result     = session.run(mapOf("input_ids" to inputIds, "attention_mask" to attMask))
        val rawBatch   = (result["sentence_embedding"].get().value as Array<*>)
            .map { it as FloatArray }

        inputIds.close(); attMask.close(); result.close()

        return Array(batchSize) { i -> l2Normalize(rawBatch[i]) }
    }

    /**
     * Find the top-k most similar code snippets for a natural-language query.
     * Returns pairs of (snippet, cosine_similarity).
     */
    fun search(
        query: String,
        snippets: List<String>,
        topK: Int = 3,
        maxLength: Int = 512,
    ): List<Pair<String, Float>> {
        val queryEmb = embed(query, maxLength)
        val codeEmbs = embedBatch(snippets, maxLength)
        return snippets
            .zip(codeEmbs.map { dot(queryEmb, it) })
            .sortedByDescending { it.second }
            .take(topK)
    }

    override fun close() {
        session.close()
        env.close()
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private fun l2Normalize(v: FloatArray): FloatArray {
        val norm = sqrt(v.fold(0f) { acc, x -> acc + x * x })
        return if (norm < 1e-12f) v else FloatArray(v.size) { v[it] / norm }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
