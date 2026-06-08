package com.example.codesearcb

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Minimal BERT WordPiece tokenizer that mirrors HuggingFace's BertTokenizer
 * (do_lower_case=true, no accent stripping).  Only needs vocab.txt from assets.
 */
class BertTokenizer(context: Context, vocabAsset: String = "vocab.txt") {

    private val vocab: Map<String, Int>

    init {
        val map = LinkedHashMap<String, Int>()
        context.assets.open(vocabAsset).use { stream ->
            BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).lineSequence().forEachIndexed { idx, line ->
                map[line.trim()] = idx
            }
        }
        vocab = map
    }

    private val clsId = vocab["[CLS]"] ?: error("[CLS] not in vocab")
    private val sepId = vocab["[SEP]"] ?: error("[SEP] not in vocab")
    private val padId = vocab["[PAD]"] ?: 0
    private val unkId = vocab["[UNK]"] ?: error("[UNK] not in vocab")

    data class Encoding(val inputIds: LongArray, val attentionMask: LongArray)

    /**
     * Encode a single string into fixed-length int64 arrays ready for the model.
     * @param maxLength  must match what you pass to OnnxEmbedder.embed()
     */
    fun encode(text: String, maxLength: Int = 512): Encoding {
        val tokens = tokenize(text.lowercase())
        // [CLS] + up to (maxLength-2) tokens + [SEP]
        val ids = ArrayList<Int>(maxLength)
        ids.add(clsId)
        tokens.take(maxLength - 2).forEach { ids.add(it) }
        ids.add(sepId)

        val inputIds = LongArray(maxLength) { padId.toLong() }
        val mask     = LongArray(maxLength) { 0L }
        ids.forEachIndexed { i, id ->
            inputIds[i] = id.toLong()
            mask[i] = 1L
        }
        return Encoding(inputIds, mask)
    }

    // ── internals ─────────────────────────────────────────────────────────────

    private fun tokenize(text: String): List<Int> {
        val result = mutableListOf<Int>()
        for (word in basicTokenize(text)) result.addAll(wordpiece(word))
        return result
    }

    private fun basicTokenize(text: String): List<String> {
        val words = mutableListOf<String>()
        val buf   = StringBuilder()
        for (ch in text) {
            when {
                ch.isWhitespace() -> flush(buf, words)
                isPunct(ch)       -> { flush(buf, words); words.add(ch.toString()) }
                else              -> buf.append(ch)
            }
        }
        flush(buf, words)
        return words
    }

    private fun flush(buf: StringBuilder, out: MutableList<String>) {
        if (buf.isNotEmpty()) { out.add(buf.toString()); buf.clear() }
    }

    private fun wordpiece(word: String): List<Int> {
        val out   = mutableListOf<Int>()
        var start = 0
        while (start < word.length) {
            var end   = word.length
            var found = -1
            while (start < end) {
                val sub = if (start == 0) word.substring(start, end)
                          else            "##${word.substring(start, end)}"
                val id = vocab[sub]
                if (id != null) { found = id; break }
                end--
            }
            if (found == -1) { out.add(unkId); break }
            out.add(found)
            start = end
        }
        return out
    }

    private fun isPunct(ch: Char): Boolean {
        val cp = ch.code
        if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)) return true
        return ch.category in setOf(
            CharCategory.CONNECTOR_PUNCTUATION, CharCategory.DASH_PUNCTUATION,
            CharCategory.START_PUNCTUATION,     CharCategory.END_PUNCTUATION,
            CharCategory.INITIAL_QUOTE_PUNCTUATION, CharCategory.FINAL_QUOTE_PUNCTUATION,
            CharCategory.OTHER_PUNCTUATION
        )
    }
}
