package com.example.codesearch   // ← must match what Android Studio set for your project

import android.os.Bundle
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var embedder: OnnxEmbedder

    // ── sample snippets to search through ────────────────────────────────────
    // Replace or extend this list with your own code snippets.
    private val snippets = listOf(
        "def add(a, b):\n    return a + b",

        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",

        "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",

        "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",

        "import json\nwith open('data.json') as f:\n    data = json.load(f)",

        "import os\nfiles = [f for f in os.listdir('.') if f.endswith('.py')]",

        "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)\n    def pop(self): return self.items.pop()\n    def is_empty(self): return len(self.items) == 0",

        "SELECT * FROM users WHERE age > 18 ORDER BY name ASC",

        "SELECT COUNT(*) FROM orders WHERE status = 'pending'",

        "for i, line in enumerate(open('file.txt')):\n    print(i, line.strip())",

        "words = ['hello', 'world']\nresult = ' '.join(words)",

        "import hashlib\nhash = hashlib.sha256(b'data').hexdigest()",

        "import re\nmatches = re.findall(r'\\d+', 'abc123def456')",

        "from datetime import datetime\nnow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')",
    )

    // ── activity setup ────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val queryInput   = findViewById<EditText>(R.id.queryInput)
        val searchButton = findViewById<Button>(R.id.searchButton)
        val progressBar  = findViewById<ProgressBar>(R.id.progressBar)
        val statusText   = findViewById<TextView>(R.id.statusText)
        val resultsText  = findViewById<TextView>(R.id.resultsText)

        // Load the 131 MB model on a background thread so the UI stays responsive
        lifecycleScope.launch {
            progressBar.visibility = View.VISIBLE
            withContext(Dispatchers.IO) {
                embedder = OnnxEmbedder(applicationContext)
            }
            progressBar.visibility = View.GONE
            statusText.text = "Ready — type a query and press Search"
            searchButton.isEnabled = true
        }

        fun doSearch() {
            val query = queryInput.text.toString().trim()
            if (query.isEmpty() || !::embedder.isInitialized) return

            searchButton.isEnabled = false
            progressBar.visibility = View.VISIBLE
            resultsText.text = ""
            statusText.text = "Searching…"

            lifecycleScope.launch {
                val results = withContext(Dispatchers.IO) {
                    embedder.search(query, snippets, topK = 3)
                }
                progressBar.visibility = View.GONE
                searchButton.isEnabled = true
                statusText.text = "Showing top 3 results for: \"$query\""

                val sb = StringBuilder()
                results.forEachIndexed { i, (snippet, score) ->
                    val pct = (score * 100).toInt()
                    sb.append("━━━  #${i + 1}  match: $pct%  ━━━\n\n")
                    sb.append(snippet).append("\n\n")
                }
                resultsText.text = sb.toString()
            }
        }

        searchButton.setOnClickListener { doSearch() }

        // allow pressing the keyboard "Search" action key
        queryInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH) { doSearch(); true } else false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::embedder.isInitialized) embedder.close()
    }
}
