// ════════════════════════════════════════════════════════════════════════════
//  What to ADD to your existing app/build.gradle.kts
//  Do NOT replace the whole file — just paste these two blocks into it.
// ════════════════════════════════════════════════════════════════════════════

// ── 1. Inside your existing  android { }  block, add this: ─────────────────

androidResources {
    // Don't re-compress the model file (it's already compressed).
    noCompress += "onnx"
}

// ── 2. Inside your existing  dependencies { }  block, add these two lines: ─

implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
