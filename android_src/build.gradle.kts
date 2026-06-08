// ── app/build.gradle.kts ─────────────────────────────────────────────────────
// Add this inside your existing android { } and dependencies { } blocks.

android {
    // Tell the asset packager not to compress .onnx files — they are already
    // compressed and re-compressing wastes APK build time with no size gain.
    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    // ONNX Runtime for Android (CPU-only, ~1 MB AAR)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
