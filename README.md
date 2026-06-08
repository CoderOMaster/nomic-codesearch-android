# On-Device Semantic Code Search (Android & ONNX)

This repository contains the complete pipeline and native Android application for **on-device semantic code search**. 

By fine-tuning `nomic-embed-text-v1.5` on code-specific queries and exporting it to a dynamically quantized INT8 ONNX model, we run real-time semantic code search entirely offline on a mobile device.

---

## Model Registry (Hugging Face)

The fine-tuned, dynamically quantized ONNX model weights and tokenizer configurations are hosted on Hugging Face:
👉 **[KingLLM/nomic-codesearch-onnx](https://huggingface.co/KingLLM/nomic-codesearch-onnx)**

---

## Repository Structure

- `android/` — The complete, ready-to-run Android Studio project.
- `android_src/` — Standalone Kotlin source code template files (MainActivity, OnnxEmbedder, BertTokenizer).
- `inference.py` — Python script to run local code search and generate embeddings using ONNX Runtime.
- `nb.ipynb` — Jupyter notebook outlining the model fine-tuning process (sentence-transformers training on MPS/cuda, MNR Loss, and dynamic quantization).

---

## Android App Setup Instructions

Since the quantized ONNX model file is ~100MB, it is excluded from this Git repository to keep cloning fast and lightweight. Follow these steps to build and run the app:

### 1. Download the Model weights
Go to the Hugging Face model repository: [KingLLM/nomic-codesearch-onnx](https://huggingface.co/KingLLM/nomic-codesearch-onnx) and download:
- `model_int8.onnx`

### 2. Add the Model to App Assets
1. Copy the downloaded `model_int8.onnx` into the Android project assets directory:
   `android/app/src/main/assets/`
2. Rename the file to `model.onnx`.

*(Note: The vocabulary file `vocab.txt` is already included in the assets folder).*

### 3. Open and Run in Android Studio
1. Launch Android Studio.
2. Select **Open an existing project** and choose the `android` folder in this repository.
3. Allow Gradle to sync and resolve dependencies.
4. Build and deploy to your Android device or emulator!

---

## Running Local Python Inference

To run semantic code search or generate embeddings locally using Python:

### 1. Install Dependencies
```bash
pip install onnxruntime transformers numpy
```

### 2. Run Inference
```bash
python inference.py
```
