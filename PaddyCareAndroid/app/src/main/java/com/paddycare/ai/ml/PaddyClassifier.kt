package com.paddycare.ai.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.sqrt

data class ClassifyResult(
    val predictions: List<Pair<String, Float>>,
    val embedding: FloatArray
)

/**
 * PaddyClassifier — runs TFLite EfficientNet-B5 inference on-device.
 * Now supports dual-outputs (logits + 128-dim embedding) for Mahalanobis OOD detection.
 *
 * Usage:
 *   val classifier = PaddyClassifier(context)
 *   val result = classifier.classify(bitmap)
 *   classifier.close()
 */
class PaddyClassifier(context: Context) {

    companion object {
        private const val MODEL_FILE = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val STATS_FILE = "class_stats.json"
        private const val META_FILE  = "model_meta.json"
        const val IMAGE_SIZE = 456  // Fallback default
        private const val PIXEL_SIZE = 3 // RGB
        private const val BYTES_PER_FLOAT = 4
        private const val EMBEDDING_DIM = 128

        // Fallback ImageNet normalization values
        private val DEFAULT_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val DEFAULT_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

        const val CONFIDENCE_THRESHOLD = 0.60f  // 60% fallback
    }

    private val interpreter: Interpreter
    val labels: List<String>

    var imageSize: Int = IMAGE_SIZE
        private set
    var mean: FloatArray = DEFAULT_MEAN
        private set
    var std: FloatArray = DEFAULT_STD
        private set
    var confidenceThreshold: Float = CONFIDENCE_THRESHOLD
        private set
    var entropyThreshold: Float = 0f
        private set

    // Mahalanobis statistics
    private var classMeans = mapOf<String, FloatArray>()
    private var precisionMatrix = Array(EMBEDDING_DIM) { FloatArray(EMBEDDING_DIM) }
    private var mahalanobisThreshold = Float.MAX_VALUE
    private var statsLoaded = false

    init {
        val model = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)
        labels = loadLabels(context)
        loadModelMeta(context)
        loadClassStats(context)

        // Verify that label count matches the model's logits output tensor
        val outputCount = interpreter.outputTensorCount
        for (i in 0 until outputCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            if (shape.size == 2 && shape[1] != EMBEDDING_DIM) {
                if (shape[1] != labels.size) {
                    Log.w(
                        "PaddyClassifier",
                        "⚠️ Model output classes (${shape[1]}) != labels.txt size (${labels.size}). " +
                        "Please re-export model.tflite and labels.txt together."
                    )
                }
            }
        }
    }

    /**
     * Classify a bitmap image.
     * @return ClassifyResult containing predictions (sorted) and the 128-dim embedding.
     */
    fun classify(bitmap: Bitmap): ClassifyResult {
        // Match Python pipeline: PadToSquare() → Resize(imageSize) → Normalize
        val squared = padToSquare(bitmap)
        val resized = Bitmap.createScaledBitmap(squared, imageSize, imageSize, true)
        val inputBuffer = preprocessImage(resized)

        // The TFLite model has two outputs: logits and embedding.
        val outputCount = interpreter.outputTensorCount
        val outputs = mutableMapOf<Int, Any>()

        var logitsIndex = 0
        var embedIndex = 1
        var modelLogitsSize = labels.size

        for (i in 0 until outputCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            if (shape.size == 2 && shape[1] != EMBEDDING_DIM) {
                logitsIndex = i
                modelLogitsSize = shape[1]
                outputs[i] = Array(1) { FloatArray(shape[1]) }
            } else if (shape.size == 2 && shape[1] == EMBEDDING_DIM) {
                embedIndex = i
                outputs[i] = Array(1) { FloatArray(EMBEDDING_DIM) }
            }
        }

        val inputs = arrayOf(inputBuffer)
        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        @Suppress("UNCHECKED_CAST")
        val logitsArray = (outputs[logitsIndex] as? Array<FloatArray>)
            ?: throw IllegalStateException("Failed to extract logits tensor from model output")
        @Suppress("UNCHECKED_CAST")
        val embedArray = (outputs[embedIndex] as? Array<FloatArray>)
            ?: Array(1) { FloatArray(EMBEDDING_DIM) }

        val logits = logitsArray[0]
        val embedding = embedArray[0]

        val probabilities = softmax(logits)

        // Pair predictions up to the minimum of available labels and logits
        val pairCount = minOf(labels.size, probabilities.size)
        val predictions = (0 until pairCount).map { i ->
            Pair(labels[i], probabilities[i])
        }.sortedByDescending { it.second }

        return ClassifyResult(predictions, embedding)
    }

    /**
     * Checks if the embedding is Out-Of-Distribution (OOD) for the predicted class
     * using the Mahalanobis distance.
     */
    fun isOOD(embedding: FloatArray, predictedClass: String): Boolean {
        if (!statsLoaded) return false
        val mean = classMeans[predictedClass] ?: return false
        val dist = mahalanobisDistance(embedding, mean)
        return dist > mahalanobisThreshold
    }

    /**
     * Get the Shannon entropy of a probability distribution.
     * High entropy = model is uncertain (possible out-of-distribution image).
     */
    fun entropy(probs: List<Float>): Float {
        return -probs.sumOf { p ->
            if (p > 1e-9f) (p * kotlin.math.ln(p.toDouble())) else 0.0
        }.toFloat()
    }

    fun close() {
        interpreter.close()
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    private fun loadModelMeta(context: Context) {
        var entropyFactor = 0.75
        try {
            val jsonStr = context.assets.open(META_FILE).bufferedReader().use { it.readText() }
            val json = JSONObject(jsonStr)
            imageSize = json.optInt("image_size", IMAGE_SIZE)
            confidenceThreshold = json.optDouble("confidence_threshold", CONFIDENCE_THRESHOLD.toDouble()).toFloat()
            entropyFactor = json.optDouble("entropy_factor", 0.75)

            val meanArr = json.optJSONArray("mean")
            if (meanArr != null && meanArr.length() == 3) {
                mean = floatArrayOf(
                    meanArr.getDouble(0).toFloat(),
                    meanArr.getDouble(1).toFloat(),
                    meanArr.getDouble(2).toFloat()
                )
            }
            val stdArr = json.optJSONArray("std")
            if (stdArr != null && stdArr.length() == 3) {
                std = floatArrayOf(
                    stdArr.getDouble(0).toFloat(),
                    stdArr.getDouble(1).toFloat(),
                    stdArr.getDouble(2).toFloat()
                )
            }
            Log.i("PaddyClassifier", "Loaded model_meta.json (imageSize=$imageSize, confThreshold=$confidenceThreshold)")
        } catch (e: Exception) {
            Log.d("PaddyClassifier", "Using default model metadata: ${e.message}")
        }
        entropyThreshold = (entropyFactor * kotlin.math.ln(labels.size.coerceAtLeast(2).toDouble())).toFloat()
    }

    private fun loadClassStats(context: Context) {
        try {
            val jsonStr = context.assets.open(STATS_FILE).bufferedReader().use { it.readText() }
            val json = JSONObject(jsonStr)
            
            mahalanobisThreshold = json.getDouble("threshold").toFloat()
            
            val meansObj = json.getJSONObject("class_means")
            val meansMap = mutableMapOf<String, FloatArray>()
            meansObj.keys().forEach { key ->
                val arr = meansObj.getJSONArray(key)
                val floatArr = FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
                meansMap[key] = floatArr
            }
            classMeans = meansMap
            
            val precJson = json.getJSONArray("precision_matrix")
            for (i in 0 until EMBEDDING_DIM) {
                val row = precJson.getJSONArray(i)
                for (j in 0 until EMBEDDING_DIM) {
                    precisionMatrix[i][j] = row.getDouble(j).toFloat()
                }
            }
            statsLoaded = true
        } catch (e: Exception) {
            Log.w("PaddyClassifier", "Could not load class_stats.json. Mahalanobis OOD disabled.", e)
            statsLoaded = false
        }
    }

    private fun mahalanobisDistance(embedding: FloatArray, mean: FloatArray): Float {
        // diff = embedding - mean
        val diff = FloatArray(EMBEDDING_DIM) { i -> embedding[i] - mean[i] }
        
        // dist = sqrt(diff^T * precision * diff)
        var sum = 0f
        for (i in 0 until EMBEDDING_DIM) {
            var rowSum = 0f
            for (j in 0 until EMBEDDING_DIM) {
                rowSum += precisionMatrix[i][j] * diff[j]
            }
            sum += diff[i] * rowSum
        }
        return if (sum > 0) sqrt(sum) else 0f
    }

    /**
     * Preprocess image: padToSquare → resize → normalize with ImageNet mean/std → ByteBuffer.
     * Matches the Python inference transform exactly:
     *   PadToSquare() → Resize(456) → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * imageSize * imageSize * PIXEL_SIZE * BYTES_PER_FLOAT)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(imageSize * imageSize)
        bitmap.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Normalize with model statistics
            buffer.putFloat((r - mean[0]) / std[0])
            buffer.putFloat((g - mean[1]) / std[1])
            buffer.putFloat((b - mean[2]) / std[2])
        }

        buffer.rewind()
        return buffer
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    /**
     * Pad the shorter side of the bitmap with black pixels to make it square,
     * preserving the original aspect ratio.  Mirrors PadToSquare() in Python.
     */
    private fun padToSquare(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w == h) return bitmap

        val maxSide = maxOf(w, h)
        val result = Bitmap.createBitmap(maxSide, maxSide, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.BLACK)
        val left = (maxSide - w) / 2f
        val top = (maxSide - h) / 2f
        canvas.drawBitmap(bitmap, left, top, null)
        return result
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val assetFd = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }

    private fun loadLabels(context: Context): List<String> {
        return context.assets.open(LABELS_FILE)
            .bufferedReader()
            .readLines()
            .filter { it.isNotBlank() }
    }
}
