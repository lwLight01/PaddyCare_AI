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
 * Supports dual-outputs (logits + 128-dim embedding) for Mahalanobis OOD detection.
 *
 * Thread-safety: [classify] is synchronized because the TFLite [Interpreter] and the
 * reusable input/output buffers must never be used by two callers at once.
 *
 * Usage:
 *   val classifier = PaddyClassifier(context)
 *   val result = classifier.classify(bitmap)
 *   classifier.close()
 *
 * @throws IllegalStateException if model.tflite, labels.txt and model_meta.json do not belong together.
 */
class PaddyClassifier(context: Context) {

    companion object {
        private const val TAG = "PaddyClassifier"
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

    // Export-time identity of the model, read from model_meta.json (null for older exports).
    private var metaFingerprint: String? = null
    private var metaLogitsIndex: Int? = null
    private var metaEmbeddingIndex: Int? = null

    // Mahalanobis statistics (Double: the precision matrix can be badly conditioned)
    private var classMeans = mapOf<String, DoubleArray>()
    private var precisionMatrix = Array(EMBEDDING_DIM) { DoubleArray(EMBEDDING_DIM) }
    private var mahalanobisThreshold = Double.MAX_VALUE
    private var statsLoaded = false

    // Resolved output layout + buffers reused across calls (guarded by `this` lock in classify()).
    private val logitsIndex: Int
    private val embedIndex: Int
    private val inputBuffer: ByteBuffer
    private val pixelBuffer: IntArray
    private val outputs: MutableMap<Int, Any>
    private val logitsOut: Array<FloatArray>
    private val embedOut: Array<FloatArray>

    init {
        val model = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)
        labels = loadLabels(context)
        loadModelMeta(context)

        // Resolve which output tensor holds the logits and which the embedding.
        val shapes = (0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it).shape() }
        val metaLi = metaLogitsIndex
        val metaEi = metaEmbeddingIndex
        if (metaLi != null && metaEi != null && metaLi != metaEi &&
            metaLi in shapes.indices && metaEi in shapes.indices
        ) {
            logitsIndex = metaLi
            embedIndex = metaEi
        } else {
            // Older exports: fall back to telling them apart by width.
            val li = shapes.indexOfFirst { it.size == 2 && it[1] != EMBEDDING_DIM }
            val ei = shapes.indexOfFirst { it.size == 2 && it[1] == EMBEDDING_DIM }
            if (li < 0 || ei < 0) {
                interpreter.close()
                throw IllegalStateException("Unexpected TFLite output shapes: ${shapes.map { it.toList() }}")
            }
            logitsIndex = li
            embedIndex = ei
        }

        val logitsSize = shapes[logitsIndex].last()
        if (logitsSize != labels.size) {
            interpreter.close()
            throw IllegalStateException(
                "Model has $logitsSize classes but labels.txt has ${labels.size}. " +
                    "Re-export model.tflite and labels.txt together with export_tflite.py."
            )
        }
        if (shapes[embedIndex].last() != EMBEDDING_DIM) {
            interpreter.close()
            throw IllegalStateException("Embedding output is ${shapes[embedIndex].last()}-dim, expected $EMBEDDING_DIM.")
        }

        logitsOut = Array(1) { FloatArray(logitsSize) }
        embedOut = Array(1) { FloatArray(EMBEDDING_DIM) }
        outputs = mutableMapOf<Int, Any>(logitsIndex to logitsOut, embedIndex to embedOut)

        inputBuffer = ByteBuffer.allocateDirect(imageSize * imageSize * PIXEL_SIZE * BYTES_PER_FLOAT)
            .order(ByteOrder.nativeOrder())
        pixelBuffer = IntArray(imageSize * imageSize)

        loadClassStats(context)
    }

    /**
     * Classify a bitmap image.
     * @return ClassifyResult containing predictions (sorted) and the 128-dim embedding.
     */
    @Synchronized
    fun classify(bitmap: Bitmap): ClassifyResult {
        // Match Python pipeline: PadToSquare() → Resize(imageSize) → Normalize
        val squared = padToSquare(bitmap)
        val resized = Bitmap.createScaledBitmap(squared, imageSize, imageSize, true)
        fillInputBuffer(resized)

        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputs)

        val probabilities = softmax(logitsOut[0])
        val predictions = labels.indices
            .map { i -> Pair(labels[i], probabilities[i]) }
            .sortedByDescending { it.second }

        // Copy: embedOut is reused by the next call.
        return ClassifyResult(predictions, embedOut[0].copyOf())
    }

    /**
     * Checks if the embedding is Out-Of-Distribution (OOD) for the predicted class
     * using the Mahalanobis distance. Always false when no matching stats are loaded.
     */
    fun isOOD(embedding: FloatArray, predictedClass: String): Boolean {
        if (!statsLoaded) return false
        val mean = classMeans[predictedClass] ?: return false
        return mahalanobisDistance(embedding, mean) > mahalanobisThreshold
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

    @Synchronized
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
            metaFingerprint = json.optString("checkpoint_fingerprint", "").ifEmpty { null }
            if (json.has("logits_output_index")) metaLogitsIndex = json.getInt("logits_output_index")
            if (json.has("embedding_output_index")) metaEmbeddingIndex = json.getInt("embedding_output_index")

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
            Log.i(TAG, "Loaded model_meta.json (imageSize=$imageSize, confThreshold=$confidenceThreshold)")
        } catch (e: Exception) {
            Log.d(TAG, "Using default model metadata: ${e.message}")
        }
        entropyThreshold = (entropyFactor * kotlin.math.ln(labels.size.coerceAtLeast(2).toDouble())).toFloat()
    }

    /**
     * Loads Mahalanobis stats. Any inconsistency (wrong model, wrong dimensions) disables OOD
     * detection instead of producing meaningless distances.
     */
    private fun loadClassStats(context: Context) {
        statsLoaded = false
        try {
            val jsonStr = context.assets.open(STATS_FILE).bufferedReader().use { it.readText() }
            val json = JSONObject(jsonStr)

            val statsFingerprint = json.optString("checkpoint_fingerprint", "").ifEmpty { null }
            if (metaFingerprint != null && statsFingerprint != metaFingerprint) {
                Log.w(TAG, "class_stats.json ($statsFingerprint) does not match model ($metaFingerprint). " +
                    "Mahalanobis OOD disabled.")
                return
            }
            if (json.optInt("embedding_dim", EMBEDDING_DIM) != EMBEDDING_DIM) {
                Log.w(TAG, "class_stats.json embedding_dim mismatch. Mahalanobis OOD disabled.")
                return
            }

            val threshold = json.getDouble("threshold")

            val meansObj = json.getJSONObject("class_means")
            val meansMap = mutableMapOf<String, DoubleArray>()
            meansObj.keys().forEach { key ->
                val arr = meansObj.getJSONArray(key)
                require(arr.length() == EMBEDDING_DIM) { "class mean '$key' has ${arr.length()} values" }
                meansMap[key] = DoubleArray(EMBEDDING_DIM) { i -> arr.getDouble(i) }
            }

            val precJson = json.getJSONArray("precision_matrix")
            require(precJson.length() == EMBEDDING_DIM) { "precision matrix has ${precJson.length()} rows" }
            val precision = Array(EMBEDDING_DIM) { i ->
                val row = precJson.getJSONArray(i)
                require(row.length() == EMBEDDING_DIM) { "precision row $i has ${row.length()} values" }
                DoubleArray(EMBEDDING_DIM) { j -> row.getDouble(j) }
            }

            mahalanobisThreshold = threshold
            classMeans = meansMap
            precisionMatrix = precision
            statsLoaded = true
        } catch (e: Exception) {
            Log.w(TAG, "Could not load class_stats.json. Mahalanobis OOD disabled.", e)
            statsLoaded = false
        }
    }

    private fun mahalanobisDistance(embedding: FloatArray, mean: DoubleArray): Double {
        // diff = embedding - mean
        val diff = DoubleArray(EMBEDDING_DIM) { i -> embedding[i].toDouble() - mean[i] }

        // dist = sqrt(diff^T * precision * diff)
        var sum = 0.0
        for (i in 0 until EMBEDDING_DIM) {
            var rowSum = 0.0
            for (j in 0 until EMBEDDING_DIM) {
                rowSum += precisionMatrix[i][j] * diff[j]
            }
            sum += diff[i] * rowSum
        }
        return if (sum > 0) sqrt(sum) else 0.0
    }

    /**
     * Fill the reusable input buffer: normalize with model mean/std (NHWC float32).
     * Matches the Python inference transform:
     *   PadToSquare() → Resize(imageSize) → Normalize(mean, std)
     */
    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        bitmap.getPixels(pixelBuffer, 0, imageSize, 0, 0, imageSize, imageSize)

        for (pixel in pixelBuffer) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            inputBuffer.putFloat((r - mean[0]) / std[0])
            inputBuffer.putFloat((g - mean[1]) / std[1])
            inputBuffer.putFloat((b - mean[2]) / std[2])
        }
        inputBuffer.rewind()
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    /**
     * Pad the shorter side of the bitmap with black pixels to make it square,
     * preserving the original aspect ratio.  Mirrors PadToSquare() in Python
     * (integer pixel offsets, so no edge blending).
     */
    private fun padToSquare(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w == h) return bitmap

        val maxSide = maxOf(w, h)
        val result = Bitmap.createBitmap(maxSide, maxSide, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.BLACK)
        val left = ((maxSide - w) / 2).toFloat()
        val top = ((maxSide - h) / 2).toFloat()
        canvas.drawBitmap(bitmap, left, top, null)
        return result
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        context.assets.openFd(MODEL_FILE).use { assetFd ->
            FileInputStream(assetFd.fileDescriptor).use { inputStream ->
                // The mapping stays valid after the descriptor and stream are closed.
                return inputStream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    assetFd.startOffset,
                    assetFd.declaredLength
                )
            }
        }
    }

    private fun loadLabels(context: Context): List<String> {
        return context.assets.open(LABELS_FILE)
            .bufferedReader()
            .use { it.readLines() }
            .map { it.trim() }
            .filter { it.isNotEmpty() }
    }
}
