package com.paddycare.ai.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.json.JSONObject
import kotlin.math.abs

/**
 * Paddy-leaf pre-screening heuristic:
 *   Stage 1 — HSV colour analysis (plant hue 14°–106° covering yellow/brown/green, green hue 34°–106°)
 *   Stage 2 — Edge/texture density (rejects plain solid backgrounds or noisy images)
 *
 * Thresholds default to config.py:LEAF_CHECK and are overridden from assets/model_meta.json["leaf_check"].
 */
object ImagePreprocessor {

    @Volatile private var minSaturation = 0.12f
    @Volatile private var minValue      = 0.16f
    @Volatile private var plantHueMin   = 14.0f
    @Volatile private var plantHueMax   = 106.0f
    @Volatile private var greenHueMin   = 34.0f
    @Volatile private var greenHueMax   = 106.0f
    @Volatile private var minPlantRatio = 0.06f
    @Volatile private var minGreenRatio = 0.03f
    @Volatile private var minEdgeRatio  = 0.02f
    @Volatile private var maxEdgeRatio  = 0.65f

    fun init(context: Context) {
        try {
            val jsonStr = context.assets.open("model_meta.json").bufferedReader().use { it.readText() }
            val lc = JSONObject(jsonStr).optJSONObject("leaf_check") ?: return
            minSaturation = lc.optDouble("min_saturation", minSaturation.toDouble()).toFloat()
            minValue      = lc.optDouble("min_value", minValue.toDouble()).toFloat()
            plantHueMin   = lc.optDouble("plant_hue_min", plantHueMin.toDouble()).toFloat()
            plantHueMax   = lc.optDouble("plant_hue_max", plantHueMax.toDouble()).toFloat()
            greenHueMin   = lc.optDouble("green_hue_min", greenHueMin.toDouble()).toFloat()
            greenHueMax   = lc.optDouble("green_hue_max", greenHueMax.toDouble()).toFloat()
            minPlantRatio = lc.optDouble("min_plant_ratio", minPlantRatio.toDouble()).toFloat()
            minGreenRatio = lc.optDouble("min_green_ratio", minGreenRatio.toDouble()).toFloat()
            minEdgeRatio  = lc.optDouble("min_edge_ratio", minEdgeRatio.toDouble()).toFloat()
            maxEdgeRatio  = lc.optDouble("max_edge_ratio", maxEdgeRatio.toDouble()).toFloat()
        } catch (e: Exception) {
            Log.d("ImagePreprocessor", "Using default leaf_check thresholds: ${e.message}")
        }
    }

    /**
     * Check if an image plausibly contains a paddy (rice) leaf.
     * @return Pair(isPlant, debugStats)
     */
    fun checkPaddyImage(bitmap: Bitmap): Pair<Boolean, Map<String, Float>> {
        val small = Bitmap.createScaledBitmap(bitmap, 100, 100, true)
        val pixels = IntArray(100 * 100)
        small.getPixels(pixels, 0, 100, 0, 0, 100, 100)

        var plant = 0
        var green = 0
        val total = pixels.size

        val hsv = FloatArray(3)
        for (pixel in pixels) {
            Color.RGBToHSV(Color.red(pixel), Color.green(pixel), Color.blue(pixel), hsv)
            val h = hsv[0]  // 0–360 degrees
            val s = hsv[1]  // 0–1
            val v = hsv[2]  // 0–1

            if (s < minSaturation || v < minValue) continue

            // Plant colours in Android HSV degrees (0°–360°):
            // 14°–106° covers brown/yellow/necrotic lesions through healthy green; 34°–106° is green.
            if (h in plantHueMin..plantHueMax) plant++
            if (h in greenHueMin..greenHueMax) green++
        }

        val plantRatio = plant.toFloat() / total
        val greenRatio = green.toFloat() / total

        val colorOk = plantRatio >= minPlantRatio && greenRatio >= minGreenRatio

        // Stage 2 — edge density on 64×64 grayscale
        val gray = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        val grayPixels = IntArray(64 * 64)
        gray.getPixels(grayPixels, 0, 64, 0, 0, 64, 64)

        val luminance = FloatArray(64 * 64) { i ->
            val p = grayPixels[i]
            (Color.red(p) * 0.299f + Color.green(p) * 0.587f + Color.blue(p) * 0.114f)
        }

        var gxSum = 0f
        var gxCount = 0
        for (y in 0 until 64) {
            for (x in 0 until 63) {
                gxSum += abs(luminance[y * 64 + x + 1] - luminance[y * 64 + x])
                gxCount++
            }
        }

        var gySum = 0f
        var gyCount = 0
        for (y in 0 until 63) {
            for (x in 0 until 64) {
                gySum += abs(luminance[(y + 1) * 64 + x] - luminance[y * 64 + x])
                gyCount++
            }
        }

        val edgeMean = ((gxSum / gxCount) + (gySum / gyCount)) / 2f
        val edgeRatio = edgeMean / 255f
        val textureOk = edgeRatio in minEdgeRatio..maxEdgeRatio

        val isPlant = colorOk && textureOk

        val stats = mapOf(
            "plant_ratio" to plantRatio,
            "green_ratio" to greenRatio,
            "edge_ratio" to edgeRatio,
        )

        return Pair(isPlant, stats)
    }
}
