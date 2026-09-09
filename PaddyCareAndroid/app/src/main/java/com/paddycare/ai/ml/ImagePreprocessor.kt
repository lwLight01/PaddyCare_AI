package com.paddycare.ai.ml

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs

/**
 * Paddy-leaf image validator.
 * Ports the Python check_paddy_image() heuristic:
 *   Stage 1 — HSV colour analysis (plant/green pixel ratios)
 *   Stage 2 — Edge/texture density (rejects plain/noisy images)
 */
object ImagePreprocessor {

    /**
     * Check if an image is a paddy (rice) leaf.
     * @return Pair(isPlant, debugStats)
     */
    fun checkPaddyImage(bitmap: Bitmap): Pair<Boolean, Map<String, Float>> {
        val small = Bitmap.createScaledBitmap(bitmap, 100, 100, true)
        val pixels = IntArray(100 * 100)
        small.getPixels(pixels, 0, 100, 0, 0, 100, 100)

        var plant = 0
        var green = 0
        val total = pixels.size

        for (pixel in pixels) {
            val r = Color.red(pixel) / 255f
            val g = Color.green(pixel) / 255f
            val b = Color.blue(pixel) / 255f

            val hsv = FloatArray(3)
            Color.RGBToHSV(Color.red(pixel), Color.green(pixel), Color.blue(pixel), hsv)
            val h = hsv[0]  // 0-360 degrees
            val s = hsv[1]  // 0-1
            val v = hsv[2]  // 0-1

            // Skip low saturation or very dark pixels
            if (s < 0.12f || v < 0.16f) continue

            // Plant colours: hue 28°–210° covers brown/yellow/green
            // (Python used PIL HSV 10-75 which maps to ~14°-106° in degrees,
            //  but Android HSV is 0-360 degrees)
            if (h in 14f..106f) plant++
            if (h in 34f..106f) green++
        }

        val plantRatio = plant.toFloat() / total
        val greenRatio = green.toFloat() / total

        // Stage 1 gate
        val colorOk = plantRatio >= 0.06f && greenRatio >= 0.03f

        // Stage 2 — edge density on 64×64 grayscale
        val gray = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        val grayPixels = IntArray(64 * 64)
        gray.getPixels(grayPixels, 0, 64, 0, 0, 64, 64)

        val luminance = FloatArray(64 * 64) { i ->
            val p = grayPixels[i]
            (Color.red(p) * 0.299f + Color.green(p) * 0.587f + Color.blue(p) * 0.114f)
        }

        // Horizontal gradient
        var gxSum = 0f
        var gxCount = 0
        for (y in 0 until 64) {
            for (x in 0 until 63) {
                gxSum += abs(luminance[y * 64 + x + 1] - luminance[y * 64 + x])
                gxCount++
            }
        }

        // Vertical gradient
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
        val textureOk = edgeRatio in 0.02f..0.65f

        val isPlant = colorOk && textureOk

        val stats = mapOf(
            "plant_ratio" to plantRatio,
            "green_ratio" to greenRatio,
            "edge_ratio" to edgeRatio,
        )

        return Pair(isPlant, stats)
    }
}
