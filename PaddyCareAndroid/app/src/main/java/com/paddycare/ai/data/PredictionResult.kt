package com.paddycare.ai.data

/**
 * Data classes for prediction results — used across UI screens.
 */
data class Prediction(
    val disease: String,
    val confidence: Float,
    val treatment: String,
    val diseaseBn: String = DiseaseInfo.getBengaliName(disease),
)

sealed class PredictionResult {
    data class Success(val predictions: List<Prediction>) : PredictionResult()
    data class NotPaddy(val message: String) : PredictionResult()
    data class LowConfidence(val message: String) : PredictionResult()
    data class Error(val message: String) : PredictionResult()
}
