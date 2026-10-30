package com.paddycare.ai.data

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * Disease metadata, treatments, severity, and translations loaded from assets/diseases.json
 * (synced from the root diseases.json by export_tflite.py).
 */
object DiseaseInfo {

    @Volatile
    private var initialized = false

    private val dynamicDiseaseBn = mutableMapOf<String, String>()
    private val dynamicTreatmentsBn = mutableMapOf<String, String>()
    private val dynamicTreatmentsEn = mutableMapOf<String, String>()
    private val dynamicRejectClasses = mutableSetOf<String>()
    private val dynamicSeverity = mutableMapOf<String, String>()

    fun init(context: Context) {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            try {
                val jsonString = context.assets.open("diseases.json").bufferedReader().use { it.readText() }
                val json = JSONObject(jsonString)
                json.keys().forEach { key ->
                    val obj = json.getJSONObject(key)
                    val nameBn = obj.optString("name_bn", "")
                    val treatmentEn = obj.optString("treatment_en", "")
                    val treatmentBn = obj.optString("treatment_bn", "")
                    val isReject = obj.optBoolean("is_reject", false)

                    if (nameBn.isNotBlank()) dynamicDiseaseBn[key] = nameBn
                    if (treatmentEn.isNotBlank()) dynamicTreatmentsEn[key] = treatmentEn
                    if (treatmentBn.isNotBlank()) dynamicTreatmentsBn[key] = treatmentBn
                    if (isReject) dynamicRejectClasses.add(key)
                    obj.optString("severity", "").takeIf { it.isNotBlank() }?.let { dynamicSeverity[key] = it }
                }
                initialized = true
            } catch (e: Exception) {
                Log.e("DiseaseInfo", "Failed to load diseases.json from assets: ${e.message}")
            }
        }
    }

    fun isRejectClass(className: String): Boolean {
        return dynamicRejectClasses.contains(className) ||
               className.equals("Not Leaf", ignoreCase = true)
    }

    fun getBengaliName(englishName: String): String =
        dynamicDiseaseBn[englishName] ?: englishName

    /** Static severity of a disease: "high", "medium", "low", "none" or "unknown". Not related to model confidence. */
    fun getSeverity(disease: String): String =
        dynamicSeverity[disease] ?: if (disease == "Healthy") "none" else "unknown"

    fun getTreatment(disease: String, lang: String = "bn"): String {
        return when (lang) {
            "en" -> dynamicTreatmentsEn[disease] ?: ""
            else -> dynamicTreatmentsBn[disease] ?: ""
        }
    }
}
