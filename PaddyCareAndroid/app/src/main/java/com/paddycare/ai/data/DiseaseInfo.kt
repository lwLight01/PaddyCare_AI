package com.paddycare.ai.data

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * All disease information, treatments, and translations.
 * Reads dynamically from assets/diseases.json if available,
 * with fallback to built-in defaults.
 */
object DiseaseInfo {

    @Volatile
    private var initialized = false

    private val dynamicDiseaseBn = mutableMapOf<String, String>()
    private val dynamicTreatmentsBn = mutableMapOf<String, String>()
    private val dynamicTreatmentsEn = mutableMapOf<String, String>()
    private val dynamicRejectClasses = mutableSetOf<String>()

    /** Built-in fallback Bengali display names */
    val DISEASE_BN = mapOf(
        "Bacterial Blight" to "ব্যাকটেরিয়াল ব্লাইট",
        "Blast" to "ব্লাস্ট রোগ",
        "Brown Spot" to "বাদামী দাগ",
        "Healthy" to "সুস্থ গাছ",
        "Not Leaf" to "ধানের পাতা নয়",
    )

    /** Built-in fallback Treatments in Bengali */
    val TREATMENTS_BN = mapOf(
        "Brown Spot" to """
            ১. ম্যানকোজেব বা প্রোপিকোনাজোল ছত্রাকনাশক স্প্রে করুন।
            ২. সুষম পটাশিয়াম ও সিলিকন সার প্রয়োগ করুন।
            ৩. কুশি পর্যায়ে পানির চাপ এড়িয়ে চলুন।
            ৪. প্রতিরোধী জাত ব্যবহার করুন।
            ৫. আক্রান্ত গাছের অবশিষ্ট পুড়িয়ে ফেলুন।
        """.trimIndent(),
        "Blast" to """
            ১. ট্রাইসাইক্লাজোল (০.১%) বা আইসোপ্রোথিওলেন ছত্রাকনাশক স্প্রে করুন।
            ২. সিলিকন-ভিত্তিক সার প্রয়োগ করুন।
            ৩. অতিরিক্ত নাইট্রোজেন সার এড়িয়ে চলুন।
            ৪. মাঠের পানি মাঝে মাঝে নিষ্কাশন করুন।
            ৫. ব্লাস্ট-প্রতিরোধী ধানের জাত ব্যবহার করুন।
        """.trimIndent(),
        "Bacterial Blight" to """
            ১. কপার অক্সিক্লোরাইড বা স্ট্রেপটোমাইসিন দ্রবণ স্প্রে করুন।
            ২. মাঠ থেকে অতিরিক্ত পানি দ্রুত নিষ্কাশন করুন।
            ৩. অতিরিক্ত নাইট্রোজেন সার এড়িয়ে চলুন।
            ৪. প্রতিরোধী ধানের জাত (যেমন IR64) ব্যবহার করুন।
            ৫. আক্রান্ত গাছ তুলে নষ্ট করুন।
        """.trimIndent(),
        "Healthy" to """
            আপনার ধান গাছ সুস্থ আছে!
            ১. নিয়মিত সেচ ও সার দেওয়া অব্যাহত রাখুন।
            ২. প্রতি সপ্তাহে রোগের প্রাথমিক লক্ষণ পর্যবেক্ষণ করুন।
            ৩. মাঠ পরিষ্কার-পরিচ্ছন্ন রাখুন।
            ৪. আশেপাশে রোগ থাকলে প্রতিরোধমূলক ছত্রাকনাশক দিন।
        """.trimIndent(),
        "Not Leaf" to """
            এই ছবিটি ধান (চাল) পাতার নয়।
            PaddyCare শুধুমাত্র ধান পাতার ছবি থেকে রোগ নির্ণয় করতে পারে।
            অনুগ্রহ করে ধান পাতার একটি স্পষ্ট ও কাছের ছবি আপলোড করুন।
        """.trimIndent(),
    )

    /** Built-in fallback Treatments in English */
    val TREATMENTS_EN = mapOf(
        "Brown Spot" to """
            1. Spray Mancozeb or Propiconazole fungicide.
            2. Apply balanced potassium and silicon fertilizer.
            3. Avoid water stress during the tillering stage.
            4. Use resistant varieties.
            5. Remove and burn infected plant debris.
        """.trimIndent(),
        "Blast" to """
            1. Spray Tricyclazole (0.1%) or Isoprothiolane fungicide.
            2. Apply silicon-based fertilizer to boost resistance.
            3. Avoid excess nitrogen fertilizer.
            4. Drain field water periodically.
            5. Use blast-resistant paddy varieties.
        """.trimIndent(),
        "Bacterial Blight" to """
            1. Spray Copper Oxychloride or Streptomycin solution.
            2. Drain excess water from the field immediately.
            3. Avoid high nitrogen fertilizer application.
            4. Use resistant paddy varieties (e.g., IR64).
            5. Remove and destroy infected plants.
        """.trimIndent(),
        "Healthy" to """
            Your paddy plant is healthy!
            1. Continue regular irrigation and fertilization.
            2. Monitor weekly for any early disease signs.
            3. Maintain proper field hygiene.
            4. Apply preventive fungicide if neighbors have infections.
        """.trimIndent(),
        "Not Leaf" to """
            This image is not a paddy (rice) leaf.
            PaddyCare can only diagnose diseases from paddy leaf photos.
            Please upload a clear, close-up photo of a paddy leaf.
        """.trimIndent(),
    )

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
                }
                initialized = true
            } catch (e: Exception) {
                Log.w("DiseaseInfo", "Could not load diseases.json from assets, using built-in defaults: ${e.message}")
            }
        }
    }

    fun isRejectClass(className: String): Boolean {
        return dynamicRejectClasses.contains(className) ||
               className.equals("Not Leaf", ignoreCase = true)
    }

    fun getBengaliName(englishName: String): String =
        dynamicDiseaseBn[englishName] ?: DISEASE_BN[englishName] ?: englishName

    fun getTreatment(disease: String, lang: String = "bn"): String {
        return when (lang) {
            "en" -> dynamicTreatmentsEn[disease] ?: TREATMENTS_EN[disease] ?: ""
            else -> dynamicTreatmentsBn[disease] ?: TREATMENTS_BN[disease] ?: ""
        }
    }
}
