package com.paddycare.ai.data

/**
 * All disease information, treatments, and Bengali translations.
 * Mirrors the Python config.py TREATMENTS dictionary exactly.
 */
object DiseaseInfo {

    /** Bengali display names for each disease class */
    val DISEASE_BN = mapOf(
        "Bacterial Blight" to "ব্যাকটেরিয়াল ব্লাইট",
        "Blast" to "ব্লাস্ট রোগ",
        "Healthy" to "সুস্থ গাছ",
        "Not Leaf" to "ধানের পাতা নয়",
    )

    /** Treatments in Bengali */
    val TREATMENTS_BN = mapOf(
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

    /** Treatments in English */
    val TREATMENTS_EN = mapOf(
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

    fun getBengaliName(englishName: String): String =
        DISEASE_BN[englishName] ?: englishName

    fun getTreatment(disease: String, lang: String = "bn"): String {
        return when (lang) {
            "en" -> TREATMENTS_EN[disease] ?: ""
            else -> TREATMENTS_BN[disease] ?: ""
        }
    }
}
