package com.paddycare.ai.ui.components

import androidx.compose.ui.graphics.Color
import com.paddycare.ai.data.DiseaseInfo
import com.paddycare.ai.ui.theme.SeverityHigh
import com.paddycare.ai.ui.theme.SeverityHighBg
import com.paddycare.ai.ui.theme.SeverityLow
import com.paddycare.ai.ui.theme.SeverityLowBg
import com.paddycare.ai.ui.theme.SeverityMedium
import com.paddycare.ai.ui.theme.SeverityMediumBg

/** Colours and Bengali label for a disease's severity badge. */
data class SeverityStyle(val color: Color, val bg: Color, val label: String)

/**
 * Severity is a property of the disease (diseases.json), NOT of the model's confidence:
 * a confident "Healthy" prediction must never be shown as "Severe".
 */
fun severityStyle(disease: String?): SeverityStyle =
    when (DiseaseInfo.getSeverity(disease ?: "")) {
        "high"   -> SeverityStyle(SeverityHigh, SeverityHighBg, "মারাত্মক")
        "medium" -> SeverityStyle(SeverityMedium, SeverityMediumBg, "মাঝারি")
        "low"    -> SeverityStyle(SeverityLow, SeverityLowBg, "সামান্য")
        "none"   -> SeverityStyle(SeverityLow, SeverityLowBg, "সুস্থ")
        else     -> SeverityStyle(SeverityMedium, SeverityMediumBg, "অজানা")
    }
