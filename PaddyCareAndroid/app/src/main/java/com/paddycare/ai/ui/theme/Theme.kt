package com.paddycare.ai.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ── Local theme accessor ──────────────────────────────────────────────────────
data class PaddyCareColors(
    val bgPrimary: Color,
    val bgCard: Color,
    val bgCardAlt: Color,
    val bgTertiary: Color,
    val greenPrimary: Color,
    val greenDark: Color,
    val greenPale: Color,
    val greenUltra: Color,
    val textPrimary: Color,
    val textSecondary: Color,
    val textMuted: Color,
    val borderLight: Color,
    val borderMid: Color,
    val divider: Color,
    val sevHighBg: Color,
    val sevHighText: Color,
    val sevMedBg: Color,
    val sevMedText: Color,
    val sevLowBg: Color,
    val sevLowText: Color,
    val isDark: Boolean
)

val LocalPaddyCareColors = compositionLocalOf {
    PaddyCareColors(
        bgPrimary = BgPrimary, bgCard = BgCard, bgCardAlt = BgCardAlt,
        bgTertiary = BgTertiary, greenPrimary = GreenPrimary, greenDark = GreenDark,
        greenPale = GreenPale, greenUltra = GreenUltra,
        textPrimary = TextPrimary, textSecondary = TextSecondary, textMuted = TextMuted,
        borderLight = BorderLight, borderMid = BorderMid, divider = DividerColor,
        sevHighBg = SeverityHighBg, sevHighText = SeverityHighText,
        sevMedBg = SeverityMediumBg, sevMedText = SeverityMediumText,
        sevLowBg = SeverityLowBg, sevLowText = SeverityLowText,
        isDark = false
    )
}

private val LightColorScheme = lightColorScheme(
    primary          = GreenPrimary,
    secondary        = GreenMid,
    tertiary         = GreenLight,
    background       = BgPrimary,
    surface          = BgCard,
    surfaceVariant   = BgCardAlt,
    onPrimary        = TextOnGreen,
    onSecondary      = TextOnGreen,
    onTertiary       = TextOnGreen,
    onBackground     = TextPrimary,
    onSurface        = TextPrimary,
    onSurfaceVariant = TextSecondary,
    outline          = BorderLight,
)

private val DarkColorScheme = darkColorScheme(
    primary          = DarkGreenPrimary,
    secondary        = GreenMid,
    tertiary         = GreenLight,
    background       = DarkBgPrimary,
    surface          = DarkBgCard,
    surfaceVariant   = DarkBgCardAlt,
    onPrimary        = TextOnGreen,
    onSecondary      = TextOnGreen,
    onTertiary       = TextOnGreen,
    onBackground     = DarkTextPrimary,
    onSurface        = DarkTextPrimary,
    onSurfaceVariant = DarkTextSecondary,
    outline          = DarkBorderLight,
)

@Composable
fun PaddyCareTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    val paddyCareColors = if (darkTheme) {
        PaddyCareColors(
            bgPrimary = DarkBgPrimary, bgCard = DarkBgCard, bgCardAlt = DarkBgCardAlt,
            bgTertiary = DarkBgTertiary, greenPrimary = DarkGreenPrimary, greenDark = DarkGreenPrimary,
            greenPale = DarkGreenPale, greenUltra = DarkGreenUltra,
            textPrimary = DarkTextPrimary, textSecondary = DarkTextSecondary, textMuted = DarkTextMuted,
            borderLight = DarkBorderLight, borderMid = DarkBorderMid, divider = DarkDivider,
            sevHighBg = DarkSevHighBg, sevHighText = DarkSevHighText,
            sevMedBg = DarkSevMedBg, sevMedText = DarkSevMedText,
            sevLowBg = DarkSevLowBg, sevLowText = DarkSevLowText,
            isDark = true
        )
    } else {
        PaddyCareColors(
            bgPrimary = BgPrimary, bgCard = BgCard, bgCardAlt = BgCardAlt,
            bgTertiary = BgTertiary, greenPrimary = GreenPrimary, greenDark = GreenDark,
            greenPale = GreenPale, greenUltra = GreenUltra,
            textPrimary = TextPrimary, textSecondary = TextSecondary, textMuted = TextMuted,
            borderLight = BorderLight, borderMid = BorderMid, divider = DividerColor,
            sevHighBg = SeverityHighBg, sevHighText = SeverityHighText,
            sevMedBg = SeverityMediumBg, sevMedText = SeverityMediumText,
            sevLowBg = SeverityLowBg, sevLowText = SeverityLowText,
            isDark = false
        )
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = Color.Transparent.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    CompositionLocalProvider(LocalPaddyCareColors provides paddyCareColors) {
        MaterialTheme(
            colorScheme = colorScheme,
            typography = Typography,
            content = content
        )
    }
}
