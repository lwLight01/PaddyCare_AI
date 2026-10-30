package com.paddycare.ai.data

import android.content.Context
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class SettingsManager(context: Context) {
    private val prefs = context.getSharedPreferences("paddy_settings", Context.MODE_PRIVATE)

    /** Null means "follow system theme"; true/false means user explicitly toggled dark/light mode. */
    private val _darkMode = MutableStateFlow<Boolean?>(
        if (prefs.contains("dark_mode")) prefs.getBoolean("dark_mode", false) else null
    )
    val darkMode: StateFlow<Boolean?> = _darkMode.asStateFlow()

    fun setDarkMode(isDark: Boolean) {
        prefs.edit { putBoolean("dark_mode", isDark) }
        _darkMode.value = isDark
    }

    private val _language = MutableStateFlow(prefs.getString("language", "bn") ?: "bn")
    val language: StateFlow<String> = _language.asStateFlow()

    fun setLanguage(lang: String) {
        prefs.edit { putString("language", lang) }
        _language.value = lang
    }
}
