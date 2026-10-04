package com.paddycare.ai.data

import android.content.Context
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class SettingsManager(context: Context) {
    private val prefs = context.getSharedPreferences("paddy_settings", Context.MODE_PRIVATE)

    private val _darkMode = MutableStateFlow(prefs.getBoolean("dark_mode", false))
    val darkMode: StateFlow<Boolean> = _darkMode.asStateFlow()

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

    private val _notifOn = MutableStateFlow(prefs.getBoolean("notif_on", true))
    val notifOn: StateFlow<Boolean> = _notifOn.asStateFlow()

    fun setNotifOn(isOn: Boolean) {
        prefs.edit { putBoolean("notif_on", isOn) }
        _notifOn.value = isOn
    }

    private val _emailOn = MutableStateFlow(prefs.getBoolean("email_on", false))
    val emailOn: StateFlow<Boolean> = _emailOn.asStateFlow()

    fun setEmailOn(isOn: Boolean) {
        prefs.edit { putBoolean("email_on", isOn) }
        _emailOn.value = isOn
    }
}
