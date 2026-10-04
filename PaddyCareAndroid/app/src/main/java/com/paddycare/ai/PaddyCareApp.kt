package com.paddycare.ai

import android.app.Application
import com.paddycare.ai.data.AppDatabase

/**
 * Application class — initialises the Room database singleton.
 */
class PaddyCareApp : Application() {

    val database: AppDatabase by lazy {
        AppDatabase.getInstance(this)
    }

    val settingsManager: com.paddycare.ai.data.SettingsManager by lazy {
        com.paddycare.ai.data.SettingsManager(this)
    }
}
