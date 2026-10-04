package com.paddycare.ai

import android.app.Application
import com.paddycare.ai.data.AppDatabase
import com.paddycare.ai.data.DiseaseInfo
import com.paddycare.ai.data.SettingsManager
import com.paddycare.ai.ml.PaddyClassifier

/**
 * Application class — initialises the Room database, settings, disease metadata,
 * and maintains a reusable PaddyClassifier instance to avoid re-mapping the 117MB model.
 */
class PaddyCareApp : Application() {

    val database: AppDatabase by lazy {
        AppDatabase.getInstance(this)
    }

    val settingsManager: SettingsManager by lazy {
        SettingsManager(this)
    }

    val classifier: PaddyClassifier by lazy {
        PaddyClassifier(this)
    }

    override fun onCreate() {
        super.onCreate()
        DiseaseInfo.init(this)
    }
}
