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
}
