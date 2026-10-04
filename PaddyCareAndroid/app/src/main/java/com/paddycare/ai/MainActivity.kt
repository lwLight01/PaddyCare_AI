package com.paddycare.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.paddycare.ai.navigation.PaddyCareNavHost
import com.paddycare.ai.ui.theme.PaddyCareTheme
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val app = application as com.paddycare.ai.PaddyCareApp
            val isDarkTheme by app.settingsManager.darkMode.collectAsState(initial = androidx.compose.foundation.isSystemInDarkTheme())
            PaddyCareTheme(darkTheme = isDarkTheme) {
                PaddyCareNavHost()
            }
        }
    }
}
