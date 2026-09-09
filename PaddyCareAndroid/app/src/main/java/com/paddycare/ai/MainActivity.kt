package com.paddycare.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.paddycare.ai.navigation.PaddyCareNavHost
import com.paddycare.ai.ui.theme.PaddyCareTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            PaddyCareTheme {
                PaddyCareNavHost()
            }
        }
    }
}
