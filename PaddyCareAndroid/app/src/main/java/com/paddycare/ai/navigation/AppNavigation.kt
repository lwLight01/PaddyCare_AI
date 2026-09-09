package com.paddycare.ai.navigation

import android.net.Uri
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.*
import com.paddycare.ai.PaddyCareApp
import com.paddycare.ai.R
import com.paddycare.ai.data.PredictionResult
import com.paddycare.ai.data.ScanRecord
import com.paddycare.ai.ui.screens.*
import com.paddycare.ai.ui.theme.*
import kotlinx.coroutines.launch

sealed class Screen(val route: String) {
    object Home : Screen("home")
    object Result : Screen("result")
    object History : Screen("history")
}

@Composable
fun PaddyCareNavHost() {
    val navController = rememberNavController()
    val context = LocalContext.current
    val database = (context.applicationContext as PaddyCareApp).database
    val coroutineScope = rememberCoroutineScope()

    // Shared state for navigation
    var currentResult by remember { mutableStateOf<PredictionResult?>(null) }
    var currentImageUri by remember { mutableStateOf<Uri?>(null) }

    Scaffold(
        bottomBar = {
            val navBackStackEntry by navController.currentBackStackEntryAsState()
            val currentRoute = navBackStackEntry?.destination?.route

            if (currentRoute == Screen.Home.route || currentRoute == Screen.History.route) {
                NavigationBar(
                    containerColor = Surface,
                    tonalElevation = 8.dp
                ) {
                    NavigationBarItem(
                        selected = currentRoute == Screen.Home.route,
                        onClick = {
                            navController.navigate(Screen.Home.route) {
                                popUpTo(navController.graph.startDestinationId) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {
                            Icon(
                                imageVector = if (currentRoute == Screen.Home.route)
                                    Icons.Filled.Home else Icons.Outlined.Home,
                                contentDescription = null
                            )
                        },
                        label = { Text("হোম") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = GreenMid,
                            selectedTextColor = GreenMid,
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted,
                            indicatorColor = GreenPale
                        )
                    )
                    NavigationBarItem(
                        selected = currentRoute == Screen.History.route,
                        onClick = {
                            navController.navigate(Screen.History.route) {
                                popUpTo(navController.graph.startDestinationId) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {
                            Icon(
                                imageVector = if (currentRoute == Screen.History.route)
                                    Icons.Filled.History else Icons.Outlined.History,
                                contentDescription = null
                            )
                        },
                        label = { Text("ইতিহাস") },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = GreenMid,
                            selectedTextColor = GreenMid,
                            unselectedIconColor = TextMuted,
                            unselectedTextColor = TextMuted,
                            indicatorColor = GreenPale
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Home.route) {
                HomeScreen(
                    onAnalyzeComplete = { result, uri ->
                        currentResult = result
                        currentImageUri = uri
                        
                        // Save to history
                        coroutineScope.launch {
                            val record = when (result) {
                                is PredictionResult.Success -> {
                                    val topPred = result.predictions.first()
                                    ScanRecord(
                                        imagePath = uri.toString(),
                                        status = "ok",
                                        diseaseName = topPred.disease,
                                        diseaseNameBn = topPred.diseaseBn,
                                        confidence = topPred.confidence,
                                        treatment = topPred.treatment
                                    )
                                }
                                is PredictionResult.NotPaddy -> ScanRecord(imagePath = uri.toString(), status = "not_paddy", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
                                is PredictionResult.LowConfidence -> ScanRecord(imagePath = uri.toString(), status = "low_confidence", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
                                is PredictionResult.Error -> ScanRecord(imagePath = uri.toString(), status = "error", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
                            }
                            database.scanDao().insert(record)
                        }
                        
                        navController.navigate(Screen.Result.route)
                    }
                )
            }
            
            composable(Screen.Result.route) {
                if (currentResult != null && currentImageUri != null) {
                    ResultScreen(
                        result = currentResult!!,
                        imageUri = currentImageUri!!,
                        onBack = { navController.popBackStack() },
                        onScanAgain = { 
                            navController.popBackStack(Screen.Home.route, false) 
                        }
                    )
                }
            }
            
            composable(Screen.History.route) {
                HistoryScreen(database = database)
            }
        }
    }
}
