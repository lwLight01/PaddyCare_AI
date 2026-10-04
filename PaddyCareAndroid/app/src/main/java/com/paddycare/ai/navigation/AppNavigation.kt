package com.paddycare.ai.navigation

import android.net.Uri
import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.compose.*
import com.paddycare.ai.PaddyCareApp
import com.paddycare.ai.R
import com.paddycare.ai.data.PredictionResult
import com.paddycare.ai.data.ScanRecord
import com.paddycare.ai.ui.screens.*
import com.paddycare.ai.ui.theme.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

sealed class Screen(val route: String) {
    object Splash   : Screen("splash")
    object Home     : Screen("home")
    object Result   : Screen("result")
    object History  : Screen("history")
    object Experts  : Screen("experts")
    object Settings : Screen("settings")
}

data class NavItem(
    val screen: Screen,
    val label: String,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector,
    val isFab: Boolean = false
)

@Composable
fun PaddyCareNavHost() {
    val navController  = rememberNavController()
    val context        = LocalContext.current
    val database       = (context.applicationContext as PaddyCareApp).database
    val coroutineScope = rememberCoroutineScope()
    val pc             = LocalPaddyCareColors.current

    var currentResult  by remember { mutableStateOf<PredictionResult?>(null) }
    var currentImageUri by remember { mutableStateOf<Uri?>(null) }
    var launchCameraEvent by remember { mutableStateOf(false) }

    val navItems = listOf(
        NavItem(Screen.Home,     "হোম",     Icons.Filled.Home,        Icons.Outlined.Home),
        NavItem(Screen.History,  "ইতিহাস", Icons.Filled.History,     Icons.Outlined.History),
        NavItem(Screen.Home,     "স্ক্যান", Icons.Filled.CameraAlt,   Icons.Outlined.CameraAlt, isFab = true),
        NavItem(Screen.Experts,  "বিশেষজ্ঞ",Icons.Filled.People,     Icons.Outlined.People),
        NavItem(Screen.Settings, "সেটিং",  Icons.Filled.Settings,    Icons.Outlined.Settings),
    )

    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    val bottomBarScreens = setOf(
        Screen.Home.route, Screen.History.route, Screen.Experts.route, Screen.Settings.route
    )
    val showBottomBar = currentRoute in bottomBarScreens

    Scaffold(
        bottomBar = {
            AnimatedVisibility(
                visible = showBottomBar,
                enter = slideInVertically { it },
                exit  = slideOutVertically { it }
            ) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .shadow(elevation = 16.dp, shape = RoundedCornerShape(topStart = 0.dp, topEnd = 0.dp)),
                    color = pc.bgCard,
                    tonalElevation = 0.dp
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(68.dp)
                            .background(pc.bgCard),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        navItems.forEach { item ->
                            if (item.isFab) {
                                // FAB scan button
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxHeight(),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .size(52.dp)
                                            .offset(y = (-6).dp)
                                            .background(
                                                Brush.linearGradient(listOf(GreenLight, pc.greenPrimary)),
                                                CircleShape
                                            )
                                            .shadow(8.dp, CircleShape),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        IconButton(
                                            onClick = { 
                                                launchCameraEvent = true
                                                if (currentRoute != Screen.Home.route) {
                                                    navController.navigate(Screen.Home.route) { launchSingleTop = true } 
                                                }
                                            },
                                            modifier = Modifier.size(52.dp)
                                        ) {
                                            Icon(
                                                item.selectedIcon, null,
                                                tint = Color.White,
                                                modifier = Modifier.size(24.dp)
                                            )
                                        }
                                    }
                                }
                            } else {
                                val isSelected = currentRoute == item.screen.route
                                NavigationBarItem(
                                    modifier = Modifier.weight(1f),
                                    selected = isSelected,
                                    onClick = {
                                        navController.navigate(item.screen.route) {
                                            popUpTo(navController.graph.startDestinationId) { saveState = true }
                                            launchSingleTop = true
                                            restoreState     = true
                                        }
                                    },
                                    icon = {
                                        Box(contentAlignment = Alignment.TopCenter) {
                                            Icon(
                                                imageVector = if (isSelected) item.selectedIcon else item.unselectedIcon,
                                                contentDescription = null,
                                                modifier = Modifier.size(22.dp)
                                            )
                                        }
                                    },
                                    label = {
                                        Text(
                                            text = item.label,
                                            fontSize = 9.sp,
                                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                        )
                                    },
                                    colors = NavigationBarItemDefaults.colors(
                                        selectedIconColor   = pc.greenPrimary,
                                        selectedTextColor   = pc.greenPrimary,
                                        unselectedIconColor = pc.textMuted,
                                        unselectedTextColor = pc.textMuted,
                                        indicatorColor      = pc.greenPale
                                    )
                                )
                            }
                        }
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController    = navController,
            startDestination = Screen.Splash.route,
            modifier         = Modifier.padding(innerPadding),
            enterTransition  = { fadeIn(tween(250)) + slideInHorizontally { it / 12 } },
            exitTransition   = { fadeOut(tween(200)) },
            popEnterTransition  = { fadeIn(tween(250)) },
            popExitTransition   = { fadeOut(tween(200)) + slideOutHorizontally { it / 12 } }
        ) {
            // Splash
            composable(Screen.Splash.route) {
                SplashScreen {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                }
            }

            // Home / Scan
            composable(Screen.Home.route) {
                HomeScreen(
                    launchCameraEvent = launchCameraEvent,
                    onCameraLaunched = { launchCameraEvent = false },
                    onAnalyzeComplete = { result, uri ->
                        currentResult   = result
                        currentImageUri = uri
                        coroutineScope.launch(kotlinx.coroutines.Dispatchers.IO) {
                            val savedPath = persistScanImage(context, uri)
                            val record = when (result) {
                                is PredictionResult.Success -> {
                                    val top = result.predictions.first()
                                    ScanRecord(
                                        imagePath    = savedPath,
                                        status       = "ok",
                                        diseaseName  = top.disease,
                                        diseaseNameBn = top.diseaseBn,
                                        confidence   = top.confidence,
                                        treatment    = top.treatment
                                    )
                                }
                                is PredictionResult.NotPaddy      -> ScanRecord(imagePath = savedPath, status = "not_paddy", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
                                is PredictionResult.LowConfidence -> ScanRecord(imagePath = savedPath, status = "low_confidence", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
                                is PredictionResult.Error         -> ScanRecord(imagePath = savedPath, status = "error", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
                            }
                            database.scanDao().insert(record)
                        }
                        navController.navigate(Screen.Result.route)
                    }
                )
            }

            // Result
            composable(Screen.Result.route) {
                if (currentResult != null && currentImageUri != null) {
                    ResultScreen(
                        result      = currentResult!!,
                        imageUri    = currentImageUri!!,
                        onBack      = { navController.popBackStack() },
                        onScanAgain = { navController.popBackStack(Screen.Home.route, false) }
                    )
                }
            }

            // History
            composable(Screen.History.route) { HistoryScreen(database = database) }

            // Experts
            composable(Screen.Experts.route) { ExpertsScreen() }

            // Settings
            composable(Screen.Settings.route) { SettingsScreen() }
        }
    }
}

private fun persistScanImage(context: android.content.Context, sourceUri: Uri): String {
    return try {
        val scansDir = java.io.File(context.filesDir, "scans").apply { mkdirs() }
        val destFile = java.io.File(scansDir, "scan_${System.currentTimeMillis()}.jpg")
        context.contentResolver.openInputStream(sourceUri)?.use { input ->
            destFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        Uri.fromFile(destFile).toString()
    } catch (e: Exception) {
        sourceUri.toString()
    }
}
