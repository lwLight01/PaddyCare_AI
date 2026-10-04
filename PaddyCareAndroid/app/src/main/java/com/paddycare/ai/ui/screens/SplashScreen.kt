package com.paddycare.ai.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

@Composable
fun SplashScreen(onFinished: () -> Unit) {
    val logoScale  by animateFloatAsState(targetValue = 1f, animationSpec = spring(dampingRatio = 0.6f, stiffness = 200f), label = "logo")
    var titleAlpha by remember { mutableFloatStateOf(0f) }
    var dotsAlpha  by remember { mutableFloatStateOf(0f) }

    LaunchedEffect(Unit) {
        delay(200)
        titleAlpha = 1f
        delay(400)
        dotsAlpha  = 1f
        delay(1600)
        onFinished()
    }

    val titleAnim by animateFloatAsState(titleAlpha, tween(500), label = "title")
    val dotsAnim  by animateFloatAsState(dotsAlpha,  tween(400), label = "dots")

    val infiniteTransition = rememberInfiniteTransition(label = "splash_dots")

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.linearGradient(
                    colors = listOf(Color(0xFF1A4A10), Color(0xFF0D2E06), Color(0xFF0A2005))
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        // Decorative circles
        Box(
            modifier = Modifier
                .size(300.dp)
                .offset(x = 130.dp, y = (-100).dp)
                .background(Color.White.copy(alpha = 0.04f), CircleShape)
        )
        Box(
            modifier = Modifier
                .size(200.dp)
                .offset(x = (-100).dp, y = 150.dp)
                .background(Color.White.copy(alpha = 0.04f), CircleShape)
        )

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier.padding(32.dp)
        ) {
            // Logo
            Surface(
                modifier = Modifier
                    .size(110.dp)
                    .scale(logoScale),
                shape = RoundedCornerShape(28.dp),
                color = Color(0xFF2E7D1F).copy(alpha = 0.9f)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text("🌾", fontSize = 48.sp)
                }
            }

            Spacer(Modifier.height(28.dp))

            // App name
            Text(
                text = "PaddyCare",
                color = Color.White,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.alpha(titleAnim)
            )
            Text(
                text = "AI",
                color = Color(0xFF7ED661),
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .offset(y = (-6).dp)
                    .alpha(titleAnim)
            )

            Spacer(Modifier.height(8.dp))

            Text(
                text = "ধান রোগ সনাক্তকরণ সিস্টেম",
                color = Color.White.copy(0.70f),
                fontSize = 14.sp,
                modifier = Modifier.alpha(titleAnim)
            )

            Spacer(Modifier.height(36.dp))

            // Badge icons
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.alpha(titleAnim)
            ) {
                listOf("🇧🇩", "🤖", "🌿").forEach { emoji ->
                    Surface(
                        shape = CircleShape,
                        color = Color.White.copy(alpha = 0.12f)
                    ) {
                        Text(emoji, fontSize = 22.sp, modifier = Modifier.padding(10.dp))
                    }
                }
            }

            Spacer(Modifier.height(48.dp))

            // Loading dots
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.alpha(dotsAnim)
            ) {
                (0..2).forEach { i ->
                    val dotAlpha by infiniteTransition.animateFloat(
                        initialValue = 0.3f, targetValue = 1f,
                        animationSpec = infiniteRepeatable(
                            tween(600, delayMillis = i * 200),
                            RepeatMode.Reverse
                        ),
                        label = "dot_$i"
                    )
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .alpha(dotAlpha)
                            .background(Color(0xFF7ED661), CircleShape)
                    )
                }
            }
        }
    }
}
