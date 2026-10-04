package com.paddycare.ai.ui.screens

import android.net.Uri
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.paddycare.ai.R
import com.paddycare.ai.data.Prediction
import com.paddycare.ai.data.PredictionResult
import com.paddycare.ai.ui.components.WarningBanner
import com.paddycare.ai.ui.theme.*

@Composable
fun ResultScreen(
    result: PredictionResult,
    imageUri: Uri,
    onBack: () -> Unit,
    onScanAgain: () -> Unit
) {
    val pc = LocalPaddyCareColors.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(pc.bgPrimary)
    ) {
        // ── Top bar ──────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(pc.bgCard)
                .padding(top = 48.dp, bottom = 12.dp, start = 4.dp, end = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, null, tint = pc.textPrimary)
            }
            Text(
                text = "বিশ্লেষণের ফলাফল",
                color = pc.textPrimary,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            // Share button
            IconButton(onClick = {}) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .background(pc.bgTertiary, CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Outlined.Share, null, tint = pc.textSecondary, modifier = Modifier.size(18.dp))
                }
            }
        }
        HorizontalDivider(thickness = 1.dp, color = pc.borderLight)

        // ── Scrollable body ───────────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
        ) {
            var showContent by remember { mutableStateOf(false) }
            LaunchedEffect(Unit) { showContent = true }

            AnimatedVisibility(
                visible = showContent,
                enter = fadeIn(animationSpec = tween(400)) + slideInVertically(initialOffsetY = { it / 6 }, animationSpec = tween(450))
            ) {
                Column {
                    when (result) {
                        is PredictionResult.Success -> SuccessResult(result, imageUri, pc)
                        is PredictionResult.NotPaddy -> {
                            Spacer(Modifier.height(20.dp))
                            WarningBanner(result.message)
                        }
                        is PredictionResult.LowConfidence -> {
                            Spacer(Modifier.height(20.dp))
                            WarningBanner(result.message)
                        }
                        is PredictionResult.Error -> {
                            Spacer(Modifier.height(20.dp))
                            WarningBanner(result.message)
                        }
                    }
                }
            }

            Spacer(Modifier.height(20.dp))

            // ── Scan Again button ─────────────────────────────────────
            Button(
                onClick = onScanAgain,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .height(52.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = pc.greenPrimary,
                    contentColor = Color.White
                ),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp, pressedElevation = 0.dp)
            ) {
                Icon(Icons.Outlined.Refresh, null, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text(stringResource(R.string.scan_again), fontSize = 15.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(32.dp))
        }
    }
}

@Composable
private fun SuccessResult(
    result: PredictionResult.Success,
    imageUri: Uri,
    pc: PaddyCareColors
) {
    val top = result.predictions.first()
    val (sevColor, sevBgColor, sevLabel) = when {
        top.confidence >= 70f -> Triple(SeverityHigh, SeverityHighBg, "মারাত্মক")
        top.confidence >= 40f -> Triple(SeverityMedium, SeverityMediumBg, "মাঝারি")
        else                  -> Triple(SeverityLow,  SeverityLowBg,  "সামান্য")
    }

    // ── Full-width image with overlay ─────────────────────────────────
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(260.dp)
    ) {
        AsyncImage(
            model = imageUri,
            contentDescription = "Analyzed leaf",
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        // Gradient overlay bottom
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        listOf(Color.Transparent, Color.Black.copy(0.65f)),
                        startY = 120f
                    )
                )
        )
        // Disease info on image
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(16.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.Bottom,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = top.diseaseBn.ifBlank { top.disease },
                    color = Color.White,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(top.disease, color = Color.White.copy(0.75f), fontSize = 12.sp)
            }
            // Confidence bubble
            Surface(
                shape = RoundedCornerShape(20.dp),
                color = Color.White.copy(0.18f),
                border = BorderStroke(1.5.dp, Color.White.copy(0.35f))
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "${top.confidence.toInt()}%",
                        color = GreenBright,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text("নিশ্চিত", color = Color.White.copy(0.7f), fontSize = 10.sp)
                }
            }
        }
    }

    // ── Body ─────────────────────────────────────────────────────────
    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {

        // Stats row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            ResultStatCard("${top.confidence.toInt()}%", "নিশ্চয়তা", pc.greenPrimary, pc, Modifier.weight(1f))
            ResultStatCard(sevLabel, "মাত্রা", sevColor, pc, Modifier.weight(1f))
            ResultStatCard("#1", "র‍্যাংক", Color(0xFF2A7DE1), pc, Modifier.weight(1f))
        }

        // Confidence bar
        Surface(
            shape = RoundedCornerShape(14.dp),
            color = pc.bgCard,
            border = BorderStroke(1.dp, pc.borderLight)
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("নিশ্চয়তার মাত্রা", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = pc.textPrimary)
                    Text("${top.confidence.toInt()}%", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = pc.greenPrimary)
                }
                Spacer(Modifier.height(8.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(10.dp)
                        .clip(RoundedCornerShape(5.dp))
                        .background(pc.bgTertiary)
                ) {
                    var animWidth by remember { mutableFloatStateOf(0f) }
                    LaunchedEffect(Unit) { animWidth = top.confidence / 100f }
                    val animatedWidth by animateFloatAsState(
                        targetValue = animWidth, animationSpec = tween(1000, easing = FastOutSlowInEasing), label = "bar"
                    )
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(animatedWidth)
                            .fillMaxHeight()
                            .clip(RoundedCornerShape(5.dp))
                            .background(
                                Brush.horizontalGradient(
                                    when {
                                        top.confidence >= 70 -> listOf(Color(0xFFF87171), SeverityHigh)
                                        top.confidence >= 40 -> listOf(Color(0xFFFBBF24), SeverityMedium)
                                        else                 -> listOf(GreenLight, pc.greenPrimary)
                                    }
                                )
                            )
                    )
                }
            }
        }

        // Second prediction if exists
        if (result.predictions.size > 1) {
            val second = result.predictions[1]
            Surface(
                shape = RoundedCornerShape(14.dp),
                color = pc.bgCard,
                border = BorderStroke(1.dp, pc.borderLight)
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(
                            text = second.diseaseBn.ifBlank { second.disease },
                            fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = pc.textPrimary
                        )
                        Text("${second.confidence.toInt()}%", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = GoldColor)
                    }
                    Spacer(Modifier.height(6.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp))
                            .background(pc.bgTertiary)
                    ) {
                        var animW by remember { mutableFloatStateOf(0f) }
                        LaunchedEffect(Unit) { animW = second.confidence / 100f }
                        val aw by animateFloatAsState(animW, tween(1000, 200), label = "bar2")
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(aw)
                                .fillMaxHeight()
                                .clip(RoundedCornerShape(4.dp))
                                .background(Brush.horizontalGradient(listOf(Color(0xFFFBBF24), SeverityMedium)))
                        )
                    }
                    Spacer(Modifier.height(4.dp))
                    Text("দ্বিতীয় সম্ভাবনা", fontSize = 10.sp, color = pc.textMuted)
                }
            }
        }

        // Treatment accordion
        TreatmentAccordion(top, pc)
    }
}

@Composable
private fun ResultStatCard(value: String, label: String, valueColor: Color, pc: PaddyCareColors, modifier: Modifier) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(value, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = valueColor)
            Spacer(Modifier.height(2.dp))
            Text(label, fontSize = 10.sp, color = pc.textMuted)
        }
    }
}

@Composable
private fun TreatmentAccordion(prediction: Prediction, pc: PaddyCareColors) {
    var expanded by remember { mutableStateOf(false) }

    Surface(
        shape = RoundedCornerShape(14.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight)
    ) {
        Column {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded }
                    .padding(14.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .background(pc.greenPale, RoundedCornerShape(10.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("💊", fontSize = 18.sp)
                    }
                    Text(
                        "চিকিৎসার পরামর্শ দেখুন",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = pc.greenPrimary
                    )
                }
                Icon(
                    imageVector = if (expanded) Icons.Outlined.KeyboardArrowUp else Icons.Outlined.KeyboardArrowDown,
                    contentDescription = null,
                    tint = pc.textMuted
                )
            }

            AnimatedVisibility(visible = expanded) {
                Column {
                    HorizontalDivider(thickness = 1.dp, color = pc.borderLight)
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        val steps = prediction.treatment.split("\n").filter { it.isNotBlank() }
                        steps.forEachIndexed { idx, step ->
                            Row(
                                verticalAlignment = Alignment.Top,
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(24.dp)
                                        .background(pc.greenPrimary, CircleShape),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        text = "${idx + 1}",
                                        color = Color.White,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                Text(
                                    text = step.trim(),
                                    fontSize = 13.sp,
                                    color = pc.textSecondary,
                                    lineHeight = 19.sp,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
