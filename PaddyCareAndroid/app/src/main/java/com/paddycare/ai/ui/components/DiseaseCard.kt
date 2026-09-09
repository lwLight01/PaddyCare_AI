package com.paddycare.ai.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.paddycare.ai.R
import com.paddycare.ai.data.Prediction
import com.paddycare.ai.ui.theme.*
import kotlin.math.roundToInt

@Composable
fun DiseaseCard(prediction: Prediction, rank: Int) {
    var expanded by remember { mutableStateOf(false) }

    // Animated confidence bar
    var animateBar by remember { mutableStateOf(false) }
    val animatedFraction by animateFloatAsState(
        targetValue = if (animateBar) prediction.confidence / 100f else 0f,
        animationSpec = tween(durationMillis = 800, easing = FastOutSlowInEasing),
        label = "conf_bar"
    )
    LaunchedEffect(Unit) { animateBar = true }

    val conf = prediction.confidence
    val (barColor, bgColor, textColor, severityText) = when {
        conf >= 70 -> listOf(SeverityHigh, SeverityHighBg, SeverityHighText, stringResource(R.string.severity_high))
        conf >= 40 -> listOf(SeverityMedium, SeverityMediumBg, SeverityMediumText, stringResource(R.string.severity_medium))
        else -> listOf(SeverityLow, SeverityLowBg, SeverityLowText, stringResource(R.string.severity_low))
    }

    val borderColor = if (rank == 1) GreenMid else Color(0xFFD4D4D8)
    val bannerGradient = if (rank == 1) {
        Brush.horizontalGradient(listOf(GradientStart, GradientEnd))
    } else {
        Brush.horizontalGradient(listOf(Color(0xFF78716C), Color(0xFFA8A29E)))
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(Surface)
            .border(if (rank == 1) 2.dp else 1.dp, borderColor, RoundedCornerShape(16.dp))
            .clickable { expanded = !expanded }
    ) {
        // ── Gradient Banner ─────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(bannerGradient)
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = if (rank == 1) "প্রধান সন্দেহ" else "অন্যান্য সম্ভাবনা",
                color = Surface,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
            Box(
                modifier = Modifier
                    .background(Color.White.copy(alpha = 0.2f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 8.dp, vertical = 2.dp)
            ) {
                Text(
                    text = "#$rank",
                    color = Surface,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // ── Content ─────────────────────────────────────────────────────
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = prediction.diseaseBn,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextMain,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = "${conf.roundToInt()}%",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = barColor as Color
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Animated Confidence Bar ─────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(10.dp)
                    .clip(RoundedCornerShape(5.dp))
                    .background(Color(0xFFE2E8F0))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(fraction = animatedFraction)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(5.dp))
                        .background(
                            Brush.horizontalGradient(
                                listOf(
                                    (barColor as Color).copy(alpha = 0.7f),
                                    barColor as Color
                                )
                            )
                        )
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Severity badge
                Box(
                    modifier = Modifier
                        .background(bgColor as Color, RoundedCornerShape(8.dp))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = severityText as String,
                        color = textColor as Color,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }

                // Chevron expand/collapse
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.clickable { expanded = !expanded }
                ) {
                    Text(
                        text = if (expanded) stringResource(R.string.hide_treatment) else stringResource(R.string.show_treatment),
                        color = GreenMid,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Icon(
                        imageVector = if (expanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                        contentDescription = null,
                        tint = GreenMid,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            // ── Expandable Treatment Section ────────────────────────────
            AnimatedVisibility(
                visible = expanded,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically()
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(GreenPale)
                        .border(1.dp, GreenLight.copy(alpha = 0.4f), RoundedCornerShape(12.dp))
                        .padding(16.dp)
                ) {
                    Text(
                        text = prediction.treatment,
                        color = TextMain,
                        fontSize = 14.sp,
                        lineHeight = 22.sp
                    )
                }
            }
        }
    }
}
