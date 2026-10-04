package com.paddycare.ai.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.KeyboardArrowDown
import androidx.compose.material.icons.outlined.KeyboardArrowUp
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
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

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .clickable { expanded = !expanded }
    ) {
        // ── Header row: disease name + confidence ───────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top
        ) {
            Column(modifier = Modifier.weight(1f)) {
                // Rank label
                Text(
                    text = if (rank == 1) "প্রধান সন্দেহ" else "অন্যান্য সম্ভাবনা",
                    color = TextMuted,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Normal,
                    letterSpacing = 0.3.sp
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = prediction.diseaseBn,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Medium,
                    color = TextPrimary
                )
            }

            // Confidence value
            Text(
                text = "${conf.roundToInt()}%",
                fontSize = 22.sp,
                fontWeight = FontWeight.SemiBold,
                color = barColor as Color
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ── Slim confidence bar ─────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(ElevatedSurface)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(fraction = animatedFraction)
                    .fillMaxHeight()
                    .clip(RoundedCornerShape(3.dp))
                    .background(barColor as Color)
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // ── Severity badge + expand toggle ──────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Severity badge — minimal pill
            Box(
                modifier = Modifier
                    .background(bgColor as Color, RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 3.dp)
            ) {
                Text(
                    text = severityText as String,
                    color = textColor as Color,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            // Expand/collapse toggle
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.clickable { expanded = !expanded }
            ) {
                Text(
                    text = if (expanded) stringResource(R.string.hide_treatment) else stringResource(R.string.show_treatment),
                    color = PrimaryAccent,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium
                )
                Icon(
                    imageVector = if (expanded) Icons.Outlined.KeyboardArrowUp else Icons.Outlined.KeyboardArrowDown,
                    contentDescription = null,
                    tint = PrimaryAccent,
                    modifier = Modifier.size(18.dp)
                )
            }
        }

        // ── Expandable Treatment Section ────────────────────────────────
        AnimatedVisibility(
            visible = expanded,
            enter = fadeIn() + expandVertically(),
            exit = fadeOut() + shrinkVertically()
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(SurfaceSecondary)
                        .padding(14.dp)
                ) {
                    Text(
                        text = prediction.treatment,
                        color = TextPrimary,
                        fontSize = 13.sp,
                        lineHeight = 20.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // ── Divider between cards ───────────────────────────────────────
        HorizontalDivider(
            thickness = 0.5.dp,
            color = DividerColor
        )

        Spacer(modifier = Modifier.height(12.dp))
    }
}
