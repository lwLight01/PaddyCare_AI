package com.paddycare.ai.ui.screens

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
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
import com.paddycare.ai.data.AppDatabase
import com.paddycare.ai.data.ScanRecord
import com.paddycare.ai.ui.theme.*
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun HistoryScreen(database: AppDatabase) {
    val pc             = LocalPaddyCareColors.current
    val coroutineScope = rememberCoroutineScope()
    val scans          by database.scanDao().getAllScans().collectAsState(initial = emptyList())
    var showDialog     by remember { mutableStateOf(false) }
    var selectedTab    by remember { mutableIntStateOf(0) } // 0=List, 1=Stats

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = {
                Text(
                    stringResource(R.string.clear_confirm_title),
                    fontWeight = FontWeight.Bold, fontSize = 16.sp
                )
            },
            text = {
                Text(
                    stringResource(R.string.clear_confirm_msg),
                    color = pc.textSecondary, fontSize = 14.sp
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    coroutineScope.launch { database.scanDao().deleteAll() }
                    showDialog = false
                }) { Text(stringResource(R.string.yes), color = SeverityHigh) }
            },
            dismissButton = {
                TextButton(onClick = { showDialog = false }) {
                    Text(stringResource(R.string.no), color = pc.textSecondary)
                }
            },
            shape = RoundedCornerShape(20.dp),
            containerColor = pc.bgCard
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(pc.bgPrimary)
    ) {
        // ── Top bar ──────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(pc.bgCard)
                .padding(top = 52.dp, bottom = 12.dp, start = 20.dp, end = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = "ইতিহাস",
                    color = pc.textPrimary,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "আপনার স্ক্যান রেকর্ড",
                    color = pc.textMuted,
                    fontSize = 12.sp
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (scans.isNotEmpty()) {
                    IconButton(onClick = { showDialog = true }) {
                        Box(
                            modifier = Modifier.size(36.dp).background(pc.bgTertiary, CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Outlined.Delete, null, tint = SeverityHigh, modifier = Modifier.size(18.dp))
                        }
                    }
                }
            }
        }
        HorizontalDivider(thickness = 1.dp, color = pc.borderLight)

        // ── Tabs ─────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(pc.bgCard)
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            HistoryTab("📋  তালিকা", selectedTab == 0, pc) { selectedTab = 0 }
            HistoryTab("📊  পরিসংখ্যান", selectedTab == 1, pc) { selectedTab = 1 }
        }

        // ── Content ───────────────────────────────────────────────────
        if (selectedTab == 0) {
            HistoryListView(scans, pc)
        } else {
            HistoryStatsView(scans, pc)
        }
    }
}

@Composable
private fun HistoryTab(label: String, selected: Boolean, pc: PaddyCareColors, onClick: () -> Unit) {
    Surface(
        modifier = Modifier.clickable { onClick() },
        shape = RoundedCornerShape(10.dp),
        color = if (selected) pc.greenPale else Color.Transparent,
        border = if (selected) BorderStroke(1.dp, pc.borderMid) else null
    ) {
        Text(
            text = label,
            fontSize = 12.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color = if (selected) pc.greenPrimary else pc.textMuted,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp)
        )
    }
}

@Composable
private fun HistoryListView(scans: List<ScanRecord>, pc: PaddyCareColors) {
    if (scans.isEmpty()) {
        Box(modifier = Modifier.fillMaxSize().padding(48.dp), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("📋", fontSize = 56.sp)
                Spacer(Modifier.height(16.dp))
                Text(
                    "কোনো ইতিহাস নেই",
                    color = pc.textPrimary,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    "এখনো কোনো ধান পাতা স্ক্যান করা হয়নি",
                    color = pc.textMuted,
                    fontSize = 13.sp
                )
            }
        }
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            items(scans) { record ->
                HistoryItemCard(record, pc)
            }
        }
    }
}

@Composable
private fun HistoryItemCard(record: ScanRecord, pc: PaddyCareColors) {
    val formatter  = SimpleDateFormat("dd MMM, hh:mm a", Locale.getDefault())
    val dateString = formatter.format(Date(record.timestamp))

    val (accentColor, accentBg, severityLabel) = when {
        record.status == "ok" && (record.confidence ?: 0f) >= 70f ->
            Triple(SeverityHigh, SeverityHighBg, "মারাত্মক")
        record.status == "ok" && (record.confidence ?: 0f) >= 40f ->
            Triple(SeverityMedium, SeverityMediumBg, "মাঝারি")
        record.status == "ok" ->
            Triple(SeverityLow, SeverityLowBg, "সুস্থ")
        else ->
            Triple(SeverityMedium, SeverityMediumBg, "অজানা")
    }

    Surface(
        shape = RoundedCornerShape(16.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight),
        shadowElevation = 1.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Thumbnail
            Box(
                modifier = Modifier
                    .size(60.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(pc.bgTertiary)
            ) {
                if (record.imagePath.isNotBlank()) {
                    AsyncImage(
                        model = record.imagePath,
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                } else {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text("🌾", fontSize = 22.sp)
                    }
                }
                // Status indicator dot
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .align(Alignment.TopEnd)
                        .offset(x = 2.dp, y = (-2).dp)
                        .background(accentColor, CircleShape)
                        .border(1.5.dp, pc.bgCard, CircleShape)
                )
            }

            // Info
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = record.diseaseNameBn ?: record.diseaseName ?: "অজানা",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = pc.textPrimary
                )
                Spacer(Modifier.height(3.dp))
                if (record.confidence != null) {
                    Text(
                        text = "${record.confidence.toInt()}% নিশ্চিত",
                        fontSize = 12.sp,
                        color = pc.greenPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                    Spacer(Modifier.height(2.dp))
                }
                Text(dateString, fontSize = 11.sp, color = pc.textMuted)
            }

            // Severity badge
            if (record.status == "ok") {
                Surface(shape = RoundedCornerShape(20.dp), color = accentBg) {
                    Text(
                        text = severityLabel,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        color = accentColor,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun HistoryStatsView(scans: List<ScanRecord>, pc: PaddyCareColors) {
    val validScans  = scans.filter { it.status == "ok" }
    val total       = validScans.size
    val diseased    = validScans.filter { (it.diseaseName ?: "") != "Healthy" }.size
    val healthy     = validScans.filter { it.diseaseName == "Healthy" }.size
    val avgConf     = if (total > 0) validScans.mapNotNull { it.confidence }.average().toInt() else 0

    val blastCount   = validScans.count { it.diseaseName == "Blast" }
    val brownCount   = validScans.count { it.diseaseName == "Brown Spot" }
    val blightCount  = validScans.count { it.diseaseName == "Bacterial Blight" }
    val healthyCount = validScans.count { it.diseaseName == "Healthy" }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // ── Stats grid ────────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            BigStatCard("$total", "মোট স্ক্যান", pc.greenPrimary, pc, Modifier.weight(1f))
            BigStatCard("$diseased", "রোগাক্রান্ত", SeverityHigh, pc, Modifier.weight(1f))
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            BigStatCard("$healthy", "সুস্থ গাছ", SeverityLow, pc, Modifier.weight(1f))
            BigStatCard(if (total > 0) "$avgConf%" else "-", "গড় নিশ্চয়তা", Color(0xFF2A7DE1), pc, Modifier.weight(1f))
        }

        // ── Bar chart ─────────────────────────────────────────────────
        if (total > 0) {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = pc.bgCard,
                border = BorderStroke(1.dp, pc.borderLight)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("📈  রোগের বিতরণ", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = pc.textPrimary)
                    Spacer(Modifier.height(16.dp))
                    val maxVal = maxOf(blastCount, brownCount, blightCount, healthyCount, 1)
                    Row(
                        modifier = Modifier.fillMaxWidth().height(90.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.Bottom
                    ) {
                        BarItem("ব্লাস্ট", blastCount, maxVal, SeverityHigh, pc)
                        BarItem("বাদামী", brownCount, maxVal, SeverityMedium, pc)
                        BarItem("ব্লাইট", blightCount, maxVal, SeverityHigh, pc)
                        BarItem("সুস্থ", healthyCount, maxVal, SeverityLow, pc)
                    }
                }
            }
        }

        // ── Tip ──────────────────────────────────────────────────────
        Surface(
            shape = RoundedCornerShape(16.dp),
            color = pc.greenPale,
            border = BorderStroke(1.dp, pc.borderMid)
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                Text("📌", fontSize = 22.sp)
                Column {
                    Text("নিয়মিত পর্যবেক্ষণ করুন", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = pc.greenDark)
                    Spacer(Modifier.height(3.dp))
                    Text(
                        "প্রতি সপ্তাহে অন্তত একবার ধান পাতা পরীক্ষা করুন। দ্রুত সনাক্ত করলে রোগ নিয়ন্ত্রণ সহজ হয়।",
                        fontSize = 12.sp, color = pc.textSecondary, lineHeight = 18.sp
                    )
                }
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun RowScope.BarItem(label: String, count: Int, maxVal: Int, barColor: Color, pc: PaddyCareColors) {
    Column(
        modifier = Modifier.weight(1f),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Bottom
    ) {
        Text("$count", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = barColor)
        Spacer(Modifier.height(4.dp))
        var animH by remember { mutableFloatStateOf(0f) }
        LaunchedEffect(count) { animH = if (maxVal > 0) count.toFloat() / maxVal else 0f }
        val ah by animateFloatAsState(animH, tween(900, easing = FastOutSlowInEasing), label = "bar")
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
            contentAlignment = Alignment.BottomCenter
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .fillMaxHeight(ah.coerceAtLeast(if (count > 0) 0.06f else 0f))
                    .clip(RoundedCornerShape(topStart = 6.dp, topEnd = 6.dp))
                    .background(Brush.verticalGradient(listOf(barColor.copy(0.7f), barColor)))
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(label, fontSize = 9.sp, color = pc.textMuted)
    }
}

@Composable
private fun BigStatCard(value: String, label: String, valueColor: Color, pc: PaddyCareColors, modifier: Modifier) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(value, fontSize = 28.sp, fontWeight = FontWeight.Bold, color = valueColor)
            Spacer(Modifier.height(4.dp))
            Text(label, fontSize = 11.sp, color = pc.textMuted)
        }
    }
}
