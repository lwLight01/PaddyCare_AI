package com.paddycare.ai.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.paddycare.ai.ui.theme.*

/**
 * Expert advice is not available yet. The previous version listed invented people at real
 * institutions with buttons that did nothing; this screen is honest about the feature state
 * and points farmers to real local help instead.
 */
@Composable
fun ExpertsScreen() {
    val pc = LocalPaddyCareColors.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(pc.bgPrimary)
    ) {
        // ── Header ────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(pc.bgCard)
                .padding(top = 52.dp, bottom = 14.dp, start = 20.dp, end = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("বিশেষজ্ঞ", color = pc.textPrimary, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text("কৃষি পরামর্শদাতা", color = pc.textMuted, fontSize = 12.sp)
            }
        }
        HorizontalDivider(thickness = 1.dp, color = pc.borderLight)

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = pc.bgCard,
                border = BorderStroke(1.dp, pc.borderLight)
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("🚧", fontSize = 44.sp)
                    Spacer(Modifier.height(12.dp))
                    Text(
                        "শীঘ্রই আসছে",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = pc.textPrimary
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "বিশেষজ্ঞদের সাথে সরাসরি যোগাযোগের সুবিধা এখনো চালু হয়নি।",
                        fontSize = 13.sp,
                        color = pc.textSecondary,
                        textAlign = TextAlign.Center,
                        lineHeight = 19.sp
                    )
                }
            }

            Surface(
                shape = RoundedCornerShape(16.dp),
                color = pc.greenPale,
                border = BorderStroke(1.dp, pc.borderMid)
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    Text("👨‍🌾", fontSize = 26.sp)
                    Column {
                        Text(
                            "এখন কী করবেন?",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = pc.greenDark
                        )
                        Spacer(Modifier.height(3.dp))
                        Text(
                            "রোগ নিয়ে সন্দেহ থাকলে বা ওষুধ প্রয়োগের আগে আপনার এলাকার উপজেলা কৃষি অফিসার " +
                                "বা কৃষি সম্প্রসারণ অধিদপ্তরের স্থানীয় কর্মকর্তার পরামর্শ নিন।",
                            fontSize = 12.sp,
                            color = pc.textSecondary,
                            lineHeight = 18.sp
                        )
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}
