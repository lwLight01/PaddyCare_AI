package com.paddycare.ai.ui.screens

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.paddycare.ai.ui.theme.*

data class Expert(
    val name: String,
    val specialty: String,
    val org: String,
    val emoji: String
)

@Composable
fun ExpertsScreen() {
    val pc = LocalPaddyCareColors.current

    val experts = listOf(
        Expert("ড. রহিম উদ্দিন", "ধান রোগ বিশেষজ্ঞ", "বাংলাদেশ কৃষি গবেষণা ইনস্টিটিউট", "👨‍🔬"),
        Expert("ড. নাসরিন সুলতানা", "উদ্ভিদ রোগতত্ত্ব", "বাংলাদেশ কৃষি বিশ্ববিদ্যালয়", "👩‍🌾"),
        Expert("কৃষিবিদ করিম", "কীটনাশক পরামর্শদাতা", "জেলা কৃষি সম্প্রসারণ অধিদপ্তর", "👨‍💼"),
        Expert("প্রফেসর ফারিদা বেগম", "কৃষি জৈব প্রযুক্তি", "শেরে বাংলা কৃষি বিশ্ববিদ্যালয়", "👩‍🔬"),
        Expert("মো. জামাল হোসেন", "জৈব কৃষি বিশেষজ্ঞ", "কৃষি সম্প্রসারণ অধিদপ্তর, ঢাকা", "👨‍🌾"),
    )

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
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text("বিশেষজ্ঞ", color = pc.textPrimary, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text("কৃষি পরামর্শদাতা", color = pc.textMuted, fontSize = 12.sp)
            }
            Box(
                modifier = Modifier.size(36.dp).background(pc.bgTertiary, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Outlined.Search, null, tint = pc.textSecondary, modifier = Modifier.size(18.dp))
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
            // Banner
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Brush.linearGradient(listOf(pc.greenPrimary, pc.greenDark)),
                        RoundedCornerShape(16.dp)
                    )
                    .padding(18.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Text("👨‍🌾", fontSize = 32.sp)
                    Column {
                        Text("বিশেষজ্ঞের সাথে কথা বলুন", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "অভিজ্ঞ কৃষি বিশেষজ্ঞদের সাথে সরাসরি যোগাযোগ করুন",
                            color = Color.White.copy(0.80f), fontSize = 12.sp, lineHeight = 17.sp
                        )
                    }
                }
            }

            Text(
                "কৃষি গবেষক ও পরামর্শদাতা",
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
                color = pc.textMuted
            )

            experts.forEach { expert ->
                ExpertCard(expert, pc)
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ExpertCard(expert: Expert, pc: PaddyCareColors) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight),
        shadowElevation = 1.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(52.dp)
                    .background(
                        Brush.linearGradient(listOf(GreenLight, GreenPrimary)),
                        CircleShape
                    )
                    .border(2.dp, pc.borderMid, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Text(expert.emoji, fontSize = 22.sp)
            }

            // Info
            Column(modifier = Modifier.weight(1f)) {
                Text(expert.name, fontSize = 14.sp, fontWeight = FontWeight.Bold, color = pc.textPrimary)
                Text(expert.specialty, fontSize = 12.sp, color = pc.greenPrimary, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(2.dp))
                Text(expert.org, fontSize = 11.sp, color = pc.textMuted, lineHeight = 15.sp)
            }

            // Action buttons
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                IconButton(
                    onClick = {},
                    modifier = Modifier
                        .size(36.dp)
                        .background(SeverityLowBg, CircleShape)
                ) {
                    Text("📞", fontSize = 14.sp)
                }
                IconButton(
                    onClick = {},
                    modifier = Modifier
                        .size(36.dp)
                        .background(Color(0xFFDBEAFE), CircleShape)
                ) {
                    Text("💬", fontSize = 14.sp)
                }
            }
        }
    }
}
