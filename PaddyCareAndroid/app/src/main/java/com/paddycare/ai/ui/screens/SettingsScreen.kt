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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.isSystemInDarkTheme
import com.paddycare.ai.ui.theme.*

@Composable
fun SettingsScreen() {
    val pc          = LocalPaddyCareColors.current
    val context = LocalContext.current
    val settingsManager = (context.applicationContext as com.paddycare.ai.PaddyCareApp).settingsManager
    
    val darkMode by settingsManager.darkMode.collectAsState()
    val notifOn by settingsManager.notifOn.collectAsState()
    val emailOn by settingsManager.emailOn.collectAsState()

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
                Text("সেটিং", color = pc.textPrimary, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text("অ্যাপ কাস্টমাইজ করুন", color = pc.textMuted, fontSize = 12.sp)
            }
        }
        HorizontalDivider(thickness = 1.dp, color = pc.borderLight)

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            // ── Appearance ────────────────────────────────────────────
            SettingsSectionLabel("চেহারা", pc)
            SettingsCard(pc) {
                SettingsToggleRow(
                    icon = "🌙",
                    iconBg = pc.greenPale,
                    title = "ডার্ক মোড",
                    subtitle = "অন্ধকার থিম ব্যবহার করুন",
                    checked = darkMode,
                    pc = pc
                ) { settingsManager.setDarkMode(it) }
                HorizontalDivider(thickness = 1.dp, color = pc.divider)
                SettingsNavRow(Icons.Outlined.Language, Color(0xFFDBEAFE), "ভাষা", "বাংলা (বর্তমান)", pc)
            }

            Spacer(Modifier.height(4.dp))

            // ── Notifications ─────────────────────────────────────────
            SettingsSectionLabel("বিজ্ঞপ্তি", pc)
            SettingsCard(pc) {
                SettingsToggleRow("🔔", Color(0xFFFFEDD5), "পুশ নোটিফিকেশন", "নতুন রোগের সতর্কতা", notifOn, pc) { settingsManager.setNotifOn(it) }
                HorizontalDivider(thickness = 1.dp, color = pc.divider)
                SettingsToggleRow("📧", pc.greenPale, "রিপোর্ট ইমেইল", "সাপ্তাহিক রিপোর্ট পান", emailOn, pc) { settingsManager.setEmailOn(it) }
            }

            Spacer(Modifier.height(4.dp))

            // ── Data ──────────────────────────────────────────────────
            SettingsSectionLabel("ডেটা", pc)
            SettingsCard(pc) {
                SettingsNavRow(Icons.Outlined.Delete, Color(0xFFFEE2E2), "ইতিহাস মুছুন", "সমস্ত স্ক্যান ডেটা মুছে ফেলুন", pc)
                HorizontalDivider(thickness = 1.dp, color = pc.divider)
                SettingsNavRow(Icons.Outlined.Cloud, Color(0xFFDBEAFE), "ব্যাকআপ", "ক্লাউডে সেভ করুন", pc)
            }

            Spacer(Modifier.height(4.dp))

            // ── About ─────────────────────────────────────────────────
            SettingsSectionLabel("অ্যাপ সম্পর্কে", pc)
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = pc.greenPale,
                border = BorderStroke(1.dp, pc.borderMid)
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth().padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("🌾", fontSize = 40.sp)
                    Spacer(Modifier.height(8.dp))
                    Text("PaddyCare AI", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = pc.textPrimary)
                    Text("সংস্করণ ২.০.১", fontSize = 12.sp, color = pc.textMuted)
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text = "AI প্রযুক্তি ব্যবহার করে ধান পাতার রোগ সনাক্তকরণ ও চিকিৎসা পরামর্শ প্রদান করে। বাংলাদেশের কৃষকদের জন্য তৈরি।",
                        fontSize = 12.sp,
                        color = pc.textSecondary,
                        lineHeight = 18.sp,
                        modifier = Modifier.padding(horizontal = 8.dp)
                    )
                }
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun SettingsSectionLabel(label: String, pc: PaddyCareColors) {
    Text(
        text = label.uppercase(),
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        color = pc.textMuted,
        letterSpacing = 0.8.sp,
        modifier = Modifier.padding(horizontal = 4.dp, vertical = 4.dp)
    )
}

@Composable
private fun SettingsCard(pc: PaddyCareColors, content: @Composable ColumnScope.() -> Unit) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight),
        shadowElevation = 1.dp
    ) {
        Column(content = content)
    }
}

@Composable
private fun SettingsToggleRow(
    icon: String, iconBg: Color,
    title: String, subtitle: String,
    checked: Boolean, pc: PaddyCareColors,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onCheckedChange(!checked) }
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Box(
            modifier = Modifier.size(38.dp).background(iconBg, RoundedCornerShape(10.dp)),
            contentAlignment = Alignment.Center
        ) { Text(icon, fontSize = 18.sp) }
        Column(modifier = Modifier.weight(1f)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = pc.textPrimary)
            Text(subtitle, fontSize = 11.sp, color = pc.textMuted)
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor       = Color.White,
                checkedTrackColor       = pc.greenPrimary,
                uncheckedThumbColor     = Color.White,
                uncheckedTrackColor     = pc.borderMid
            )
        )
    }
}

@Composable
private fun SettingsNavRow(
    icon: ImageVector, iconBg: Color,
    title: String, subtitle: String,
    pc: PaddyCareColors
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { }
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Box(
            modifier = Modifier.size(38.dp).background(iconBg, RoundedCornerShape(10.dp)),
            contentAlignment = Alignment.Center
        ) { Icon(icon, null, tint = pc.textSecondary, modifier = Modifier.size(18.dp)) }
        Column(modifier = Modifier.weight(1f)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = pc.textPrimary)
            Text(subtitle, fontSize = 11.sp, color = pc.textMuted)
        }
        Icon(Icons.Outlined.ChevronRight, null, tint = pc.textMuted, modifier = Modifier.size(18.dp))
    }
}
