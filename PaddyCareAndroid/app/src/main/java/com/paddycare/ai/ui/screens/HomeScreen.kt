package com.paddycare.ai.ui.screens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import com.paddycare.ai.PaddyCareApp
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.paddycare.ai.R
import com.paddycare.ai.data.DiseaseInfo
import com.paddycare.ai.data.Prediction
import com.paddycare.ai.data.PredictionResult
import com.paddycare.ai.ml.ImagePreprocessor
import com.paddycare.ai.ml.PaddyClassifier
import com.paddycare.ai.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun HomeScreen(
    launchCameraEvent: Boolean = false,
    onCameraLaunched: () -> Unit = {},
    onAnalyzeComplete: (PredictionResult, Uri) -> Unit
) {
    val context      = LocalContext.current
    val pc           = LocalPaddyCareColors.current
    val coroutineScope = rememberCoroutineScope()

    var imageUri       by remember { mutableStateOf<Uri?>(null) }
    var tempCameraUri  by remember { mutableStateOf<Uri?>(null) }
    var bitmap         by remember { mutableStateOf<Bitmap?>(null) }
    var isAnalyzing    by remember { mutableStateOf(false) }

    // Scanning animation
    val infiniteTransition = rememberInfiniteTransition(label = "scan")
    val scanLineY by infiniteTransition.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1800, easing = LinearEasing), RepeatMode.Reverse),
        label = "scan_line"
    )
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.05f, targetValue = 0.22f,
        animationSpec = infiniteRepeatable(tween(1100, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "pulse"
    )

    fun createTempFile(): Uri {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir   = File(context.cacheDir, "camera_photos").apply { mkdirs() }
        val file  = File.createTempFile("JPEG_${stamp}_", ".jpg", dir)
        return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    }

    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { imageUri = it; bitmap = loadBitmap(context, it) }
    }
    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success && tempCameraUri != null) { imageUri = tempCameraUri; bitmap = loadBitmap(context, tempCameraUri!!) }
    }
    val cameraPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) { val uri = createTempFile(); tempCameraUri = uri; cameraLauncher.launch(uri) }
    }

    LaunchedEffect(launchCameraEvent) {
        if (launchCameraEvent) {
            cameraPermLauncher.launch(android.Manifest.permission.CAMERA)
            onCameraLaunched()
        }
    }

    fun analyzeImage() {
        val bmp = bitmap ?: return
        val uri = imageUri ?: return
        isAnalyzing = true
        coroutineScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    val (isPlant, _) = ImagePreprocessor.checkPaddyImage(bmp)
                    if (!isPlant) return@withContext PredictionResult.NotPaddy(context.getString(R.string.not_paddy_msg))

                    val app = context.applicationContext as? PaddyCareApp
                    val classifier = app?.classifier ?: PaddyClassifier(context)
                    val lang = app?.settingsManager?.language?.value ?: "bn"

                    val classResult = classifier.classify(bmp)
                    val predictions = classResult.predictions
                    if (predictions.isEmpty()) {
                        return@withContext PredictionResult.Error("No prediction available from model.")
                    }

                    val entropy = classifier.entropy(predictions.map { it.second })
                    val topPred = predictions.first()

                    if (DiseaseInfo.isRejectClass(topPred.first)) {
                        return@withContext PredictionResult.NotPaddy(DiseaseInfo.getTreatment(topPred.first, lang))
                    }

                    if (classifier.isOOD(classResult.embedding, topPred.first)) {
                        val oodMsg = if (lang == "en") {
                            "This image resembles a paddy leaf, but the model cannot recognize it reliably.\nPlease provide a clearer photo of a paddy leaf."
                        } else {
                            "🚫 এই ছবিটি ধান পাতার মতো দেখালেও, মডেল এটিকে চিনতে পারছে না।\nঅনুগ্রহ করে একটি স্পষ্ট ধান পাতার ছবি দিন।"
                        }
                        return@withContext PredictionResult.NotPaddy(oodMsg)
                    }

                    if (topPred.second < PaddyClassifier.CONFIDENCE_THRESHOLD || entropy > classifier.entropyThreshold) {
                        return@withContext PredictionResult.LowConfidence(context.getString(R.string.low_confidence_msg))
                    }

                    val resultList = predictions.filter { !DiseaseInfo.isRejectClass(it.first) }
                        .filterIndexed { i, p -> i == 0 || p.second >= 0.10f }
                        .take(2)
                        .map { (disease, conf) ->
                            Prediction(
                                disease = disease,
                                confidence = conf * 100f,
                                treatment = DiseaseInfo.getTreatment(disease, lang)
                            )
                        }

                    if (resultList.isEmpty()) PredictionResult.LowConfidence(context.getString(R.string.low_confidence_msg))
                    else PredictionResult.Success(resultList)
                }
                onAnalyzeComplete(result, uri)
            } catch (e: Throwable) {
                Log.e("HomeScreen", "Inference error", e)
                onAnalyzeComplete(
                    PredictionResult.Error(e.localizedMessage ?: "Unexpected error during classification"),
                    uri
                )
            } finally {
                isAnalyzing = false
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(pc.bgPrimary)
            .verticalScroll(rememberScrollState())
    ) {
        // ═══════════════════════════════════════════════════════════════
        // HERO SECTION — green gradient with title
        // ═══════════════════════════════════════════════════════════════
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(210.dp)
        ) {
            // Green gradient background
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.linearGradient(
                            colors = if (pc.isDark)
                                listOf(Color(0xFF0B2A06), Color(0xFF0F3A0A), Color(0xFF0A2005))
                            else
                                listOf(Color(0xFF1A4A10), Color(0xFF2E7D1F), Color(0xFF1A5C10))
                        )
                    )
            )
            // Decorative leaf shapes
            Box(
                modifier = Modifier
                    .size(180.dp)
                    .offset(x = 220.dp, y = (-40).dp)
                    .background(Color.White.copy(alpha = 0.04f), CircleShape)
            )
            Box(
                modifier = Modifier
                    .size(120.dp)
                    .offset(x = 260.dp, y = 60.dp)
                    .background(Color.White.copy(alpha = 0.04f), CircleShape)
            )
            // Content
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(start = 24.dp, end = 24.dp, top = 56.dp, bottom = 24.dp),
                verticalArrangement = Arrangement.Bottom
            ) {
                // Badge
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = Color.White.copy(alpha = 0.15f),
                    modifier = Modifier.padding(bottom = 10.dp)
                ) {
                    Text(
                        text = "🌾  PaddyCare AI",
                        color = Color.White,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 5.dp)
                    )
                }
                Text(
                    text = "আমাদের ফসল\nসুস্থ রাখুন",
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                    lineHeight = 34.sp,
                    letterSpacing = (-0.5).sp
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    text = "AI প্রযুক্তিতে তাৎক্ষণিক রোগ নির্ণয়",
                    color = Color.White.copy(alpha = 0.78f),
                    fontSize = 13.sp
                )
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // STATS ROW
        // ═══════════════════════════════════════════════════════════════
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .offset(y = (-16).dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            StatMiniCard("📸", "স্ক্যান", "0", pc, Modifier.weight(1f))
            StatMiniCard("🦠", "রোগ", "0", pc, Modifier.weight(1f))
            StatMiniCard("✅", "সুস্থ", "0", pc, Modifier.weight(1f))
        }

        // ═══════════════════════════════════════════════════════════════
        // UPLOAD CARD
        // ═══════════════════════════════════════════════════════════════
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            shape = RoundedCornerShape(20.dp),
            color = pc.bgCard,
            shadowElevation = 2.dp,
            border = BorderStroke(1.dp, pc.borderLight)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {

                Text(
                    text = "📸  ছবি আপলোড করুন",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color = pc.textPrimary,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                Text(
                    text = "ধান পাতার স্পষ্ট ছবি তুলুন বা গ্যালারি থেকে বেছে নিন",
                    fontSize = 12.sp,
                    color = pc.textMuted,
                    modifier = Modifier.padding(bottom = 14.dp)
                )

                // ── Image preview zone ──────────────────────────────────
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(4f / 3f)
                        .clip(RoundedCornerShape(14.dp))
                        .border(
                            width = if (bitmap == null) 2.dp else 0.dp,
                            brush = Brush.linearGradient(listOf(pc.greenPrimary, GreenLight)),
                            shape = RoundedCornerShape(14.dp)
                        )
                        .background(if (bitmap == null) pc.bgTertiary else Color.Transparent)
                        .clickable { galleryLauncher.launch("image/*") },
                    contentAlignment = Alignment.Center
                ) {
                    if (bitmap != null) {
                        Image(
                            bitmap = bitmap!!.asImageBitmap(),
                            contentDescription = "Selected leaf",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(14.dp))
                        )
                        // Scan overlay
                        if (isAnalyzing) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(pc.greenPrimary.copy(alpha = pulseAlpha))
                                    .clip(RoundedCornerShape(14.dp))
                            )
                            // Scan line
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(3.dp)
                                    .offset(y = (scanLineY * 200 - 10).dp)
                                    .background(
                                        Brush.horizontalGradient(
                                            listOf(Color.Transparent, GreenBright, GreenLight, GreenBright, Color.Transparent)
                                        )
                                    )
                            )
                            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(44.dp), strokeWidth = 3.dp)
                            }
                        }
                    } else {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center,
                            modifier = Modifier.padding(24.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(64.dp)
                                    .background(pc.greenPale, CircleShape),
                                contentAlignment = Alignment.Center
                            ) {
                                Text("🌿", fontSize = 28.sp)
                            }
                            Spacer(Modifier.height(14.dp))
                            Text(
                                text = "এখানে ট্যাপ করুন",
                                color = pc.greenPrimary,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                text = "গ্যালারি বা ক্যামেরা থেকে ছবি বেছে নিন",
                                color = pc.textMuted,
                                fontSize = 12.sp,
                                textAlign = TextAlign.Center,
                                lineHeight = 17.sp
                            )
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))

                // ── Gallery / Camera buttons ────────────────────────────
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    OutlinedButton(
                        onClick = { galleryLauncher.launch("image/*") },
                        modifier = Modifier.weight(1f).height(46.dp),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.5.dp, pc.borderMid),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = pc.textPrimary)
                    ) {
                        Icon(Icons.Outlined.Image, null, modifier = Modifier.size(18.dp), tint = pc.textSecondary)
                        Spacer(Modifier.width(7.dp))
                        Text("গ্যালারি", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    }
                    Button(
                        onClick = { cameraPermLauncher.launch(android.Manifest.permission.CAMERA) },
                        modifier = Modifier.weight(1f).height(46.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = pc.greenPrimary,
                            contentColor = Color.White
                        ),
                        elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
                    ) {
                        Icon(Icons.Outlined.CameraAlt, null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(7.dp))
                        Text("ক্যামেরা", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    }
                }

                Spacer(Modifier.height(10.dp))

                // ── Analyze button ──────────────────────────────────────
                Button(
                    onClick = { analyzeImage() },
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    enabled = bitmap != null && !isAnalyzing,
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = pc.greenPrimary,
                        contentColor = Color.White,
                        disabledContainerColor = pc.greenPrimary.copy(alpha = 0.3f),
                        disabledContentColor = Color.White.copy(alpha = 0.5f)
                    ),
                    elevation = ButtonDefaults.buttonElevation(
                        defaultElevation = 4.dp, pressedElevation = 0.dp
                    )
                ) {
                    if (isAnalyzing) {
                        CircularProgressIndicator(color = Color.White, modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(10.dp))
                        Text("বিশ্লেষণ চলছে...", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    } else {
                        Icon(Icons.Outlined.Search, null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("রোগ সনাক্ত করুন", fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        Spacer(Modifier.height(20.dp))

        // ═══════════════════════════════════════════════════════════════
        // COMMON DISEASES SECTION
        // ═══════════════════════════════════════════════════════════════
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("🦠  সাধারণ রোগসমূহ", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = pc.textPrimary)
        }
        Spacer(Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            DiseaseQuickCard("Blast", "ব্লাস্ট রোগ", "🌾", "বিপজ্জনক", SeverityHigh, SeverityHighBg, pc, Modifier.weight(1f))
            DiseaseQuickCard("Brown Spot", "বাদামী দাগ", "🍂", "মাঝারি", SeverityMedium, SeverityMediumBg, pc, Modifier.weight(1f))
        }
        Spacer(Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            DiseaseQuickCard("Bacterial Blight", "ব্যাকটেরিয়াল ব্লাইট", "🔬", "বিপজ্জনক", SeverityHigh, SeverityHighBg, pc, Modifier.weight(1f))
            DiseaseQuickCard("Healthy", "সুস্থ গাছ", "✅", "সুস্থ", SeverityLow, SeverityLowBg, pc, Modifier.weight(1f))
        }

        Spacer(Modifier.height(20.dp))

        // ═══════════════════════════════════════════════════════════════
        // DAILY TIP
        // ═══════════════════════════════════════════════════════════════
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            shape = RoundedCornerShape(16.dp),
            color = pc.greenPale,
            border = BorderStroke(1.dp, pc.borderMid)
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(14.dp),
                verticalAlignment = Alignment.Top
            ) {
                Text("💡", fontSize = 26.sp)
                Column {
                    Text(
                        text = "সঠিক সময়ে স্প্রে করুন",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = pc.greenDark
                    )
                    Spacer(Modifier.height(3.dp))
                    Text(
                        text = "সকাল বা বিকেলে কীটনাশক স্প্রে করুন। তাপমাত্রা কম থাকলে ওষুধ বেশি কার্যকর হয়।",
                        fontSize = 12.sp,
                        color = pc.textSecondary,
                        lineHeight = 18.sp
                    )
                }
            }
        }

        Spacer(Modifier.height(20.dp))

        // ═══════════════════════════════════════════════════════════════
        // SCAN TIPS
        // ═══════════════════════════════════════════════════════════════
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            shape = RoundedCornerShape(16.dp),
            color = pc.bgCard,
            border = BorderStroke(1.dp, pc.borderLight)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.Info, null, tint = pc.greenPrimary, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("ভালো ফলাফলের জন্য", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = pc.textPrimary)
                }
                Spacer(Modifier.height(10.dp))
                listOf(
                    "পাতার স্পষ্ট ও পরিষ্কার ছবি তুলুন",
                    "ভালো আলোতে ছবি তোলার চেষ্টা করুন",
                    "রোগাক্রান্ত অংশটি ফোকাসে রাখুন",
                    "ক্যামেরা স্থির রাখুন, ঝাপসা এড়িয়ে চলুন"
                ).forEach { tip ->
                    Row(
                        modifier = Modifier.padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(20.dp)
                                .background(pc.greenPale, CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Text("✓", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = pc.greenPrimary)
                        }
                        Text(tip, fontSize = 12.sp, color = pc.textSecondary, lineHeight = 17.sp)
                    }
                }
            }
        }

        Spacer(Modifier.height(32.dp))
    }
}

@Composable
private fun StatMiniCard(
    icon: String, label: String, value: String,
    pc: PaddyCareColors, modifier: Modifier
) {
    Surface(
        modifier = modifier.shadow(3.dp, RoundedCornerShape(14.dp)),
        shape = RoundedCornerShape(14.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(icon, fontSize = 20.sp)
            Spacer(Modifier.height(4.dp))
            Text(value, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = pc.greenPrimary)
            Text(label, fontSize = 10.sp, color = pc.textMuted)
        }
    }
}

@Composable
private fun DiseaseQuickCard(
    name: String, nameBn: String, icon: String,
    severityLabel: String, severityColor: Color, severityBg: Color,
    pc: PaddyCareColors, modifier: Modifier
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(14.dp),
        color = pc.bgCard,
        border = BorderStroke(1.dp, pc.borderLight)
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(72.dp)
                    .background(pc.bgTertiary),
                contentAlignment = Alignment.Center
            ) {
                Text(icon, fontSize = 30.sp)
            }
            Column(modifier = Modifier.padding(10.dp)) {
                Text(name, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = pc.textPrimary)
                Text(nameBn, fontSize = 11.sp, color = pc.textMuted, modifier = Modifier.padding(top = 1.dp))
                Spacer(Modifier.height(6.dp))
                Surface(
                    shape = RoundedCornerShape(20.dp),
                    color = severityBg
                ) {
                    Text(
                        text = severityLabel,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        color = severityColor,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                    )
                }
            }
        }
    }
}

@Suppress("DEPRECATION")
private fun loadBitmap(context: android.content.Context, uri: Uri): Bitmap? = try {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        ImageDecoder.decodeBitmap(source) { dec, info, _ ->
            dec.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            dec.isMutableRequired = true
            // Scale down to max 1024px to prevent memory spikes on large camera captures
            val maxSide = maxOf(info.size.width, info.size.height)
            if (maxSide > 1024) {
                val scale = 1024f / maxSide
                val targetW = (info.size.width * scale).toInt().coerceAtLeast(456)
                val targetH = (info.size.height * scale).toInt().coerceAtLeast(456)
                dec.setTargetSize(targetW, targetH)
            }
        }
    } else {
        val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, boundsOptions)
        }
        val maxSide = maxOf(boundsOptions.outWidth, boundsOptions.outHeight)
        var sampleSize = 1
        while (maxSide / (sampleSize * 2) >= 1024) {
            sampleSize *= 2
        }
        val decodeOptions = BitmapFactory.Options().apply { inSampleSize = sampleSize }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, decodeOptions)
        }
    }
} catch (e: Exception) { e.printStackTrace(); null }
