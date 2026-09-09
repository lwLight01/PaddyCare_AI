package com.paddycare.ai.ui.screens

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Eco
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
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
    onAnalyzeComplete: (PredictionResult, Uri) -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var tempCameraUri by remember { mutableStateOf<Uri?>(null) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isAnalyzing by remember { mutableStateOf(false) }

    // Prepare temp file for camera
    fun createTempFile(): Uri {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir = File(context.cacheDir, "camera_photos").apply { mkdirs() }
        val file = File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
        return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            imageUri = it
            bitmap = loadBitmap(context, it)
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        val uri = tempCameraUri

        if (success && uri != null) {
            imageUri = uri
            bitmap = loadBitmap(context, uri)
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            val uri = createTempFile()
            tempCameraUri = uri
            cameraLauncher.launch(uri)
        }
    }

    // ── Pulsing scan animation ──────────────────────────────────────────
    val infiniteTransition = rememberInfiniteTransition(label = "scan_pulse")
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.15f,
        targetValue = 0.45f,
        animationSpec = infiniteRepeatable(
            animation = tween(900, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_alpha"
    )

    // ── Floating bounce for hero icon ───────────────────────────────────
    val bounceOffset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = -8f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "bounce"
    )

    fun analyzeImage() {
        val currentBitmap = bitmap
        val currentUri = imageUri

        if (currentBitmap == null || currentUri == null) return

        isAnalyzing = true

        coroutineScope.launch {
            val result = withContext(Dispatchers.Default) {

                // 1. Validate image
                val (isPlant, _) =
                    ImagePreprocessor.checkPaddyImage(currentBitmap)

                if (!isPlant) {
                    return@withContext PredictionResult.NotPaddy(
                        context.getString(R.string.not_paddy_msg)
                    )
                }

                // 2. Classify
                val classifier = PaddyClassifier(context)
                val classifyResult = classifier.classify(currentBitmap)
                val predictions = classifyResult.predictions
                val embedding = classifyResult.embedding
                
                val entropy = classifier.entropy(
                    predictions.map { it.second }
                )
                val oodThreshold = classifier.entropyThreshold
                
                val topPrediction = predictions.first()
                val maxConf = topPrediction.second
                val topDisease = topPrediction.first

                // 2a. "Not Leaf" Class Gate
                if (topDisease == "Not Leaf") {
                    classifier.close()
                    return@withContext PredictionResult.NotPaddy(
                        DiseaseInfo.getTreatment("Not Leaf", "bn")
                    )
                }

                // 2b. Mahalanobis Distance Gate
                if (classifier.isOOD(embedding, topDisease)) {
                    classifier.close()
                    return@withContext PredictionResult.NotPaddy(
                        "🚫 এই ছবিটি ধান পাতার মতো দেখালেও, মডেল এটিকে পরিচিত ধান পাতা হিসেবে চিনতে পারছে না।\n" +
                        "অনুগ্রহ করে একটি স্পষ্ট ধান পাতার ছবি আপলোড করুন।"
                    )
                }
                
                classifier.close()

                // 3. Confidence/Entropy check
                if (
                    maxConf < PaddyClassifier.CONFIDENCE_THRESHOLD ||
                    entropy > oodThreshold
                ) {
                    return@withContext PredictionResult.LowConfidence(
                        context.getString(R.string.low_confidence_msg)
                    )
                }

                // 4. Build success result (excluding "Not Leaf")
                val resultList = predictions
                    .filter { it.first != "Not Leaf" }
                    .filterIndexed { index, pair ->
                        index == 0 || pair.second >= 0.10f
                    }
                    .take(2)
                    .map { (disease, conf) ->
                        Prediction(
                            disease = disease,
                            confidence = conf * 100f,
                            treatment = DiseaseInfo.getTreatment(
                                disease,
                                "bn"
                            )
                        )
                    }

                if (resultList.isEmpty()) {
                    return@withContext PredictionResult.LowConfidence(
                        context.getString(R.string.low_confidence_msg)
                    )
                }
                
                PredictionResult.Success(resultList)
            }

            isAnalyzing = false
            onAnalyzeComplete(result, currentUri)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bg)
            .verticalScroll(rememberScrollState())
    ) {
        // ═════════════════════════════════════════════════════════════════
        // GRADIENT HERO HEADER
        // ═════════════════════════════════════════════════════════════════
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(GradientStart, GradientEnd)
                    )
                )
                .padding(top = 56.dp, bottom = 32.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                // Floating paddy icon
                Box(
                    modifier = Modifier
                        .offset(y = bounceOffset.dp)
                        .size(64.dp)
                        .background(
                            Color.White.copy(alpha = 0.15f),
                            CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Filled.Eco,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(36.dp)
                    )
                }
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.hero_title),
                    color = Color.White,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.ExtraBold,
                    letterSpacing = (-0.5).sp
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = stringResource(R.string.hero_subtitle),
                    color = Color.White.copy(alpha = 0.8f),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }

        // ═════════════════════════════════════════════════════════════════
        // MAIN UPLOAD CARD
        // ═════════════════════════════════════════════════════════════════
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .offset(y = (-20).dp)
                .shadow(8.dp, RoundedCornerShape(20.dp))
                .clip(RoundedCornerShape(20.dp))
                .background(Surface)
                .padding(20.dp)
        ) {
            Text(
                text = stringResource(R.string.upload_title),
                color = GreenDark,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = stringResource(R.string.upload_hint),
                color = TextSecondary,
                fontSize = 14.sp,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // ── Image Preview ───────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(4f / 3f)
                    .clip(RoundedCornerShape(16.dp))
                    .background(GreenPale)
                    .then(
                        if (bitmap == null) {
                            Modifier.border(
                                2.dp,
                                GreenLight.copy(alpha = 0.5f),
                                RoundedCornerShape(16.dp)
                            )
                        } else Modifier
                    )
                    .clickable { galleryLauncher.launch("image/*") },
                contentAlignment = Alignment.Center
            ) {
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap!!.asImageBitmap(),
                        contentDescription = "Selected Leaf",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )

                    // Bottom gradient overlay on image
                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                            .height(60.dp)
                            .background(
                                Brush.verticalGradient(
                                    colors = listOf(
                                        Color.Transparent,
                                        Color.Black.copy(alpha = 0.3f)
                                    )
                                )
                            )
                    )

                    // Pulsing scan overlay
                    if (isAnalyzing) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .alpha(pulseAlpha)
                                .background(GreenMid)
                        )
                        // Scanning text
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = stringResource(R.string.analyzing),
                                color = Color.White,
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                } else {
                    // Empty state placeholder
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Icon(
                            imageVector = Icons.Filled.Eco,
                            contentDescription = null,
                            tint = GreenLight,
                            modifier = Modifier.size(56.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = stringResource(R.string.preview_hint),
                            color = GreenMid.copy(alpha = 0.7f),
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Gallery / Camera Buttons ────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedButton(
                    onClick = { galleryLauncher.launch("image/*") },
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = GreenDark
                    ),
                    border = ButtonDefaults.outlinedButtonBorder
                ) {
                    Icon(
                        imageVector = Icons.Filled.PhotoLibrary,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "গ্যালারি",
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 14.sp
                    )
                }

                Button(
                    onClick = {
                        cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = GreenDark,
                        contentColor = Color.White
                    ),
                    shape = RoundedCornerShape(12.dp),
                    elevation = ButtonDefaults.buttonElevation(
                        defaultElevation = 4.dp,
                        pressedElevation = 2.dp
                    )
                ) {
                    Icon(
                        imageVector = Icons.Filled.CameraAlt,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "ক্যামেরা",
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 14.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Detect Button ───────────────────────────────────────────
            Button(
                onClick = { analyzeImage() },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = bitmap != null && !isAnalyzing,
                colors = ButtonDefaults.buttonColors(
                    containerColor = GreenMid,
                    disabledContainerColor = GreenMid.copy(alpha = 0.4f)
                ),
                shape = RoundedCornerShape(14.dp),
                elevation = ButtonDefaults.buttonElevation(
                    defaultElevation = 6.dp,
                    pressedElevation = 2.dp
                )
            ) {
                if (isAnalyzing) {
                    CircularProgressIndicator(
                        color = Color.White,
                        modifier = Modifier.size(22.dp),
                        strokeWidth = 2.5.dp
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = stringResource(R.string.analyzing),
                        fontSize = 17.sp,
                        fontWeight = FontWeight.Bold
                    )
                } else {
                    Icon(
                        imageVector = Icons.Filled.Search,
                        contentDescription = null,
                        modifier = Modifier.size(22.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = stringResource(R.string.btn_detect),
                        fontSize = 17.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // ═════════════════════════════════════════════════════════════════
        // TIPS CARD
        // ═════════════════════════════════════════════════════════════════
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(bottom = 32.dp)
                .offset(y = (-8).dp)
                .shadow(2.dp, RoundedCornerShape(14.dp))
                .clip(RoundedCornerShape(14.dp))
                .background(WarningBg)
                .border(1.dp, WarningBorder, RoundedCornerShape(14.dp))
                .padding(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            Icon(
                imageVector = Icons.Filled.Info,
                contentDescription = null,
                tint = WarningTitle,
                modifier = Modifier
                    .size(24.dp)
                    .padding(top = 2.dp)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text(
                    text = stringResource(R.string.tips_title),
                    color = WarningTitle,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 6.dp)
                )
                Text(
                    text = "${stringResource(R.string.tip_1)}\n${stringResource(R.string.tip_2)}\n${stringResource(R.string.tip_3)}",
                    color = WarningText,
                    fontSize = 13.sp,
                    lineHeight = 20.sp
                )
            }
        }
    }
}

// Helper to load bitmap from URI
@Suppress("DEPRECATION")
private fun loadBitmap(context: android.content.Context, uri: Uri): Bitmap? {
    return try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}
