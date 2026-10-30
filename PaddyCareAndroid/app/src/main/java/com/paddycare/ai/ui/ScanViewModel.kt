package com.paddycare.ai.ui

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.paddycare.ai.PaddyCareApp
import com.paddycare.ai.R
import com.paddycare.ai.data.DiseaseInfo
import com.paddycare.ai.data.Prediction
import com.paddycare.ai.data.PredictionResult
import com.paddycare.ai.data.ScanRecord
import com.paddycare.ai.ml.ImagePreprocessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * ViewModel owning leaf inference, result state across configuration changes / process death,
 * and scan history persistence + pruning.
 */
class ScanViewModel(
    application: Application,
    private val savedStateHandle: SavedStateHandle
) : AndroidViewModel(application) {

    private val app: PaddyCareApp
        get() = getApplication()

    val scans: StateFlow<List<ScanRecord>> = app.database.scanDao().getAllScans()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    private val _currentResult = MutableStateFlow<PredictionResult?>(restoreResult())
    val currentResult: StateFlow<PredictionResult?> = _currentResult.asStateFlow()

    private val _currentImageUri = MutableStateFlow<Uri?>(
        savedStateHandle.get<String>("image_uri")?.let { Uri.parse(it) }
    )
    val currentImageUri: StateFlow<Uri?> = _currentImageUri.asStateFlow()

    fun analyzeImage(bitmap: Bitmap, fallbackUri: Uri, onComplete: () -> Unit) {
        if (_isAnalyzing.value) return
        _isAnalyzing.value = true

        viewModelScope.launch {
            val (result, scanUri) = try {
                withContext(Dispatchers.Default) {
                    val classifier = app.classifier
                    val lang = app.settingsManager.language.value

                    val (isPlant, _) = ImagePreprocessor.checkPaddyImage(bitmap)
                    if (!isPlant) {
                        val saved = saveScanBitmap(app, bitmap)
                        return@withContext Pair(
                            PredictionResult.NotPaddy(app.getString(R.string.not_paddy_msg)),
                            saved
                        )
                    }

                    val classResult = classifier.classify(bitmap)
                    val predictions = classResult.predictions
                    if (predictions.isEmpty()) {
                        val saved = saveScanBitmap(app, bitmap)
                        return@withContext Pair(
                            PredictionResult.Error("No prediction available from model."),
                            saved
                        )
                    }

                    val entropy = classifier.entropy(predictions.map { it.second })
                    val topPred = predictions.first()
                    val saved = saveScanBitmap(app, bitmap)

                    if (DiseaseInfo.isRejectClass(topPred.first)) {
                        return@withContext Pair(
                            PredictionResult.NotPaddy(DiseaseInfo.getTreatment(topPred.first, lang)),
                            saved
                        )
                    }

                    if (classifier.isOOD(classResult.embedding, topPred.first)) {
                        val oodMsg = if (lang == "en") {
                            "This image resembles a paddy leaf, but the model cannot recognize it reliably.\nPlease provide a clearer photo of a paddy leaf."
                        } else {
                            "🚫 এই ছবিটি ধান পাতার মতো দেখালেও, মডেল এটিকে চিনতে পারছে না।\nঅনুগ্রহ করে একটি স্পষ্ট ধান পাতার ছবি দিন।"
                        }
                        return@withContext Pair(PredictionResult.NotPaddy(oodMsg), saved)
                    }

                    if (topPred.second < classifier.confidenceThreshold || entropy > classifier.entropyThreshold) {
                        return@withContext Pair(
                            PredictionResult.LowConfidence(app.getString(R.string.low_confidence_msg)),
                            saved
                        )
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

                    val finalResult = if (resultList.isEmpty()) {
                        PredictionResult.LowConfidence(app.getString(R.string.low_confidence_msg))
                    } else {
                        PredictionResult.Success(resultList)
                    }
                    Pair(finalResult, saved)
                }
            } catch (e: Throwable) {
                Log.e("ScanViewModel", "Inference error", e)
                val saved = withContext(Dispatchers.IO) { saveScanBitmap(app, bitmap) }
                Pair(
                    PredictionResult.Error(e.localizedMessage ?: "Unexpected error during classification"),
                    if (saved != Uri.EMPTY) saved else fallbackUri
                )
            } finally {
                _isAnalyzing.value = false
            }

            setResult(result, scanUri)
            withContext(Dispatchers.IO) {
                persistAndPruneHistory(result, scanUri)
            }
            onComplete()
        }
    }

    private fun setResult(result: PredictionResult, uri: Uri) {
        _currentResult.value = result
        _currentImageUri.value = uri
        savedStateHandle["image_uri"] = uri.toString()
        val serialized: ArrayList<String> = when (result) {
            is PredictionResult.Success -> arrayListOf("ok").apply {
                result.predictions.forEach {
                    add(it.disease)
                    add(it.confidence.toString())
                    add(it.treatment)
                }
            }
            is PredictionResult.NotPaddy      -> arrayListOf("not_paddy", result.message)
            is PredictionResult.LowConfidence -> arrayListOf("low_confidence", result.message)
            is PredictionResult.Error         -> arrayListOf("error", result.message)
        }
        savedStateHandle["result_state"] = serialized
    }

    private fun restoreResult(): PredictionResult? {
        val l = savedStateHandle.get<ArrayList<String>>("result_state") ?: return null
        if (l.isEmpty()) return null
        return when (l[0]) {
            "ok" -> PredictionResult.Success(
                l.drop(1).chunked(3).mapNotNull { chunk ->
                    if (chunk.size == 3) {
                        Prediction(chunk[0], chunk[1].toFloatOrNull() ?: 0f, chunk[2])
                    } else null
                }
            )
            "not_paddy"      -> PredictionResult.NotPaddy(l.getOrElse(1) { "" })
            "low_confidence" -> PredictionResult.LowConfidence(l.getOrElse(1) { "" })
            "error"          -> PredictionResult.Error(l.getOrElse(1) { "" })
            else             -> null
        }
    }

    private suspend fun persistAndPruneHistory(result: PredictionResult, uri: Uri) {
        val dao = app.database.scanDao()
        val savedPath = uri.toString()
        val record = when (result) {
            is PredictionResult.Success -> {
                val top = result.predictions.first()
                ScanRecord(
                    imagePath     = savedPath,
                    status        = "ok",
                    diseaseName   = top.disease,
                    diseaseNameBn = top.diseaseBn,
                    confidence    = top.confidence,
                    treatment     = top.treatment
                )
            }
            is PredictionResult.NotPaddy      -> ScanRecord(imagePath = savedPath, status = "not_paddy", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
            is PredictionResult.LowConfidence -> ScanRecord(imagePath = savedPath, status = "low_confidence", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
            is PredictionResult.Error         -> ScanRecord(imagePath = savedPath, status = "error", diseaseName = null, diseaseNameBn = null, confidence = null, treatment = null)
        }
        dao.insert(record)

        try {
            val overflowPaths = dao.getOverflowImagePaths(50)
            dao.deleteOverflowRecords(50)
            for (path in overflowPaths) {
                val fileUri = Uri.parse(path)
                if (fileUri.scheme == "file") {
                    val file = File(fileUri.path ?: "")
                    if (file.exists()) file.delete()
                }
            }
        } catch (_: Exception) {}
    }

    private fun saveScanBitmap(context: Context, bitmap: Bitmap): Uri {
        return try {
            val scansDir = File(context.filesDir, "scans").apply { mkdirs() }
            val scanFile = File(scansDir, "scan_${System.currentTimeMillis()}.jpg")
            scanFile.outputStream().use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
            Uri.fromFile(scanFile)
        } catch (e: Exception) {
            Uri.EMPTY
        }
    }
}

