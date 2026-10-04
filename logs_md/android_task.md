# PaddyCare AI — Native Android App Tasks

## Phase 1: Model Conversion
- [x] Create `export_tflite.py` (PyTorch → ONNX → TFLite)
- [x] Create `models/labels.txt`

## Phase 2: Android Studio Project
- [x] Create project structure with Gradle config
- [x] Configure `AndroidManifest.xml`
- [x] Set up dependencies in `build.gradle.kts`

## Phase 3: Core App Code
- [x] `PaddyClassifier.kt` — TFLite inference engine
- [x] `ImagePreprocessor.kt` — Image prep + paddy leaf validation
- [x] `DiseaseInfo.kt` — Treatments data (EN + BN)
- [x] `PredictionResult.kt` — Data classes

## Phase 4: UI Screens
- [x] Theme (`Color.kt`, `Theme.kt`, `Type.kt`)
- [x] `HomeScreen.kt` — Main screen with camera/gallery
- [x] `ResultScreen.kt` — Disease results
- [x] `HistoryScreen.kt` — Past scans
- [x] `DiseaseCard.kt` — Result card component
- [x] `AppNavigation.kt` — Navigation setup
- [x] `MainActivity.kt` — Entry point

## Phase 5: Resources
- [x] `strings.xml` (English + Bengali)
- [x] `colors.xml`
- [x] App icon assets (Can be generated via Android Studio Image Asset Studio using your existing `icon.png`)

## Phase 6: Publishing Guide
- [x] `PUBLISH_GUIDE.md` — Step-by-step Play Store guide
