# PaddyCare AI — Real Native Android App (Android Studio)

> **No Expo. No React Native. Pure native Android with Kotlin + Jetpack Compose + TensorFlow Lite.**

---

## Overview: What You're Building

```
┌─────────────────────────────────────────────────┐
│              Your Current System                │
│                                                 │
│  PyTorch Model (.pth)  →  Flask Server (PC)     │
│         ↓                      ↓                │
│  Expo App (phone)  ──HTTP──→  /predict          │
│                                                 │
│  ❌ Can't publish: needs your PC running        │
└─────────────────────────────────────────────────┘

                    ▼ ▼ ▼

┌─────────────────────────────────────────────────┐
│              What You'll Build                  │
│                                                 │
│  PyTorch Model → ONNX → TFLite (.tflite)       │
│         ↓                                       │
│  Native Android App (Kotlin)                    │
│    ├── TFLite model bundled inside APK          │
│    ├── Camera / Gallery image picker            │
│    ├── On-device inference (no internet needed) │
│    └── Results + treatments displayed           │
│                                                 │
│  ✅ Standalone APK — works offline anywhere     │
└─────────────────────────────────────────────────┘
```

---

## Prerequisites (Install These First)

| Tool | Version | Download |
|------|---------|----------|
| **Android Studio** | Latest (Ladybug+) | https://developer.android.com/studio |
| **JDK** | 17+ | Bundled with Android Studio |
| **Python** | 3.10+ | Already installed (for model conversion) |
| **Android SDK** | API 34+ | Install via Android Studio SDK Manager |
| **A real Android phone** | Android 7.0+ | For testing (emulator works too) |

---

## Phase 1 — Convert Your PyTorch Model to TFLite

This happens on your PC using Python. You must do this **before** touching Android Studio.

### Step 1.1: Train Your Model (if not done)

```bash
cd D:\Model_final\paddy_ai
python train.py
```
This produces `models/model.pth`.

### Step 1.2: Install Conversion Tools

```bash
pip install onnx onnxruntime onnx2tf tensorflow tf2onnx flatbuffers
```

### Step 1.3: Create the Conversion Script

Create a file `D:\Model_final\paddy_ai\export_tflite.py` with this logic:

1. **Load** your trained `model.pth` checkpoint
2. **Build** the MobileNetV2 model (same architecture as `model.py`)
3. **Export to ONNX**: `torch.onnx.export(model, dummy_input, "model.onnx")`
4. **Convert ONNX → TFLite**: Use `onnx2tf` or `tf.lite.TFLiteConverter`
5. **Quantize** (optional): Apply int8 quantization to shrink from ~14MB → ~3.5MB
6. **Validate**: Run a test image through the TFLite model and verify output matches PyTorch

Output: `models/model.tflite` (~3-14 MB depending on quantization)

### Step 1.4: Create Labels File

Create `models/labels.txt`:
```
Bacterial Blight
Blast
Healthy
```

> [!IMPORTANT]
> The order in `labels.txt` MUST match the order your model was trained with. Your `ImageFolder` dataset sorts alphabetically, so the order above is correct.

---

## Phase 2 — Create Android Studio Project

### Step 2.1: Create New Project

1. Open **Android Studio**
2. **File → New → New Project**
3. Select **"Empty Activity"** (Compose)
4. Configure:
   - **Name**: `PaddyCare AI`
   - **Package name**: `com.paddycare.ai` (or your own domain)
   - **Save location**: `D:\Model_final\paddy_ai\PaddyCareAndroid`
   - **Language**: **Kotlin**
   - **Minimum SDK**: API 24 (Android 7.0) — covers 99% of devices
   - **Build configuration language**: Kotlin DSL

### Step 2.2: Project Structure You'll Create

```
PaddyCareAndroid/
├── app/
│   ├── src/main/
│   │   ├── java/com/paddycare/ai/
│   │   │   ├── MainActivity.kt            ← Entry point
│   │   │   ├── ui/
│   │   │   │   ├── screens/
│   │   │   │   │   ├── HomeScreen.kt       ← Camera/Gallery + scan button
│   │   │   │   │   ├── ResultScreen.kt     ← Disease result + treatment
│   │   │   │   │   └── HistoryScreen.kt    ← Past scan history
│   │   │   │   ├── components/
│   │   │   │   │   ├── DiseaseCard.kt      ← Result card with confidence bar
│   │   │   │   │   └── WarningBanner.kt    ← Error/warning display
│   │   │   │   └── theme/
│   │   │   │       ├── Color.kt            ← Green/gold color palette
│   │   │   │       ├── Theme.kt            ← Material 3 theme
│   │   │   │       └── Type.kt             ← Typography
│   │   │   ├── ml/
│   │   │   │   ├── PaddyClassifier.kt      ← TFLite inference engine
│   │   │   │   └── ImagePreprocessor.kt    ← Resize, normalize images
│   │   │   ├── data/
│   │   │   │   ├── DiseaseInfo.kt          ← Treatments (EN + BN)
│   │   │   │   ├── PredictionResult.kt     ← Data classes
│   │   │   │   └── HistoryRepository.kt    ← Local storage (Room DB)
│   │   │   └── navigation/
│   │   │       └── AppNavigation.kt        ← Jetpack Navigation
│   │   ├── assets/
│   │   │   ├── model.tflite               ← Your converted model
│   │   │   └── labels.txt                 ← Class names
│   │   ├── res/
│   │   │   ├── drawable/                  ← Icons, backgrounds
│   │   │   ├── mipmap-xxxhdpi/            ← App icon (all sizes)
│   │   │   ├── values/
│   │   │   │   ├── strings.xml            ← Bengali + English strings
│   │   │   │   ├── colors.xml             ← Color definitions
│   │   │   │   └── themes.xml             ← App theme
│   │   │   └── values-bn/
│   │   │       └── strings.xml            ← Bengali translations
│   │   └── AndroidManifest.xml            ← Permissions (camera, storage)
│   └── build.gradle.kts                   ← Dependencies
├── build.gradle.kts                       ← Project-level config
└── gradle.properties
```

### Step 2.3: Add Dependencies to `app/build.gradle.kts`

```kotlin
dependencies {
    // Core Android + Compose
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("androidx.activity:activity-compose:1.9.1")
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")

    // Navigation
    implementation("androidx.navigation:navigation-compose:2.7.7")

    // TensorFlow Lite (THE KEY DEPENDENCY)
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // Camera + Image picking
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
    implementation("io.coil-kt:coil-compose:2.7.0")  // Image loading

    // Local database for history
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")

    // Permissions
    implementation("com.google.accompanist:accompanist-permissions:0.34.0")
}
```

### Step 2.4: AndroidManifest.xml — Permissions

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />

<uses-feature android:name="android.hardware.camera" android:required="false" />
```

---

## Phase 3 — Core App Code (What Each File Does)

### 3.1: `PaddyClassifier.kt` — The Brain (TFLite Inference)

This is the most critical file. It:
1. Loads `model.tflite` from the `assets/` folder
2. Takes a `Bitmap` image as input
3. Resizes it to 224×224
4. Normalizes pixel values with ImageNet mean/std: `(pixel/255 - mean) / std`
5. Runs the TFLite interpreter
6. Applies softmax to get probabilities
7. Returns top predictions with confidence percentages

```
Input: Bitmap (any size photo from camera/gallery)
  ↓ Resize to 224×224
  ↓ Normalize (ImageNet stats)
  ↓ TFLite Interpreter.run()
  ↓ Softmax
Output: [("Blast", 87.3%), ("Bacterial Blight", 8.1%), ("Healthy", 4.6%)]
```

### 3.2: `HomeScreen.kt` — Main Screen

- Hero banner with app name and icon
- Image preview box (tap to select from gallery)
- Two buttons: 📁 Gallery | 📷 Camera
- "🔍 রোগ সনাক্ত করুন" (Detect Disease) button
- Scanning animation overlay while processing
- Usage tips at the bottom

### 3.3: `ResultScreen.kt` — Results Display

- Shows the analyzed leaf image
- Disease cards with confidence bars
- Expandable treatment sections (Bengali + English)
- "Scan Again" button
- Warning banners for low confidence / non-paddy images

### 3.4: `DiseaseInfo.kt` — Treatment Data

Hardcode all treatment info (same as your Python `config.py` TREATMENTS dict):
- Blast → treatment in EN + BN
- Bacterial Blight → treatment in EN + BN
- Healthy → message in EN + BN

### 3.5: `ImagePreprocessor.kt` — Paddy Leaf Validator

Port your `check_paddy_image()` logic from `predict.py`:
- HSV color analysis to reject non-paddy images
- Edge density check
- Returns pass/fail before running the model

### 3.6: `HistoryRepository.kt` — Local Storage

Use **Room Database** to store past scans:
- Timestamp, image path, disease name, confidence, treatment shown

---

## Phase 4 — Build & Test

### Step 4.1: Run on Emulator
1. In Android Studio: **Run → Run 'app'**
2. Select an emulator or connected device
3. Test the full flow: pick image → analyze → view result

### Step 4.2: Run on Real Phone
1. Enable **Developer Options** on your phone (tap Build Number 7 times)
2. Enable **USB Debugging**
3. Connect phone via USB
4. Android Studio will detect it → click Run

### Step 4.3: Test Checklist
- [ ] Camera capture works
- [ ] Gallery selection works
- [ ] Model loads without crash
- [ ] Correct disease detected on known test images
- [ ] Non-paddy images rejected properly
- [ ] Low confidence warning shows for unclear images
- [ ] History saves and displays correctly
- [ ] App works in airplane mode (offline)
- [ ] Bengali text renders correctly
- [ ] Back navigation works properly

---

## Phase 5 — Build Signed APK / AAB

### Step 5.1: Generate Signing Key

```bash
keytool -genkey -v -keystore paddycare-release-key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias paddycare
```

> [!CAUTION]
> **NEVER lose this keystore file or its password.** You cannot update your Play Store app without it. Back it up to multiple safe locations.

### Step 5.2: Configure Signing in `build.gradle.kts`

```kotlin
android {
    signingConfigs {
        create("release") {
            storeFile = file("paddycare-release-key.jks")
            storePassword = "your_password"
            keyAlias = "paddycare"
            keyPassword = "your_password"
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = true       // Shrinks APK size
            isShrinkResources = true     // Removes unused resources
            proguardFiles(...)
            signingConfig = signingConfigs.getByName("release")
        }
    }
}
```

### Step 5.3: Build

**For testing (APK):**
- Android Studio → **Build → Build Bundle(s) / APK(s) → Build APK(s)**
- Output: `app/build/outputs/apk/release/app-release.apk`
- Share this APK directly to anyone for testing

**For Play Store (AAB — required by Google):**
- Android Studio → **Build → Generate Signed Bundle / APK**
- Select **Android App Bundle**
- Sign with your release key
- Output: `app/build/outputs/bundle/release/app-release.aab`

---

## Phase 6 — Publish to Google Play Store

### Step 6.1: Create Developer Account

1. Go to https://play.google.com/console
2. Sign in with Google account
3. Pay **$25 USD** one-time fee
4. Verify your identity (may take 48 hours)

### Step 6.2: Create App

1. Click **"Create app"**
2. Fill in:
   - **App name**: `PaddyCare AI - ধান পাতার রোগ সনাক্তকরণ`
   - **Default language**: Bengali (bn)
   - **App type**: App
   - **Free / Paid**: Free
   - **Category**: Tools or Education

### Step 6.3: Store Listing (Content You Need)

| Item | Requirement |
|------|-------------|
| **App Icon** | 512 × 512 px, PNG, 32-bit (no alpha for Play Store) |
| **Feature Graphic** | 1024 × 500 px, PNG or JPEG |
| **Screenshots** | Min 2, max 8 per device type. Phone: min 320px, max 3840px |
| **Short Description** | Max 80 characters: `ধান পাতার রোগ সনাক্ত করুন - AI দিয়ে তাৎক্ষণিক ফলাফল পান` |
| **Full Description** | Max 4000 characters: describe features, how it works, etc. |
| **Privacy Policy URL** | **Required**. Host a simple page (GitHub Pages works free) |

### Step 6.4: Content Rating

- Fill out the IARC questionnaire (takes 5 minutes)
- Your app will likely get **"Everyone"** rating

### Step 6.5: Target Audience

- Select **18+ only** (simplest) or specify age groups
- Since this is a farming tool, "Everyone" is appropriate

### Step 6.6: Upload AAB & Release

1. Go to **Production → Create new release**
2. Let Google manage signing (recommended) or upload your own key
3. Upload the `.aab` file
4. Add release notes: `v1.0.0 - প্রথম রিলিজ | First Release`
5. Click **"Review release"** → **"Start rollout to production"**

### Step 6.7: Review Timeline

- **First submission**: 3-7 days (sometimes up to 14 days)
- **Updates**: Usually 1-3 days
- Google may ask for clarification on permissions (camera usage)

---

## Phase 7 — Privacy Policy (Required)

Create a simple privacy policy page. You can host it free on GitHub Pages.

Key points to include:
- App processes images **on-device only** — no images are uploaded to any server
- No personal data is collected
- Camera and gallery permissions are used solely for leaf disease detection
- History is stored locally on the device only
- No third-party analytics or advertising SDKs

---

## Cost Summary

| Item | Cost | Notes |
|------|------|-------|
| Android Studio | **Free** | |
| TensorFlow Lite | **Free** | Open source |
| Google Play Developer | **$25** | One-time, lifetime |
| Hosting/Server | **$0** | On-device inference, no backend |
| Privacy Policy (GitHub Pages) | **$0** | Free hosting |
| **Total** | **$25** | |

---

## Estimated Timeline

| Phase | Time | Description |
|-------|------|-------------|
| Phase 1 | 1 day | Model conversion (PyTorch → TFLite) |
| Phase 2 | 1 day | Android Studio project setup |
| Phase 3 | 3-5 days | Core app development (all screens + inference) |
| Phase 4 | 1 day | Testing on real device |
| Phase 5 | 1 hour | Build signed APK/AAB |
| Phase 6 | 1-2 days | Play Store listing + submission |
| Phase 7 | 1 hour | Privacy policy page |
| **Review wait** | **3-7 days** | Google review period |
| **Total** | **~7-10 days + review** | |

---

## Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| **Language** | Kotlin |
| **UI Framework** | Jetpack Compose (Material 3) |
| **AI Inference** | TensorFlow Lite 2.16 |
| **Image Loading** | Coil |
| **Camera** | CameraX |
| **Image Picking** | ActivityResultContracts |
| **Local Database** | Room |
| **Navigation** | Jetpack Navigation Compose |
| **Build System** | Gradle (Kotlin DSL) |
| **Min Android** | API 24 (Android 7.0) |
| **Target Android** | API 34 (Android 14) |

---

> [!TIP]
> ## Quick Start: What to Do Right Now
> 1. **Install Android Studio** if you haven't already
> 2. **Train your model** (`python train.py`) if not done
> 3. Tell me to proceed and I will create the model conversion script (`export_tflite.py`) first, then build the full Android app project
