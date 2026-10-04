# Google Play Store Publishing Guide

This guide walks you through publishing the **PaddyCare AI** native Android app to the Google Play Store.

## 1. Create a Google Play Developer Account
1. Go to the [Google Play Console](https://play.google.com/console/).
2. Sign in with your Google account.
3. Pay the **$25 USD** one-time registration fee.
4. Verify your identity with a government-issued ID (NID or Passport). It may take up to 48 hours to be approved.

## 2. Build the Release AAB (Android App Bundle)
Google Play requires an `.aab` file (not an `.apk`).

1. Open your project in Android Studio.
2. Go to **Build** -> **Generate Signed Bundle / APK...**
3. Select **Android App Bundle** and click Next.
4. **Create a new key store path**:
   - Path: Choose a safe folder on your PC (e.g., `D:\PaddyCareKeys\release-key.jks`)
   - Password: Make it strong and **save it somewhere safe**.
   - Alias: `paddycare`
   - Key password: Can be the same as the store password.
   - Validity (years): 25
   - Certificate: Fill in your name, organization, etc.
   > ⚠️ **CRITICAL:** Do NOT lose this `.jks` file or its passwords. You will need it to publish future updates to your app.
5. Choose the **release** build variant and click Finish.
6. Android Studio will build the bundle. You can find it at `PaddyCareAndroid/app/release/app-release.aab`.

## 3. Create the App in Play Console
1. In the Play Console, click **Create app**.
2. **App name**: PaddyCare AI - ধান পাতার রোগ সনাক্তকরণ
3. **Default language**: Bengali (bn-BD)
4. **App or game**: App
5. **Free or paid**: Free
6. Accept the Developer Program Policies and US export laws, then click **Create app**.

## 4. Prepare Store Listing Assets
You will need the following graphics ready:
- **App Icon**: 512x512 pixels (PNG).
- **Feature Graphic**: 1024x500 pixels (PNG or JPG).
- **Screenshots**: Take at least 2 screenshots of the app running on a phone (can be from an emulator).

## 5. Setup the Store Listing
In the left menu, go to **Store presence** -> **Main store listing**.
1. **Short description** (max 80 chars): ধান পাতার রোগ সনাক্ত করুন - AI দিয়ে তাৎক্ষণিক ফলাফল পান।
2. **Full description**: Explain what the app does, how to use it, and that it works offline.
3. Upload your Icon, Feature Graphic, and Screenshots.
4. Click **Save**.

## 6. App Content Questionnaire
In the left menu, go to **Policy** -> **App content**. You must complete all tasks here:
1. **Privacy Policy**: You must provide a link to a privacy policy. Since the app doesn't collect data, you can use a free privacy policy generator and host it on Google Sites or GitHub Pages. The policy should state that camera and gallery access are used solely for local image processing and no images are uploaded to any server.
2. **Ads**: Select "My app does not contain ads".
3. **App access**: Select "All functionality is available without special access".
4. **Content rating**: Fill out the questionnaire. Your app should get an "Everyone" rating.
5. **Target audience**: Select 18 and over (to keep things simple regarding COPPA laws).
6. **Data safety**: Answer the questions stating that you do not collect or share any user data.

## 7. Upload the AAB and Release
1. Go to **Release** -> **Production**.
2. Click **Create new release**.
3. Under "App integrity", Google Play App Signing should be enabled.
4. Click **Upload** and select your `app-release.aab` file.
5. Add release notes: `v1.0.0 - প্রথম রিলিজ | First Release`
6. Click **Next**, review the details, and click **Save**.
7. Go to the "Publishing overview" page and click **Send x changes for review**.

## 8. Wait for Review
Google will review your app. For a first-time developer account, this can take anywhere from 3 to 7 days. You will receive an email once your app is live on the Play Store!
