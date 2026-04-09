# Phase 1 Setup Guide — AI SDR Call Infrastructure

## Step 1: Install Dependencies
```bash
cd "AI SDR"
pip install -r requirements.txt
```

## Step 2: Set Up Ngrok (Expose local server to internet)
Telnyx needs a public URL to send webhooks to your local machine.

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000
```
Copy the HTTPS URL (e.g. https://abc123.ngrok.io)

## Step 3: Fill in .env
```
APP_BASE_URL=https://abc123.ngrok.io     ← your ngrok URL
TELNYX_API_KEY=your_telnyx_api_key_here
TELNYX_PHONE_NUMBER=+1XXXXXXXXXX         ← buy from Telnyx dashboard
TELNYX_CONNECTION_ID=XXXXXXXXXX          ← from Telnyx Call Control App
```

## Step 4: Telnyx Dashboard Setup
1. Go to https://portal.telnyx.com
2. **Buy a Phone Number** → Numbers → Search & Buy
3. **Create a Call Control App**:
   - Voice → Call Control → Create App
   - Webhook URL: https://your-ngrok-url/webhooks/telnyx
   - Copy the App ID → paste as TELNYX_CONNECTION_ID in .env
4. **Assign number to the app**:
   - Numbers → My Numbers → Edit → assign to your Call Control App

## Step 5: Run the Server
```bash
python main.py
```

## Step 6: Test a Call
```bash
curl -X POST http://localhost:8000/call/outbound \
  -H "Content-Type: application/json" \
  -d '{"to_number": "+1YOUR_TEST_NUMBER", "prospect_name": "Test"}'
```

## What Should Happen:
1. ✅ Your phone rings
2. ✅ You answer
3. ✅ AI says: "Hey! This is the AI SDR system. Phase 1 is working!"
4. ✅ Audio stream starts
5. ✅ Call logs appear in terminal

## Phase 1 Complete ✅ → Then we move to Phase 2: Voice Pipeline (Deepgram + Claude + Cartesia)
