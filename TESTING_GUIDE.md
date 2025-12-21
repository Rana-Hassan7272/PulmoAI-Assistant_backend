# FastAPI Testing Guide

This guide shows you how to test the Doctor Assistant backend using FastAPI's built-in interactive documentation.

## 🚀 Starting the Server

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Activate virtual environment:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Run the FastAPI server:**
```bash
uvicorn app.main:app --reload
```

The server will start at: `http://localhost:8000`

## 📚 Accessing API Documentation

FastAPI automatically generates interactive API documentation:

1. **Swagger UI** (Recommended): 
   - Open browser: `http://localhost:8000/docs`
   - Interactive interface with "Try it out" buttons

2. **ReDoc** (Alternative):
   - Open browser: `http://localhost:8000/redoc`
   - Clean, readable documentation

## 🧪 Testing Workflow

### Step 1: Create User Account

**Endpoint:** `POST /auth/signup`

1. Go to `http://localhost:8000/docs`
2. Find `/auth/signup` endpoint
3. Click "Try it out"
4. Enter test data:
```json
{
  "username": "testuser",
  "email": "test@example.com",
  "password": "testpass123",
  "patient_name": "Test Patient",
  "patient_age": 25,
  "patient_gender": "Male"
}
```
5. Click "Execute"
6. **Save the response** - You'll get a JWT token like:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Step 2: Authorize in Swagger UI

1. In Swagger UI, click the **"Authorize"** button (top right, lock icon)
2. Enter your token: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (the full token from Step 1)
3. Click "Authorize"
4. Click "Close"

Now all authenticated endpoints will use this token automatically.

### Step 3: Start Diagnostic Session

**Endpoint:** `POST /diagnostic/start`

1. Find `/diagnostic/start` in Swagger UI
2. Click "Try it out"
3. Click "Execute"
4. **Save the `visit_id`** from response:
```json
{
  "message": "Welcome! I'm your Pulmonologist Assistant...",
  "visit_id": "visit_abc123...",
  "patient_id": 1,
  ...
}
```

### Step 4: Send Patient Information

**Endpoint:** `POST /diagnostic/chat`

1. Find `/diagnostic/chat` endpoint
2. Click "Try it out"
3. Fill in the form:
   - **message**: `"my name is John, age 30, male, weight 70 kg, smoker, symptoms: fever and cough for 3 days, medical history: dust allergy"`
   - **visit_id**: (paste the visit_id from Step 3)
   - Leave other fields empty for now
4. Click "Execute"
5. Review the response - should show confirmation request

### Step 5: Confirm Patient Data

**Endpoint:** `POST /diagnostic/chat`

1. Use same endpoint
2. Fill in:
   - **message**: `"yes"` or `"confirm"`
   - **visit_id**: (same visit_id)
3. Click "Execute"
4. Response should show clinical assessment and test recommendations

### Step 6: Upload X-ray Image (Optional)

**Endpoint:** `POST /diagnostic/upload-xray`

1. Find `/diagnostic/upload-xray` endpoint
2. Click "Try it out"
3. Fill in:
   - **file**: Click "Choose File" and select an X-ray image (JPEG/PNG)
   - **visit_id**: (same visit_id)
4. Click "Execute"
5. Response confirms upload

### Step 7: Continue Chat with Test Data

**Endpoint:** `POST /diagnostic/chat`

1. If you uploaded X-ray, continue with:
   - **message**: `"i have both xray and cbc"`
   - **visit_id**: (same visit_id)
2. Or provide CBC data:
   - **message**: `"give me cbc form"`
   - **visit_id**: (same visit_id)
3. Click "Execute"
4. Continue the conversation to complete the workflow

### Step 8: Provide CBC Data

**Endpoint:** `POST /diagnostic/chat`

1. If form was requested, provide data:
   - **message**: `"WBC: 7.5, RBC: 4.8, HGB: 14.2, HCT: 42.0, PLT: 250"`
   - **visit_id**: (same visit_id)
   - **cbc_data**: (optional, can also send as JSON string)
2. Click "Execute"

### Step 9: Provide Spirometry Data

**Endpoint:** `POST /diagnostic/chat`

1. Continue with:
   - **message**: `"fev1: 3.5, fvc: 4.5"`
   - **visit_id**: (same visit_id)
   - **spirometry_data**: (optional JSON string)
2. Click "Execute"

### Step 10: Approve Treatment

**Endpoint:** `POST /diagnostic/chat`

1. When treatment plan is shown:
   - **message**: `"approve"` or `"yes"`
   - **visit_id**: (same visit_id)
2. Click "Execute"
3. System will generate final report

## 🔍 Testing Individual Endpoints

### Test Authentication

**Login:**
- Endpoint: `POST /auth/login`
- Body:
```json
{
  "username": "testuser",
  "password": "testpass123"
}
```

### Test Patient Info

**Get Current Patient:**
- Endpoint: `GET /patients/me`
- No body needed (uses token from Authorization)

### Test Visits

**Get All Visits:**
- Endpoint: `GET /visits`
- No body needed

### Test RAG System

**Get RAG Stats:**
- Endpoint: `GET /rag/stats`
- No body needed

## 📝 Example Complete Test Flow

Here's a complete test sequence you can copy-paste:

### 1. Signup
```json
POST /auth/signup
{
  "username": "testuser",
  "email": "test@example.com",
  "password": "testpass123",
  "patient_name": "John Doe",
  "patient_age": 30,
  "patient_gender": "Male"
}
```

### 2. Start Diagnostic
```
POST /diagnostic/start
(Save visit_id from response)
```

### 3. Provide Patient Info
```
POST /diagnostic/chat
message: "my name is John, age 30, male, weight 70 kg, smoker, symptoms: fever cough chest pain for 5 days, medical history: dust allergy, occupation: engineer"
visit_id: <from step 2>
```

### 4. Confirm
```
POST /diagnostic/chat
message: "yes"
visit_id: <same>
```

### 5. Upload X-ray (if you have an image)
```
POST /diagnostic/upload-xray
file: <select image file>
visit_id: <same>
```

### 6. Continue with Tests
```
POST /diagnostic/chat
message: "both xray and cbc"
visit_id: <same>
```

### 7. Provide CBC
```
POST /diagnostic/chat
message: "WBC: 7.5, RBC: 4.8, HGB: 14.2, HCT: 42.0, PLT: 250"
visit_id: <same>
```

### 8. Provide Spirometry
```
POST /diagnostic/chat
message: "fev1: 3.5, fvc: 4.5"
visit_id: <same>
```

### 9. Approve Treatment
```
POST /diagnostic/chat
message: "approve"
visit_id: <same>
```

## 🛠️ Using cURL (Alternative to Swagger UI)

If you prefer command line:

### 1. Signup
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "patient_name": "John Doe",
    "patient_age": 30,
    "patient_gender": "Male"
  }'
```

### 2. Login (get token)
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass123"
```

### 3. Start Diagnostic (use token from login)
```bash
curl -X POST "http://localhost:8000/diagnostic/start" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 4. Chat (use visit_id from start)
```bash
curl -X POST "http://localhost:8000/diagnostic/chat" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "message=my name is John, age 30, male, symptoms: fever and cough" \
  -F "visit_id=YOUR_VISIT_ID_HERE"
```

### 5. Upload X-ray
```bash
curl -X POST "http://localhost:8000/diagnostic/upload-xray" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "file=@/path/to/xray/image.jpg" \
  -F "visit_id=YOUR_VISIT_ID_HERE"
```

## 🐛 Troubleshooting

### Issue: "401 Unauthorized"
- **Solution**: Make sure you've authorized in Swagger UI (click "Authorize" button and enter token)

### Issue: "visit_id not found"
- **Solution**: Use the `visit_id` from the `/diagnostic/start` response

### Issue: "No patient_id associated"
- **Solution**: Make sure you created a user account with patient information in signup

### Issue: "State not found"
- **Solution**: Make sure you're using the same `visit_id` throughout the session

### Issue: Server not starting
- **Solution**: 
  - Check if port 8000 is available
  - Make sure virtual environment is activated
  - Check if all dependencies are installed: `pip install -r requirements.txt`

## 📊 Monitoring Workflow State

To see the current state at any point:

1. After any `/diagnostic/chat` call, check the `state` field in the response
2. Look for:
   - `current_step`: Which agent is currently active
   - `test_collection_complete`: Whether tests are done
   - `treatment_approved`: Whether treatment is approved
   - `final_report`: The generated report (when ready)

## ✅ Expected Responses

### After Start:
- `current_step`: `"patient_intake_waiting_input"`
- `message`: Welcome message asking for patient info

### After Patient Info:
- `current_step`: `"patient_intake_awaiting_confirmation"`
- `message`: Confirmation summary

### After Confirmation:
- `current_step`: `"doctor_note_generator"` or `"supervisor"`
- `message`: Clinical assessment + test recommendations

### After Tests:
- `current_step`: `"rag_specialist"` or `"treatment_approval"`
- `message`: Treatment plan

### After Approval:
- `current_step`: `"dosage_calculator"` → `"report_generator"` → `"history_saver"`
- `message`: Final report
- `final_report`: Complete report text

## 🎯 Quick Test Checklist

- [ ] Server running on `http://localhost:8000`
- [ ] Can access `/docs` (Swagger UI)
- [ ] Created user account via `/auth/signup`
- [ ] Authorized in Swagger UI
- [ ] Started diagnostic session via `/diagnostic/start`
- [ ] Provided patient information
- [ ] Confirmed patient data
- [ ] Received clinical assessment
- [ ] Provided test data (X-ray, CBC, Spirometry)
- [ ] Received treatment plan
- [ ] Approved treatment
- [ ] Received final report

---

**Tip**: Use Swagger UI's "Try it out" feature - it's the easiest way to test all endpoints interactively!

