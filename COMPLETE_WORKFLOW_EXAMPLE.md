# Complete Workflow Example - Doctor Assistant System

## Overview
This document shows the complete end-to-end flow of the Doctor Assistant system with a real patient example.

---

## Example Patient Case

**Patient:** Sarah Johnson, 8 years old, 30kg, Female, Non-smoker
**Symptoms:** Cough, fever (101°F), difficulty breathing for 3 days
**Medical History:** Asthma (mild)
**Occupation:** Student

---

## Complete Flow Step-by-Step

### **PHASE 1: Starting the Session**

#### Step 1: User Calls `/diagnostic/start`
```
POST http://localhost:8000/diagnostic/start
Body: { "patient_id": null }  // New patient
```

**What Happens:**
- System creates a new `visit_id` (e.g., "abc-123-def-456")
- Initializes empty state (all fields = None)
- Starts LangGraph workflow with `patient_intake` agent
- State is saved to SqliteSaver checkpointer

**Response:**
```json
{
  "message": "Welcome! I'm your Pulmonologist Assistant. Let me collect some information.\n\nPlease provide:\n- Your name, age, and gender\n- Your weight (in kg or lbs)\n- Are you a smoker? (Yes/No)\n- Your current symptoms...",
  "current_step": "patient_intake",
  "visit_id": "abc-123-def-456",
  "patient_id": null,
  "emergency_flag": false
}
```

---

### **PHASE 2: Patient Intake (HITL - Human in the Loop)**

#### Step 2: User Provides Information
```
POST http://localhost:8000/diagnostic/chat
Body:
  message: "My name is Sarah Johnson, I'm 8 years old, female, I weigh 30kg. I'm not a smoker. I have a cough, fever around 101 degrees, and difficulty breathing. This started 3 days ago. I have mild asthma."
  visit_id: "abc-123-def-456"
```

**What Happens:**
1. **Patient Intake Agent** processes the message:
   - Extracts: name="Sarah Johnson", age=8, gender="female", weight=30kg
   - Extracts: smoker=false, symptoms="cough, fever, difficulty breathing"
   - Extracts: duration="3 days", chronic_conditions="asthma (mild)"
   - Stores all in state

2. **Confirmation Check:**
   - Agent checks if `patient_data_confirmed = False`
   - Generates summary and asks for confirmation

**State After This Step:**
```python
{
  "patient_name": "Sarah Johnson",
  "patient_age": 8,
  "patient_gender": "female",
  "patient_weight": 30.0,
  "patient_smoker": False,
  "symptoms": "cough, fever, difficulty breathing",
  "symptom_duration": "3 days",
  "patient_chronic_conditions": "asthma (mild)",
  "patient_data_confirmed": False,
  "current_step": "patient_intake_awaiting_confirmation"
}
```

**Response:**
```json
{
  "message": "I need your confirmation. Please review:\n\nName: Sarah Johnson\nAge: 8\nGender: female\nWeight: 30.0kg\n...\n\nReply 'Yes' if correct, or tell me what needs to be changed.",
  "current_step": "patient_intake_awaiting_confirmation"
}
```

#### Step 3: User Confirms
```
POST http://localhost:8000/diagnostic/chat
Body:
  message: "Yes, that's correct"
  visit_id: "abc-123-def-456"
```

**What Happens:**
- Patient Intake Agent detects "Yes"
- Sets `patient_data_confirmed = True`
- Workflow moves to next agent

**Response:**
```json
{
  "message": null,  // No message, workflow continues
  "current_step": "emergency_detector"
}
```

---

### **PHASE 3: Emergency Detection**

#### Step 4: Emergency Detector Agent Runs
**What Happens:**
- Agent analyzes symptoms: "difficulty breathing" + "fever" + "asthma history"
- Uses LLM to check if this is an emergency
- LLM determines: **NOT an emergency** (moderate symptoms, not life-threatening)
- Sets `emergency_flag = False`

**State:**
```python
{
  "emergency_flag": False,
  "emergency_reason": None,
  "current_step": "doctor_note_generator"
}
```

**Response:** (Workflow continues automatically)

---

### **PHASE 4: Doctor Note Generation**

#### Step 5: Doctor Note Generator Agent Runs
**What Happens:**
- Agent creates a clinical note summarizing:
  - Patient demographics (8-year-old female, 30kg)
  - Symptoms (cough, fever, difficulty breathing, 3 days)
  - Medical history (mild asthma)
- Uses LLM to generate professional medical note

**State:**
```python
{
  "doctor_note": "8-year-old female patient presents with 3-day history of cough, fever (101°F), and difficulty breathing. Patient has history of mild asthma. Symptoms suggest possible respiratory infection or asthma exacerbation. Further diagnostic testing recommended."
}
```

**Response:** (Workflow continues)

---

### **PHASE 5: Diagnostic Controller (Test Recommendations & Processing)**

#### Step 6: Diagnostic Controller Agent Runs
**What Happens:**
1. **Test Recommendation:**
   - LLM analyzes symptoms: "difficulty breathing" + "fever" + "cough"
   - Recommends: **X-ray** (for pneumonia), **Spirometry** (for breathing), **CBC** (general)

2. **Asks User Which Tests They Have:**
   - Agent checks conversation for test data
   - No test data found yet
   - Asks user which tests they want to provide

**Response:**
```json
{
  "message": "Based on your symptoms, I recommend:\n1. X-ray (to check for pneumonia)\n2. Spirometry (to check lung function)\n3. CBC blood test (general health check)\n\nWhich of these tests do you have results for? You can upload X-ray images or provide Spirometry/CBC values.",
  "current_step": "diagnostic_controller"
}
```

#### Step 7: User Provides Test Data
```
POST http://localhost:8000/diagnostic/chat
Body:
  message: "I have X-ray and Spirometry results. X-ray shows no pneumonia. Spirometry: FEV1 is 2.1L, FVC is 2.8L, FEV1/FVC is 75%"
  visit_id: "abc-123-def-456"
```

**What Happens:**
1. **X-ray Processing:**
   - User mentioned "no pneumonia" in text
   - If X-ray image uploaded, ML model processes it
   - Stores result: `xray_result = {"prediction": {"disease_name": "No disease", "confidence": 0.95}}`

2. **Spirometry Processing:**
   - Extracts values: FEV1=2.1L, FVC=2.8L, FEV1/FVC=75%
   - Calls ML model: `predict_spirometry(fev1=2.1, fvc=2.8, fev1_fvc=75, age=8, gender="female")`
   - Result: **Mild obstruction** (asthma-related)
   - Stores: `spirometry_result = {"pattern": "obstruction", "severity": "mild"}`

3. **CBC:**
   - User didn't provide CBC
   - Marks as missing: `missing_tests = ["cbc"]`

**State:**
```python
{
  "xray_available": True,
  "xray_result": {"prediction": {"disease_name": "No disease", "confidence": 0.95}},
  "spirometry_available": True,
  "spirometry_result": {"pattern": "obstruction", "severity": "mild"},
  "cbc_available": False,
  "missing_tests": ["cbc"]
}
```

**Response:**
```json
{
  "message": "Test results processed:\n- X-ray: No pneumonia detected\n- Spirometry: Mild obstruction detected (consistent with asthma)\n- CBC: Not provided (will use healthy defaults if needed)\n\nProceeding with diagnosis...",
  "current_step": "rag_specialist"
}
```

---

### **PHASE 6: RAG Specialist (Diagnosis & Treatment)**

#### Step 8: RAG Specialist Agent Runs
**What Happens:**
1. **RAG Retrieval:**
   - Queries RAG system with: symptoms, age, test results
   - Retrieves relevant medical knowledge about:
     - Asthma exacerbation in children
     - Treatment guidelines
     - Medication recommendations

2. **Diagnosis Generation:**
   - LLM uses RAG context + test results + patient info
   - Generates diagnosis: **"Asthma exacerbation with possible viral respiratory infection"**

3. **Treatment Plan Generation:**
   - Based on RAG context and diagnosis
   - Generates treatment:
     - Albuterol inhaler (for breathing difficulty)
     - Prednisone (for inflammation)
     - Rest and hydration
     - Follow-up in 3 days

**State:**
```python
{
  "rag_context": "[Retrieved medical knowledge about asthma treatment...]",
  "diagnosis": "Asthma exacerbation with possible viral respiratory infection",
  "treatment_plan": [
    "Albuterol inhaler: 2 puffs every 4-6 hours as needed",
    "Prednisone: 20mg once daily for 5 days",
    "Rest and adequate hydration",
    "Monitor symptoms closely"
  ],
  "home_remedies": [
    "Use humidifier",
    "Avoid triggers (dust, smoke)",
    "Stay hydrated"
  ],
  "followup_instruction": "Follow up in 3 days or sooner if symptoms worsen",
  "treatment_approved": False
}
```

**Response:**
```json
{
  "message": "Here's the treatment plan:\n\nDiagnosis: Asthma exacerbation with possible viral respiratory infection\n\nTreatment Plan:\n1. Albuterol inhaler: 2 puffs every 4-6 hours as needed\n2. Prednisone: 20mg once daily for 5 days\n3. Rest and adequate hydration\n4. Monitor symptoms closely\n\nHome Remedies:\n1. Use humidifier\n2. Avoid triggers (dust, smoke)\n3. Stay hydrated\n\nFollow-up: Follow up in 3 days or sooner if symptoms worsen\n\nPlease confirm if you're okay with this plan.",
  "current_step": "rag_specialist_awaiting_approval"
}
```

---

### **PHASE 7: Treatment Approval (HITL)**

#### Step 9: User Reviews Treatment
```
POST http://localhost:8000/diagnostic/chat
Body:
  message: "How do I take the Prednisone? Should I take it with food?"
  visit_id: "abc-123-def-456"
```

**What Happens:**
- **Treatment Approval Agent** detects question
- Uses LLM to answer: "Yes, take Prednisone with food to reduce stomach upset. Take it once daily in the morning."
- Still waiting for approval

**Response:**
```json
{
  "message": "Yes, take Prednisone with food to reduce stomach upset. Take it once daily in the morning. This helps reduce inflammation and improve breathing. Continue for 5 days as prescribed.\n\nIs this treatment plan acceptable? Please reply 'Yes' to approve.",
  "current_step": "rag_specialist_awaiting_approval"
}
```

#### Step 10: User Approves
```
POST http://localhost:8000/diagnostic/chat
Body:
  message: "Yes, I approve"
  visit_id: "abc-123-def-456"
```

**What Happens:**
- Treatment Approval Agent detects "Yes"
- Sets `treatment_approved = True`
- Workflow moves to Dosage Calculator

**Response:**
```json
{
  "message": "Thank you for approving the treatment plan. Calculating specific dosages...",
  "current_step": "dosage_calculator"
}
```

---

### **PHASE 8: Dosage Calculator**

#### Step 11: Dosage Calculator Agent Runs
**What Happens:**
1. **Extracts Medications:**
   - "Albuterol inhaler" → medication = "albuterol"
   - "Prednisone 20mg" → medication = "prednisone"

2. **Calculates Dosages:**
   - **Albuterol** (inhaler, not weight-based):
     - Pediatric: 1-2 puffs (uses 2 puffs as recommended)
     - Result: "Albuterol 2 puffs every 4-6 hours as needed"
   
   - **Prednisone** (weight-based for pediatric):
     - Rule: 1mg/kg (max 40mg)
     - Calculation: 30kg × 1mg/kg = 30mg
     - But treatment says 20mg, so uses 20mg (lower is safer)
     - Result: "Prednisone 20mg once daily (with food)"

3. **Updates Treatment Plan:**
   - Replaces generic items with specific dosages
   - Adds food instructions

**State:**
```python
{
  "calculated_dosages": {
    "albuterol": {
      "dose": "2 puffs",
      "frequency": "every 4-6 hours as needed",
      "with_food": False,
      "notes": "Inhaler - use as needed for breathing difficulty"
    },
    "prednisone": {
      "dose": "20mg",
      "frequency": "once daily",
      "with_food": True,
      "notes": "Take with food, taper dose gradually",
      "calculation": "Based on weight: 30kg × 1mg/kg = 20mg (adjusted to treatment plan)"
    }
  },
  "treatment_plan": [
    "Albuterol 2 puffs every 4-6 hours as needed (can be taken with or without food) - Note: Inhaler - use as needed for breathing difficulty",
    "Prednisone 20mg once daily (with food) - Note: Take with food, taper dose gradually"
  ]
}
```

**Response:**
```json
{
  "message": "Dosages calculated based on your age and weight (30kg):\n\nAlbuterol: 2 puffs every 4-6 hours as needed\n  Note: Inhaler - use as needed for breathing difficulty\n\nPrednisone: 20mg once daily\n  Note: Take with food, taper dose gradually",
  "current_step": "report_generator"
}
```

---

### **PHASE 9: Report Generation**

#### Step 12: Report Generator Agent Runs
**What Happens:**
- Combines all information into final report:
  - Patient information
  - Symptoms and history
  - Test results (X-ray, Spirometry)
  - Diagnosis
  - Treatment plan with calculated dosages
  - Home remedies
  - Follow-up instructions

**State:**
```python
{
  "message": "[Comprehensive final report generated]"
}
```

**Response:**
```json
{
  "message": "=== PATIENT VISIT REPORT ===\n\nPatient: Sarah Johnson\nAge: 8 years\nGender: Female\nWeight: 30kg\n\nChief Complaints:\n- Cough, fever (101°F), difficulty breathing\n- Duration: 3 days\n- Medical History: Mild asthma\n\nDiagnostic Test Results:\n- X-ray: No pneumonia detected (confidence: 95%)\n- Spirometry: Mild obstruction detected (FEV1/FVC: 75%)\n- CBC: Not performed\n\nDiagnosis:\nAsthma exacerbation with possible viral respiratory infection\n\nTreatment Plan:\n1. Albuterol 2 puffs every 4-6 hours as needed\n   Note: Inhaler - use as needed for breathing difficulty\n2. Prednisone 20mg once daily (with food)\n   Note: Take with food, taper dose gradually\n3. Rest and adequate hydration\n4. Monitor symptoms closely\n\nHome Remedies:\n- Use humidifier\n- Avoid triggers (dust, smoke)\n- Stay hydrated\n\nFollow-up Instructions:\nFollow up in 3 days or sooner if symptoms worsen.\n\n=== END OF REPORT ===",
  "current_step": "history_saver"
}
```

---

### **PHASE 10: History Saver**

#### Step 13: History Saver Agent Runs
**What Happens:**
1. **Creates/Updates Patient Record:**
   - Patient: Sarah Johnson (new patient, creates record)
   - Stores: name, age, gender, weight, smoker status, chronic conditions

2. **Creates Visit Record:**
   - Visit ID: abc-123-def-456
   - Stores: symptoms, duration, test results (as JSON)
   - Links to patient

3. **Creates Diagnosis Record:**
   - Diagnosis: "Asthma exacerbation with possible viral respiratory infection"
   - Treatment plan (as JSON)
   - Home remedies (as JSON)
   - Follow-up instructions

**Database Records Created:**
```sql
Patient:
  id: 1
  name: "Sarah Johnson"
  age: 8
  gender: "female"
  weight: 30.0
  smoker: false
  chronic_conditions: "asthma (mild)"

Visit:
  id: 1
  patient_id: 1
  visit_id: "abc-123-def-456"
  symptoms: "cough, fever, difficulty breathing"
  xray_result: '{"prediction": {"disease_name": "No disease"}}'
  spirometry_result: '{"pattern": "obstruction", "severity": "mild"}'

Diagnosis:
  id: 1
  visit_id: 1
  diagnosis: "Asthma exacerbation with possible viral respiratory infection"
  treatment_plan: '["Albuterol 2 puffs...", "Prednisone 20mg..."]'
  followup_instruction: "Follow up in 3 days or sooner if symptoms worsen"
```

**Response:**
```json
{
  "message": "Visit data saved successfully to database.",
  "current_step": "followup_agent"
}
```

---

### **PHASE 11: Follow-up Agent (If Returning Patient)**

#### Step 14: Follow-up Agent Runs
**What Happens:**
- Checks if `patient_id` exists (Sarah is new patient, so `patient_id = None` initially)
- After saving, `patient_id = 1` is set
- But this is first visit, so no previous visits to compare
- Skips follow-up analysis

**State:**
```python
{
  "previous_visits": None,
  "progress_summary": None
}
```

**Response:**
```json
{
  "message": "Visit completed successfully. Your report has been saved.",
  "current_step": "end"
}
```

---

## Complete Flow Summary

```
1. START → Patient Intake (collects info, asks confirmation)
2. Patient Confirms → Emergency Detector (checks for emergencies)
3. No Emergency → Doctor Note Generator (creates clinical note)
4. → Diagnostic Controller (recommends tests, processes results)
5. → RAG Specialist (generates diagnosis & treatment using medical knowledge)
6. → Treatment Approval (patient reviews, asks questions, approves)
7. → Dosage Calculator (calculates specific dosages based on age/weight)
8. → Report Generator (creates final comprehensive report)
9. → History Saver (saves to database)
10. → Follow-up Agent (compares with previous visits if returning patient)
11. END → Complete!
```

---

## Key Features Demonstrated

✅ **Human-in-the-Loop (HITL):**
- Patient data confirmation
- Treatment plan approval with Q&A

✅ **ML Model Integration:**
- X-ray pneumonia detection
- Spirometry pattern analysis

✅ **RAG (Retrieval-Augmented Generation):**
- Evidence-based diagnosis
- Treatment recommendations from medical knowledge

✅ **Dosage Calculator:**
- Rule-based for common medications
- Weight-based pediatric dosing
- LLM-based for complex cases

✅ **State Persistence:**
- LangGraph checkpointer (SqliteSaver)
- Conversation continues across multiple messages
- State saved automatically at each step

✅ **Error Handling:**
- Graceful degradation
- User-friendly error messages
- Full error logging

---

## Example: Returning Patient (Follow-up Visit)

If Sarah comes back 1 week later:

**Step 1:** User calls `/diagnostic/start` with `patient_id: 1`

**Step 2-13:** Same flow as above

**Step 14: Follow-up Agent:**
- Retrieves last 3 visits for patient_id=1
- Compares:
  - Previous: "Asthma exacerbation" → Current: "Asthma exacerbation" (recurrence)
  - Previous: "Mild obstruction" → Current: "Moderate obstruction" (worsening)
  - Previous: "Albuterol + Prednisone" → Current: Same treatment (not effective enough)
- Generates progress summary:
  - "Patient's asthma symptoms have recurred. Spirometry shows worsening (mild to moderate obstruction). Previous treatment may need adjustment. Consider increasing Prednisone dose or adding controller medication."

**Response includes:**
- Current diagnosis and treatment
- Progress summary comparing to previous visits
- Treatment modification suggestions

---

## API Endpoints Used

1. `POST /diagnostic/start` - Start new session
2. `POST /diagnostic/chat` - Continue conversation (multiple times)
3. `GET /diagnostic/state/{visit_id}` - Check current state (optional)
4. `DELETE /diagnostic/state/{visit_id}` - Clean up session (optional)

---

## Total Messages in This Example

1. `/start` → Welcome message
2. `/chat` → User provides info
3. `/chat` → System asks for confirmation
4. `/chat` → User confirms
5. `/chat` → System asks about tests
6. `/chat` → User provides test results
7. `/chat` → System shows treatment plan
8. `/chat` → User asks question
9. `/chat` → System answers question
10. `/chat` → User approves
11. `/chat` → Final report

**Total: 11 API calls for complete workflow**

---

## State Evolution

**Initial State:** Empty (all None)
**After Intake:** Patient info filled
**After Tests:** Test results added
**After RAG:** Diagnosis & treatment added
**After Approval:** Treatment approved
**After Dosage:** Specific dosages calculated
**After Report:** Final report generated
**After Save:** Saved to database

---

This is the complete flow of your Doctor Assistant system! 🎉

