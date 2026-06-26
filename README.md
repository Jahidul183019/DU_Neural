# QueueStorm Investigator

AI-powered support-ticket analysis API for digital finance platforms. Classifies complaints, cross-references transaction evidence, and generates structured responses for routing and resolution.

---

## Setup

```bash
git clone https://github.com/Jahidul183019/DU_Neural.git
cd DU_Neural
cp .env.example .env
# Add your GEMINI_API_KEY to .env
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t queuestorm .
docker run -p 8000:8000 --env-file .env queuestorm
```

---

## Endpoints

| Method | Path              | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/health`         | Returns `{"status": "ok"}`         |
| POST   | `/analyze-ticket` | Structured JSON ticket analysis    |

### POST /analyze-ticket

**Request body** — `TicketRequest`:

```json
{
  "ticket_id": "TKT-001",
  "complaint": "I sent 5000 taka to a wrong number",
  "language": "en",
  "channel": "in_app_chat",
  "user_type": "customer",
  "transaction_history": [
    {
      "transaction_id": "TXN-9101",
      "timestamp": "2026-04-14T14:08:22Z",
      "type": "transfer",
      "amount": 5000,
      "counterparty": "+8801719876543",
      "status": "completed"
    }
  ]
}
```

**Response** — `TicketResponse` with all 12 fields:

```json
{
  "ticket_id": "TKT-001",
  "relevant_transaction_id": "TXN-9101",
  "evidence_verdict": "consistent",
  "case_type": "wrong_transfer",
  "severity": "high",
  "department": "dispute_resolution",
  "agent_summary": "Customer reports sending 5000 BDT to the wrong recipient.",
  "recommended_next_action": "Initiate wrong-transfer dispute workflow.",
  "customer_reply": "We have noted your concern. Our team will review your case and contact you through official support channels.",
  "human_review_required": true,
  "confidence": 0.9,
  "reason_codes": ["wrong_transfer_claimed", "high_value_transaction"]
}
```

### Error Responses

| Status | Condition                                            |
|--------|------------------------------------------------------|
| `422`  | Empty/whitespace complaint or Pydantic validation error |
| `400`  | Malformed JSON                                       |
| `503`  | LLM service temporarily unavailable                  |
| `500`  | Unhandled error (no stack traces or secrets exposed)  |

---

## Models

**Model:** Gemini 2.0 Flash (Google Generative AI API)
**Used for:** Generating `agent_summary`, `recommended_next_action`, `customer_reply` text only
**Why chosen:** Fast response time (<5s typical), strong multilingual Bangla/English support, reliable JSON output, instruction-following for safety constraints
**Cost:** Pay-per-token on team's own Google AI account
**Evidence reasoning and safety rules:** Fully rule-based, no model involvement

---

## AI Approach

Hybrid rule + AI architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    9-Step Pipeline                       │
│                                                         │
│  1. Language Detection ─────────────── Rule-based       │
│  2. Prompt Injection Pre-screen ───── Rule-based        │
│  3. Evidence Extraction ───────────── Rule-based        │
│  4. Case Classification ──────────── Rule-based         │
│  5. LLM Text Generation ─────────── Gemini API         │
│  6. Response Parsing ─────────────── JSON parse         │
│  7. Response Assembly ────────────── Deterministic      │
│  8. Safety Post-filter ───────────── Rule-based         │
│  9. Return ───────────────────────── Final response     │
└─────────────────────────────────────────────────────────┘
```

- **Rule engine handles:** transaction matching, evidence verdict, case classification, routing, severity, `human_review_required` decision
- **LLM handles ONLY:** natural language text generation (summaries, replies)
- **Safety post-filter:** rule-based, runs unconditionally after all AI output

---

## Safety Logic

Six hard safety rules enforced by `post_process_safety()` after **every** response:

1. **Credential guard** — `customer_reply` never asks for PIN, OTP, password, or card number
2. **Refund language guard** — `customer_reply` and `recommended_next_action` never confirm unauthorized refund/reversal/unblock
3. **Third-party redirect guard** — `customer_reply` never redirects to WhatsApp, Facebook, or external contacts
4. **Phishing lockdown** — Phishing cases always get `severity=critical`, `department=fraud_risk`, `human_review=true`
5. **Ticket ID echo** — `response.ticket_id` always matches `request.ticket_id`
6. **Prompt injection detection** — Injection patterns in complaint text are detected and flagged; complaint is treated as untrusted data

---

## Known Limitations

- Transaction matching uses heuristics (amount, time, counterparty regex) — complex natural language references may miss
- Bangla/Banglish detection is regex-based — edge cases in transliteration may misclassify
- LLM call adds ~2–5s latency; timeout fallback activates at 25s
- Service has no database — analysis is stateless per request
- No real payment system integration — synthetic data only
- Bengali digit support (`৫০০০` → `5000`) covers standard Unicode range only

---

## Project Structure

```
├── main.py            # FastAPI app, routes, middleware, error handlers
├── models.py          # Pydantic v2 request/response models
├── analyzer.py        # 9-step async analysis pipeline
├── evidence.py        # Rule-based evidence engine (4 public functions)
├── safety.py          # Hard safety post-processor (6 rules)
├── prompts.py         # LLM system prompt + user prompt builder
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container image (python:3.11-slim)
├── .env.example       # Environment variable template
├── sample_output.json # Example request/response for TKT-001
└── README.md          # This file
```

---

## Environment Variables

| Variable         | Description                    | Required |
|------------------|--------------------------------|----------|
| `GEMINI_API_KEY` | Google Generative AI API key   | Yes      |
| `PORT`           | Server port (default: 8000)    | No       |
