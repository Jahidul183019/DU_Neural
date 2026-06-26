# QueueStorm

AI-powered support-ticket analysis API. Classifies complaints, cross-references transaction evidence, and generates structured responses for routing and resolution.

## Quickstart

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run the server
uvicorn main:app --reload --port 8000
```

## API Endpoints

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /analyze-ticket`

Accepts a `TicketRequest` JSON body and returns a `TicketResponse`.

**Example request:**

```bash
curl -X POST http://localhost:8000/analyze-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TK-001",
    "complaint": "I sent money to the wrong number and need it back",
    "language": "en",
    "channel": "in_app_chat",
    "user_type": "customer"
  }'
```

## Docker

```bash
docker build -t queuestorm .
docker run -p 8000:8000 --env-file .env queuestorm
```

## Project Structure

```
queuestorm/
├── main.py            # FastAPI app, routes, middleware
├── models.py          # Pydantic v2 request/response models
├── analyzer.py        # LLM-based analysis engine (TBD)
├── evidence.py        # Transaction evidence cross-referencing (TBD)
├── safety.py          # Safety guardrails & PII redaction (TBD)
├── prompts.py         # LLM prompt templates (TBD)
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container image definition
├── .env.example       # Environment variable template
└── README.md          # This file
```

## Error Handling

| Status | Condition |
|--------|-----------|
| `422`  | Empty/whitespace complaint or validation error |
| `400`  | Malformed JSON |
| `500`  | Unhandled error (no stack traces or secrets exposed) |
