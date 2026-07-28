from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import os
import io
import re
import json
import uuid
import base64
import tempfile
import logging

import httpx
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# ---------------- Mongo ----------------
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="Bill To Excel API")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Models ----------------
class ExtractedTable(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DownloadPayload(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    filename: Optional[str] = "bill_data.xlsx"


# ---------------- Extraction Prompt ----------------
EXTRACTION_PROMPT = """You are an expert invoice / bill / receipt data extractor.

Your job: extract the ITEM / LINE-ITEM TABLE from this document.

The bill may be in ANY language: Arabic, Hindi, English, Urdu, Chinese, French, etc.
Read the ORIGINAL script but if a header is in a non-Latin script, ALSO provide an English label
for that column so it is usable in Excel. If a column is a currency amount, keep the numeric value in the row.

Return ONLY a strictly valid JSON object with this EXACT shape:

{
  "columns": ["col1", "col2", "col3"],
  "rows": [
    {"col1": "value", "col2": "value", "col3": "value"},
    {"col1": "value", "col2": "value", "col3": "value"}
  ],
  "meta": {
    "vendor": "name if visible else empty string",
    "invoice_number": "if visible else empty string",
    "date": "if visible else empty string",
    "currency": "if visible else empty string",
    "language_detected": "e.g. arabic / english / mixed"
  }
}

Hard rules:
- Return ONLY the JSON. NO markdown fences, NO explanation, NO trailing text.
- Every row MUST have the same keys as "columns".
- Preserve numbers as strings exactly as printed (do NOT re-format).
- Do NOT invent rows. Only extract what is visually present.
- Include EVERY item row you can see, including subtotal / tax / total rows if they belong to the item table.
- If there is no item table at all, return {"columns":[], "rows":[], "meta":{...}}.
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Robustly pull the first {...} JSON object from an LLM response."""
    if not text:
        raise ValueError("Empty response from model")
    # Strip common fences
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Find largest balanced { ... }
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(match.group(0))


def _normalize_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    columns = data.get("columns") or []
    rows = data.get("rows") or []
    # Ensure every row has all columns
    normalized_rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        normalized_rows.append({c: str(r.get(c, "")) for c in columns})
    return {
        "columns": [str(c) for c in columns],
        "rows": normalized_rows,
        "meta": data.get("meta") or {},
    }


async def _extract_via_gemini(tmp_path: str, mime: str) -> Dict[str, Any]:
    api_key = os.environ.get("EMERGENT_LLM_KEY")
    if not api_key:
        raise HTTPException(500, "EMERGENT_LLM_KEY missing in backend env")

    chat = LlmChat(
        api_key=api_key,
        session_id=str(uuid.uuid4()),
        system_message="You are a precise multilingual invoice table extractor.",
    ).with_model("gemini", "gemini-2.5-flash")

    file_attach = FileContentWithMimeType(file_path=tmp_path, mime_type=mime)
    msg = UserMessage(text=EXTRACTION_PROMPT, file_contents=[file_attach])
    response = await chat.send_message(msg)
    parsed = _extract_json_object(response if isinstance(response, str) else str(response))
    return _normalize_extraction(parsed)


async def _extract_via_ollama(file_bytes: bytes, mime: str) -> Dict[str, Any]:
    ollama_url = os.environ.get("OLLAMA_URL", "").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")
    if not ollama_url:
        raise HTTPException(500, "OLLAMA_URL not configured")
    if mime == "application/pdf":
        raise HTTPException(400, "Ollama flow supports images only. Send PDF pages as images.")

    b64 = base64.b64encode(file_bytes).decode()
    async with httpx.AsyncClient(timeout=180) as hc:
        resp = await hc.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": EXTRACTION_PROMPT,
                "images": [b64],
                "stream": False,
                "format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "")
        parsed = _extract_json_object(raw)
        return _normalize_extraction(parsed)


# ---------------- Routes ----------------
@api_router.get("/")
async def root():
    return {"status": "ok", "service": "Bill To Excel"}


@api_router.get("/config")
async def get_config():
    return {
        "backend_ready": True,
        "provider": "ollama" if os.environ.get("OLLAMA_URL") else "gemini",
        "model": os.environ.get("OLLAMA_MODEL", "gemini-2.5-flash"),
    }


ALLOWED_MIME = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "heic": "image/heic",
}


@api_router.post("/extract")
async def extract_bill(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = ALLOWED_MIME.get(ext)
    if not mime:
        # Fall back on incoming content_type
        mime = file.content_type or ""
        if mime not in ALLOWED_MIME.values():
            raise HTTPException(400, f"Unsupported file type: {ext or file.content_type}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    suffix = f".{ext}" if ext else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if os.environ.get("OLLAMA_URL"):
            result = await _extract_via_ollama(content, mime)
        else:
            result = await _extract_via_gemini(tmp_path, mime)

        record = ExtractedTable(
            filename=filename,
            columns=result["columns"],
            rows=result["rows"],
            meta=result.get("meta", {}),
        )
        # Persist history (best-effort, never break the API)
        try:
            await db.extractions.insert_one(record.model_dump())
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to persist extraction: %s", e)

        return {
            "id": record.id,
            "filename": record.filename,
            "columns": record.columns,
            "rows": record.rows,
            "meta": record.meta,
        }
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.exception("JSON parse failure")
        raise HTTPException(422, f"Model returned invalid JSON: {e}")
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@api_router.post("/download-excel")
async def download_excel(payload: DownloadPayload):
    wb = Workbook()
    ws = wb.active
    ws.title = "Bill Data"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="09090B", end_color="09090B", fill_type="solid")
    center = Alignment(horizontal="left", vertical="center", wrap_text=True)

    for i, col in enumerate(payload.columns, 1):
        cell = ws.cell(row=1, column=i, value=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center

    for r_idx, row in enumerate(payload.rows, 2):
        for c_idx, col in enumerate(payload.columns, 1):
            val = row.get(col, "")
            ws.cell(row=r_idx, column=c_idx, value=val).alignment = center

    # Auto-width (simple heuristic)
    for c_idx, col in enumerate(payload.columns, 1):
        max_len = len(str(col))
        for row in payload.rows:
            v = str(row.get(col, ""))
            if len(v) > max_len:
                max_len = len(v)
        ws.column_dimensions[ws.cell(row=1, column=c_idx).column_letter].width = min(max(12, max_len + 2), 60)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", payload.filename or "bill_data.xlsx")
    if not safe_name.lower().endswith(".xlsx"):
        safe_name += ".xlsx"

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@api_router.get("/history")
async def get_history(limit: int = 20):
    docs = await db.extractions.find({}, {"_id": 0}).sort("created_at", -1).to_list(limit)
    return docs


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
