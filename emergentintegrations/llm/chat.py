import os
import base64
import httpx
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class FileContentWithMimeType:
    def __init__(self, file_path: str, mime_type: str):
        self.file_path = file_path
        self.mime_type = mime_type

class UserMessage:
    def __init__(self, text: str, file_contents: Optional[List[FileContentWithMimeType]] = None):
        self.text = text
        self.file_contents = file_contents or []

class LlmChat:
    def __init__(self, api_key: str, session_id: str, system_message: str = ""):
        self.api_key = api_key
        self.session_id = session_id
        self.system_message = system_message
        self.provider = "gemini"
        self.model = "gemini-2.5-flash"

    def with_model(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        return self

    async def send_message(self, message: UserMessage) -> str:
        parts = []
        if message.text:
            parts.append({"text": message.text})
            
        for fc in message.file_contents:
            if os.path.exists(fc.file_path):
                with open(fc.file_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": fc.mime_type,
                        "data": b64_data
                    }
                })

        # Gemini model mapping fallback
        target_model = "gemini-1.5-flash" if ("2.5" in self.model or "1.5" in self.model) else self.model
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": parts}]
        }
        if self.system_message:
            payload["system_instruction"] = {
                "parts": [{"text": self.system_message}]
            }

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Send request to Google Gemini API
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                res_json = resp.json()
                try:
                    return res_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return str(res_json)

            # If gemini-1.5-flash failed, try gemini-1.5-pro or model as specified
            if target_model != "gemini-1.5-pro":
                url_pro = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
                resp_pro = await client.post(url_pro, json=payload)
                if resp_pro.status_code == 200:
                    res_json_pro = resp_pro.json()
                    try:
                        return res_json_pro["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError):
                        return str(res_json_pro)

            # Provide clear error details
            error_body = resp.text[:300]
            error_msg = f"Gemini API returned HTTP {resp.status_code}. Details: {error_body}. Please verify GEMINI_API_KEY in environment variables."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

