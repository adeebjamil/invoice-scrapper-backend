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

        # Gemini model mapping fallback if 2.5-flash not supported directly
        target_model = "gemini-1.5-flash" if "2.5" in self.model else self.model
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": parts}]
        }
        if self.system_message:
            payload["system_instruction"] = {
                "parts": [{"text": self.system_message}]
            }

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Try Google Gemini REST API
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                res_json = resp.json()
                try:
                    return res_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return str(res_json)
            
            # Also attempt with original model name if substituted
            if target_model != self.model:
                url_orig = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                resp_orig = await client.post(url_orig, json=payload)
                if resp_orig.status_code == 200:
                    res_json = resp_orig.json()
                    try:
                        return res_json["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError):
                        return str(res_json)

            # Try Emergent Proxy
            proxy_url = "https://api.emergentmethods.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp_proxy = await client.post(
                proxy_url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": message.text}
                    ]
                }
            )
            if resp_proxy.status_code == 200:
                res_json_proxy = resp_proxy.json()
                return res_json_proxy["choices"][0]["message"]["content"]

            error_msg = f"Gemini API returned {resp.status_code}: {resp.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
