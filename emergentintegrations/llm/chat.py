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
        api_key = self.api_key or os.environ.get("EMERGENT_LLM_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("API Key missing. Please set EMERGENT_LLM_KEY, OPENROUTER_API_KEY or GEMINI_API_KEY in .env")

        # Prepare base64 images and parts
        b64_images = []
        gemini_parts = []
        if message.text:
            gemini_parts.append({"text": message.text})
            
        for fc in message.file_contents:
            if os.path.exists(fc.file_path):
                with open(fc.file_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
                b64_images.append((fc.mime_type, b64_str))
                gemini_parts.append({
                    "inline_data": {
                        "mime_type": fc.mime_type,
                        "data": b64_str
                    }
                })

        async with httpx.AsyncClient(timeout=180.0) as client:
            # 1. If key is an OpenRouter key (sk-or-...) or configured for OpenRouter
            if api_key.startswith("sk-or-") or os.environ.get("OPENROUTER_API_KEY"):
                openrouter_key = os.environ.get("OPENROUTER_API_KEY", api_key)
                
                # Vision-capable model fallbacks for OpenRouter
                vision_models = [
                    self.model if "vision" in self.model or "gemini" in self.model or "flash" in self.model else "google/gemini-2.5-flash",
                    "google/gemini-2.5-flash",
                    "google/gemini-flash-1.5",
                    "qwen/qwen-2.5-vl-72b-instruct:free",
                    "meta-llama/llama-3.2-11b-vision-instruct:free"
                ]

                # Format user content array for OpenRouter
                content_list = [{"type": "text", "text": message.text}]
                for mime, data in b64_images:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{data}"
                        }
                    })

                messages = []
                if self.system_message:
                    messages.append({"role": "system", "content": self.system_message})
                messages.append({"role": "user", "content": content_list})

                last_error = ""
                for model in vision_models:
                    try:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {openrouter_key}",
                                "HTTP-Referer": "https://bill-to-excel.app",
                                "X-Title": "Bill To Excel OCR"
                            },
                            json={
                                "model": model,
                                "messages": messages
                            }
                        )
                        if resp.status_code == 200:
                            res_json = resp.json()
                            return res_json["choices"][0]["message"]["content"]
                        else:
                            last_error = f"OpenRouter model {model} returned HTTP {resp.status_code}: {resp.text[:200]}"
                            logger.warning(last_error)
                    except Exception as ex:
                        last_error = str(ex)

                raise RuntimeError(f"OpenRouter extraction failed: {last_error}")

            # 2. Try Google Gemini Direct API
            target_model = "gemini-1.5-flash" if ("2.5" in self.model or "1.5" in self.model) else self.model
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
            payload = {"contents": [{"parts": gemini_parts}]}
            if self.system_message:
                payload["system_instruction"] = {"parts": [{"text": self.system_message}]}

            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                res_json = resp.json()
                try:
                    return res_json["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return str(res_json)

            # 3. Fallback to OpenRouter if Gemini Direct returned non-200
            try:
                content_list = [{"type": "text", "text": message.text}]
                for mime, data in b64_images:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"}
                    })
                messages = []
                if self.system_message:
                    messages.append({"role": "system", "content": self.system_message})
                messages.append({"role": "user", "content": content_list})

                for alt_model in ["google/gemini-2.5-flash", "google/gemini-flash-1.5", "qwen/qwen-2.5-vl-72b-instruct:free"]:
                    resp_or = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": alt_model, "messages": messages}
                    )
                    if resp_or.status_code == 200:
                        return resp_or.json()["choices"][0]["message"]["content"]
            except Exception:
                pass

            error_body = resp.text[:300]
            raise RuntimeError(f"Model API error HTTP {resp.status_code}: {error_body}")
