"""
Uses Gemini Vision (gemini-1.5-flash) to generate a rich, searchable
description of each extracted image. This caption is what gets embedded
and matched against user queries — it's the core trick that makes
images retrievable via a text-based vector search.
"""
import io

import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

genai.configure(api_key=settings.google_api_key)

CAPTION_PROMPT = """Describe this image in detail for a document search system.
Include:
- What type of visual it is (chart, diagram, photo, screenshot, table, flowchart, etc.)
- All visible text, labels, numbers, or data points
- The key concept, relationship, or information it conveys

Be specific and factual. Do not speculate beyond what's visible. Write 2-4 sentences."""


class ImageCaptioner:
    def __init__(self) -> None:
        self.model = genai.GenerativeModel(settings.vision_model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def caption(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Skip captioning trivially small images (icons, bullets, dividers)
            # that add noise rather than retrievable signal.
            if image.width < 50 or image.height < 50:
                return ""

            response = self.model.generate_content([CAPTION_PROMPT, image])
            return response.text.strip()
        except Exception as exc:
            logger.warning("image_captioning_failed", error=str(exc))
            return ""