
import io

import google.generativeai as genai
from PIL import Image
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
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
        self.model = genai.GenerativeModel(settings.vision_model)#creates gemini model obj once 

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),#increases the wait time for each try, capped at 10s
           retry=retry_if_not_exception_type(ResourceExhausted),
           )
    def caption(self, image_bytes: bytes) -> str:
        """
        Returns "" ONLY for genuinely non-informative images (too small)
        Any real API error is allowed to raise — the caller is responsible
        for catching it and deciding how to handle the failure
        """
        image = Image.open(io.BytesIO(image_bytes))

        if image.width < 50 or image.height < 50: #if image is too samll
            return ""

        response = self.model.generate_content([CAPTION_PROMPT, image])
        return response.text.strip()