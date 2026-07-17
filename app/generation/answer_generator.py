
import json
import re

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.logging_config import get_logger
from app.generation.chart_generator import ChartGenerationError, generate_chart
from app.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_image_context, build_text_context
from app.retrieval.relevance import filter_relevant_images
from app.retrieval.retriever import RetrievedChunkResult, RetrievedImageResult
from app.storage.image_store import ImageStore

settings = get_settings()
logger = get_logger(__name__)

IMAGE_TAG_PATTERN = re.compile(r"\[IMAGE:(\d+)\]")
CHART_TAG_PATTERN = re.compile(r"\[CHART:(\{.*?\})\]", re.DOTALL)


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        self.image_store = ImageStore()

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunkResult],
        images: list[RetrievedImageResult],
    ) -> tuple[str, list[RetrievedImageResult]]:
        relevant_images = filter_relevant_images(images)

        raw_answer = self._call_llm(question, chunks, relevant_images)
        answer_with_images, used_images = self._inject_inline_images(raw_answer, relevant_images)
        final_answer = self._inject_generated_charts(
            answer_with_images, document_id=chunks[0].chunk_id if chunks else 0
        )

        return final_answer, used_images

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, question: str, chunks, images) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            text_context=build_text_context(chunks),
            image_context=build_image_context(images),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _inject_inline_images(
        self, raw_answer: str, images: list[RetrievedImageResult]
    ) -> tuple[str, list[RetrievedImageResult]]:
        used_images: list[RetrievedImageResult] = []

        def replace_tag(match: re.Match) -> str:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(images):
                img = images[idx]
                used_images.append(img)
                return f"\n\n![{img.caption}]({img.file_path})\n"
            logger.warning("invalid_image_tag_referenced", tag_index=idx)
            return ""

        final_answer = IMAGE_TAG_PATTERN.sub(replace_tag, raw_answer)
        return final_answer, used_images

    def _inject_generated_charts(self, text: str, document_id: int) -> str:
        def replace_chart(match: re.Match) -> str:
            raw_json = match.group(1)
            try:
                spec = json.loads(raw_json)
                chart_bytes = generate_chart(spec)
                file_path = self.image_store.save(chart_bytes, "png", document_id)
                title = spec.get("title", "Generated chart")
                return f"\n\n![{title}]({file_path})\n"
            except (json.JSONDecodeError, ChartGenerationError) as exc:
                logger.warning("chart_generation_failed", error=str(exc))
                return ""

        return CHART_TAG_PATTERN.sub(replace_chart, text)