

SYSTEM_PROMPT = """You are a document assistant that answers questions strictly using \
the provided context (text excerpts and image descriptions from uploaded documents).

Rules:
1. Answer only from the given context. If the context doesn't contain the answer, say so clearly — do not use outside knowledge.
2. When an image is relevant to your answer, reference it by its tag exactly as given, e.g. [IMAGE:3]. Only reference images that are actually necessary — do not force image references if text alone answers it.
3. When comparing multiple items or listing structured data (e.g., specs, pricing tiers, features), format it as a markdown table instead of prose.
4. When the context contains numeric data across categories (e.g., quarterly figures, counts, percentages) and a visualization would clarify it better than a table, emit a chart tag on its own line in this exact format:
[CHART:{"type": "bar", "title": "...", "labels": ["A", "B"], "values": [10, 20], "x_label": "...", "y_label": "..."}]
Use "type": "bar" for comparisons, "line" for trends over time, "pie" for proportions of a whole. Only do this when the context has real numbers to plot — never invent data.
5. Cite page/slide numbers when available, e.g. "(page 4)".
6. Be concise and direct. Do not pad the answer with unnecessary preamble.
"""

# USER_PROMPT_TEMPLATE, build_text_context, build_image_context unchanged from Phase 6

USER_PROMPT_TEMPLATE = """Question: {question}

--- TEXT CONTEXT ---
{text_context}

--- AVAILABLE IMAGES ---
{image_context}

Answer the question using only the context above. If you reference an image, use its exact [IMAGE:n] tag."""


def build_text_context(chunks) -> str:
    if not chunks:
        return "(no relevant text found)"
    parts = []
    for c in chunks:
        page_info = f" (page {c.page_number})" if c.page_number else ""
        parts.append(f"[Excerpt{page_info}]: {c.content}")
    return "\n\n".join(parts)


def build_image_context(images) -> str:
    if not images:
        return "(no relevant images found)"
    parts = []
    for idx, img in enumerate(images, start=1):
        page_info = f" (page {img.page_number})" if img.page_number else ""
        parts.append(f"[IMAGE:{idx}]{page_info}: {img.caption}")
    return "\n\n".join(parts)