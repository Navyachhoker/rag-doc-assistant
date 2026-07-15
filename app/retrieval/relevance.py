"""
A small, explicit relevance gate for images. Retrieval always returns
top_k results even if none are truly relevant (that's how nearest-neighbor
search works) — this threshold prevents the LLM from being handed an
unrelated image just because it was the "least bad" match.
"""
IMAGE_RELEVANCE_THRESHOLD = 0.55  # tuned empirically; raise if irrelevant images slip through


def filter_relevant_images(images, threshold: float = IMAGE_RELEVANCE_THRESHOLD):
    return [img for img in images if img.score >= threshold]