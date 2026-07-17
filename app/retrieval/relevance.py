
IMAGE_RELEVANCE_THRESHOLD = 0.55  # tuned empirically; raise if irrelevant images slip through


def filter_relevant_images(images, threshold: float = IMAGE_RELEVANCE_THRESHOLD):
    return [img for img in images if img.score >= threshold]