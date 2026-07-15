#common interface for all file types extractor

#abc -> used for creating abstract base class
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
#automatically creates constructors and helper method


@dataclass
class ExtractedImage:
    image_bytes: bytes
    page_number: int | None
    extension: str = "png"


@dataclass
class ExtractedContent:
    text_by_page: dict[int, str]  # page_number -> raw text (page 0 if not paginated)
    images: list[ExtractedImage] = field(default_factory=list)#crate a default empty list

#every extractor class must have a method named extract() 
# and that method must return an ExtractedContent object
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> ExtractedContent:
        #Extracts text (per page) and embedded/rendered images from a file
        raise NotImplementedError