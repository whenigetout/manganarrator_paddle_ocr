from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Literal, Any, Optional, List
from enum import Enum

class MediaNamespace(str, Enum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"

class MediaRef(BaseModel):
    namespace: MediaNamespace
    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        p = Path(v)

        # must be relative
        if p.is_absolute():
            raise ValueError("MediaRef.path must be a relative path")

        # no traversal
        if ".." in p.parts:
            raise ValueError("MediaRef.path must not contain '..'")

        # normalize to posix
        return p.as_posix()

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix
    
    @property
    def posix_path_from_media_root(self) -> str:
        return str(Path(self.namespace.value) / self.path)
    
    @property
    def posix_path_from_namespace(self) -> str:
        return str(Path(self.path).as_posix)
    
    def namespace_path(self, media_root: Path) -> Path:
        return media_root / self.namespace.value 
    
    def resolve(self, media_root: Path) -> Path:
        return media_root / self.namespace.value / self.path

class InferImageResponse(BaseModel):
    image_ref: MediaRef
    image_text: Any
    image_width: int
    image_height: int
    input_tokens: int
    output_tokens: int
    throughput: float

class DialogueLineResponse(BaseModel):
    id: int
    image_id: str
    speaker: str
    gender: str
    emotion: str
    text: str

class OCRImage(BaseModel):
    image_id: str
    inferImageRes: Optional[InferImageResponse] = None
    parsedDialogueLines: Optional[list[DialogueLineResponse]] = None

class OCRRunResponse(BaseModel):
    run_id: str
    imageResults: Optional[List[OCRImage]] = None
    error: Optional[str] = None

# Augmented by paddleocr (this service)
class PaddleDialogueLineResponse(DialogueLineResponse):
    paddlebbox: Optional[list[list[float]]] = None

class PaddleOCRImage(OCRImage):
    paddleocr_result: Optional[Any] = None

class PaddleAugmentedOCRRunResponse(OCRRunResponse):
    imageResults: Optional[List[PaddleOCRImage]] = None

# Exceptions
class InferImageError(Exception):
    pass

class ProcessImageError(InferImageError):
    pass

class OCRRunError(ProcessImageError):
    pass

class ParseDialogueError(Exception):
    pass

class SaveJSONError(Exception):
    pass