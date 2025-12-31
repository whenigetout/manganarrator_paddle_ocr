from pydantic import BaseModel
from pathlib import Path
from typing import Literal, Any, Optional, List

class MediaRef(BaseModel):
    namespace: Literal["inputs", "outputs"]
    path: str

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix

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