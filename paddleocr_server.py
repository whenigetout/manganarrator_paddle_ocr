from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from paddleocr import PaddleOCR
import yaml
import shutil
from datetime import datetime
import uuid
from typing import Optional, Any, Mapping
import traceback
from models.domain import *

# ---- Config ----
CONFIG_PATH = 'config.yaml'
if Path(CONFIG_PATH).exists():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}

OCR_ARGS = dict(
    use_doc_orientation_classify=config.get("use_doc_orientation_classify", False),
    use_doc_unwarping=config.get("use_doc_unwarping", False),
    use_textline_orientation=config.get("use_textline_orientation", False),
    text_det_limit_side_len=8000
)
OUTPUT_DIR = Path(config.get("output_root_folder", "./output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_ROOT: str = config.get("media_root", "")

if not MEDIA_ROOT:
    raise ValueError("media_root not found in config, please set a valid media_root folder path")

ocr = PaddleOCR(**OCR_ARGS)

app = FastAPI()

def extract_paddle_fields(ocr_res: Any):
    """
    Normalizes PaddleOCR output (dict or OCRResult object)
    into (rec_texts, rec_polys, rec_boxes)
    """
    if isinstance(ocr_res, Mapping):
        rec_texts = ocr_res.get("rec_texts", [])
        rec_polys = ocr_res.get("rec_polys", [])
        rec_boxes = ocr_res.get("rec_boxes", [])
    else:
        rec_texts = getattr(ocr_res, "rec_texts", [])
        rec_polys = getattr(ocr_res, "rec_polys", [])
        rec_boxes = getattr(ocr_res, "rec_boxes", [])

    return rec_texts, rec_polys, rec_boxes


def log_exception(context: str = "Unhandled exception", label: str = "💀"):
    print(f"\n{label} {context}:")
    traceback.print_exc()

def process_and_save(img_path, out_dir):
    try:
        result = ocr.predict(input=str(img_path))
        for idx, res in enumerate(result):
            res.save_to_img(str(out_dir))
            json_path = res.save_to_json(str(out_dir))
        return [str(json_path) for res in result]
    except Exception:
        log_exception("process_and_save")
        raise  # re-raise so caller gets proper 500 response


@app.post("/paddleocr/image")
async def ocr_single_image(
    file: UploadFile = File(...),
    save_uploaded_image: bool = Form(False)
):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        img_name = f"{ts}_{uid}_{file.filename}"
        img_path = OUTPUT_DIR / img_name

        with open(img_path, "wb") as f:
            f.write(await file.read())

        img_out_dir = OUTPUT_DIR / img_path.stem
        img_out_dir.mkdir(exist_ok=True)

        json_paths = process_and_save(img_path, img_out_dir)

        if not save_uploaded_image:
            img_path.unlink(missing_ok=True)

        return JSONResponse(content={
            "json_outputs": json_paths,
            "output_dir": str(img_out_dir),
            "count": len(json_paths)
        })
    except Exception:
        log_exception("ocr_single_image")
        return JSONResponse(status_code=500, content={"error": "See server logs for details"})

@app.post("/paddleocr/folder")
def ocr_folder(input_dir: str = Form(...)):
    try:
        input_dir_path = Path(input_dir).resolve()
        imgs = sorted([p for p in input_dir_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]])
        all_json = []
        for img_path in imgs:
            img_out_dir = OUTPUT_DIR / img_path.stem
            img_out_dir.mkdir(exist_ok=True)
            json_paths = process_and_save(img_path, img_out_dir)
            all_json.extend(json_paths)
        return JSONResponse(content={
            "json_outputs": all_json,
            "output_dir": str(OUTPUT_DIR),
            "count": len(all_json)
        })
    except Exception:
        log_exception("ocr_folder")
        return JSONResponse(status_code=500, content={"error": "See server logs for details"})


@app.post("/paddleocr/augment_json")
def augment_json_with_paddle(
    ocr_json_path: str = Form(..., description="Path to existing OCR JSON file"),
    input_img_root_folder: Optional[str] = Form(
        default=None,
        description="(Optional) Root folder where images are stored",
        example=None
    ),
    output_json_root_folder: Optional[str] = Form(
        default=None,
        description="(Optional) Root folder where images are stored",
        example=None
    ),
    use_process_and_save: bool = Form(
        False,
        description="Whether to also run process_and_save for debugging"
    )
):


    """
    Augments an existing OCR JSON file with PaddleOCR results.
    - Uses root_folder param if provided
    - Else falls back to config.yml
    - If neither available -> error
    - Optionally reuses process_and_save for debugging
    """

    try:
        import json
        from pathlib import Path

        # --- resolve root folders ---
        # Input folder
        if not input_img_root_folder or input_img_root_folder.strip().lower() == "string":
            img_root_folder = config.get("input_root_folder")

        if img_root_folder is None:
            return JSONResponse(
                status_code=400,
                content={"error": "input_root_folder not provided and not found in config.yml"}
            )

        img_root_folder = Path(img_root_folder)

        # Output folder
        if not output_json_root_folder or output_json_root_folder.strip().lower() == "string":
            ocr_json_root_folder = config.get("output_root_folder")

        if ocr_json_root_folder is None:
            return JSONResponse(
                status_code=400,
                content={"error": "output_root_folder not provided and not found in config.yml"}
            )

        ocr_json_root_folder = Path(ocr_json_root_folder)

        # --- load JSON ---
        ocr_json_path_ = Path(ocr_json_path).resolve()
        # if not ocr_json_path_.is_absolute():
        #     ocr_json_path_ = ocr_json_root_folder / ocr_json_path_

        if not ocr_json_path_.exists():
            return JSONResponse(
                status_code=400,
                content={"error": f"OCR JSON file not found at {ocr_json_path_}"}
            )

        with open(ocr_json_path_, "r", encoding="utf-8") as f:
            ocr_data = OCRRunResponse.model_validate(json.load(f))

        # --- validate ---
        if not ocr_data or not ocr_data.imageResults:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid JSON file: {ocr_json_path_}"}
            )

        # --- process each entry ---
        augmented_images: list[PaddleOCRImage] = []
        for ocrimg in ocr_data.imageResults:
            paddle_img = PaddleOCRImage.model_validate(ocrimg.model_dump())
            if not paddle_img.inferImageRes:
                return JSONResponse(
                status_code=400,
                content={"error": f"Invalid JSON file, Invalid image result: {ocr_json_path_}"}
            )
            img_path = Path(MEDIA_ROOT) / paddle_img.inferImageRes.image_ref.namespace / paddle_img.inferImageRes.image_ref.path

            if not img_path.exists():
                paddle_img.paddleocr_result = {"error": f"Image not found at {img_path}"}
                continue

            if use_process_and_save:
                # Reuse full pipeline (debug mode)
                results = process_and_save(
                    str(img_path),
                    "output",
                )
                paddle_img.paddleocr_result = results
            else:
                # Minimal run
                result = ocr.predict(input=str(img_path))

                # right after: result = ocr.predict(input=str(img_path))
                print("\n🔍 DEBUG: Raw result type ->", type(result))
                if isinstance(result, list):
                    print("  Length of result:", len(result))
                    if len(result) > 0:
                        print("  First element type:", type(result[0]))
                        # If it’s an object:
                        if hasattr(result[0], "__dict__"):
                            print("  First element keys:", list(result[0].__dict__.keys()))
                        # If it’s a tuple/list:
                        if isinstance(result[0], (list, tuple)):
                            print("  First element sample (truncated):", json.dumps(result[0], indent=2)[:500])
                else:
                    print("Result is NOT a list:", result)


                rec_texts, rec_polys, rec_boxes = [], [], []

                if result and len(result) > 0:
                    ocr_res = result[0]  # OCRResult object (like in process_and_save)
                    rec_texts, rec_polys, rec_boxes = extract_paddle_fields(ocr_res)
                    # if type(ocr_res) == dict:

                    #     if "rec_texts" in ocr_res:
                    #         rec_texts = ocr_res["rec_texts"]
                    #     else:
                    #         rec_texts = []

                    #     if "rec_polys" in ocr_res:
                    #         rec_polys = ocr_res["rec_polys"]
                    #     else:
                    #         rec_polys = []

                    #     if "rec_boxes" in ocr_res:
                    #         rec_boxes = ocr_res["rec_boxes"]
                    #     else:
                    #         rec_boxes = []

                    # convert ndarray -> list
                    if rec_polys is not None:
                        rec_polys = [p.tolist() if hasattr(p, "tolist") else p for p in rec_polys]
                    if rec_boxes is not None:
                        rec_boxes = [b.tolist() if hasattr(b, "tolist") else b for b in rec_boxes]

                paddle_img.paddleocr_result = {
                    "rec_texts": rec_texts,
                    "rec_polys": rec_polys,
                    "rec_boxes": rec_boxes
                }

                augmented_images.append(paddle_img)

        augmented_ocrrun = PaddleAugmentedOCRRunResponse(
            run_id=ocr_data.run_id,
            imageResults=augmented_images,
            error=ocr_data.error
        )

        # --- save augmented file ---
        out_path = Path(ocr_json_path_).with_name(
            Path(ocr_json_path_).stem + "_with_paddle.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(augmented_ocrrun.model_dump(), f, indent=2, ensure_ascii=False)

        return JSONResponse(content={
            "message": "Augmentation complete",
            "output_file": str(out_path),
            "count": len(augmented_ocrrun.imageResults) if augmented_ocrrun.imageResults else 0,
            "mode": "process_and_save" if use_process_and_save else "minimal"
        })

    except Exception as e:
        log_exception("augment_json_with_paddle")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------- Run with: ---------------
# uvicorn paddleocr_server:app --host 0.0.0.0 --port 6002
