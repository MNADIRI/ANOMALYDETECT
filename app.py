"""
CT Control Volume — FastAPI server.

Compares two CT scans of the same patient and generates a DICOM SEG
highlighting regions of change.
"""

import asyncio
import json
import os
import shutil
import threading
import uuid
import zipfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingestion import ingest_dicom_folder
from src.registration import (
    register_to_reference,
    apply_transform_to_multichannel,
)
from src.features import load_model, extract_features, reduce_features
from src.scoring import compute_change_scores, upsample_scores
from src.export import create_dicom_seg

app = FastAPI(title="CT Control Volume")

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    ref_path: str
    new_path: str
    threshold: float = 3.0


# ---------------------------------------------------------------------------
# Pipeline runner (runs in a background thread)
# ---------------------------------------------------------------------------

def run_pipeline_job(job_id: str, ref_path: str, new_path: str, threshold: float):
    try:
        update = lambda p, m: jobs[job_id].update(
            {"progress": p, "message": m}
        )

        # Phase 1: Ingest reference
        update(5, "Reading reference scan...")
        vol_ref_3ch, vol_ref_hu, meta_ref = ingest_dicom_folder(ref_path)

        # Phase 1b: Ingest new scan
        update(15, "Reading new scan...")
        vol_new_3ch, vol_new_hu, meta_new = ingest_dicom_folder(new_path)

        # Phase 2: Registration
        update(25, "Spatial registration...")
        vol_ref_hu_reg, transform, fixed_image = register_to_reference(
            vol_new_hu, vol_ref_hu, meta_new, meta_ref
        )

        update(30, "Applying transform to 3-channel volume...")
        vol_ref_3ch_reg = apply_transform_to_multichannel(
            vol_ref_3ch, transform, fixed_image, meta_ref
        )

        # Phase 3: Feature extraction
        update(35, "Loading DINOv2 model...")
        model, device, patch_size, n_register = load_model()

        update(40, "Extracting features - reference...")
        feat_ref = extract_features(
            model, vol_ref_3ch_reg, device,
            patch_size=patch_size,
            n_register=n_register,
            progress_callback=lambda p: update(
                40 + p * 15, f"Features reference: {int(p * 100)}%"
            ),
        )

        update(60, "Extracting features - new scan...")
        feat_new = extract_features(
            model, vol_new_3ch, device,
            patch_size=patch_size,
            n_register=n_register,
            progress_callback=lambda p: update(
                60 + p * 15, f"Features new scan: {int(p * 100)}%"
            ),
        )

        # Free the model from memory
        del model
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # PCA
        update(78, "Dimensionality reduction (PCA)...")
        feat_ref_r, feat_new_r = reduce_features(feat_ref, feat_new)
        del feat_ref, feat_new

        # Phase 4: Scoring
        update(82, "Computing anomaly scores...")
        z_scores = compute_change_scores(
            feat_new_r, feat_ref_r,
            volume_hu_new=vol_new_hu,
            patch_size=patch_size,
        )
        del feat_ref_r, feat_new_r

        update(87, "Upsampling scores to native resolution...")
        print(f"  Feature grid shape: {z_scores.shape}")
        print(f"  Target (original) shape: {meta_new['original_shape']}")
        z_scores_full = upsample_scores(z_scores, meta_new["original_shape"])
        del z_scores

        import numpy as _np
        print(f"  Upsampled z-scores: min={z_scores_full.min():.2f}, "
              f"max={z_scores_full.max():.2f}, "
              f"above threshold ({threshold}): "
              f"{(z_scores_full > threshold).sum()}/{z_scores_full.size} voxels")

        # Phase 5: Export
        update(90, "Generating DICOM SEG file...")
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"{job_id}.dcm")
        create_dicom_seg(
            z_scores_full,
            meta_new["source_files"],
            threshold=threshold,
            output_path=output_path,
        )

        jobs[job_id].update(
            {
                "status": "done",
                "progress": 100,
                "message": "Done!",
                "output_path": output_path,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({"status": "error", "message": f"Error: {e}"})


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Start analysis from local DICOM folder paths."""
    if not os.path.isdir(request.ref_path):
        raise HTTPException(400, f"Reference path not found: {request.ref_path}")
    if not os.path.isdir(request.new_path):
        raise HTTPException(400, f"New scan path not found: {request.new_path}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting...",
        "output_path": None,
    }

    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, request.ref_path, request.new_path, request.threshold),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.post("/api/analyze_zip")
async def analyze_zip(
    ref_zip: UploadFile = File(...),
    new_zip: UploadFile = File(...),
    threshold: float = Form(3.0),
):
    """Start analysis from uploaded ZIP files."""
    job_id = str(uuid.uuid4())
    upload_dir = os.path.join("uploads", job_id)
    ref_dir = os.path.join(upload_dir, "ref")
    new_dir = os.path.join(upload_dir, "new")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    # Save and extract ZIPs
    for zip_file, dest_dir in [(ref_zip, ref_dir), (new_zip, new_dir)]:
        zip_path = os.path.join(upload_dir, zip_file.filename or "upload.zip")
        with open(zip_path, "wb") as f:
            content = await zip_file.read()
            f.write(content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)

    # Find the actual DICOM directory (may be nested)
    ref_dicom = _find_dicom_dir(ref_dir)
    new_dicom = _find_dicom_dir(new_dir)

    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting...",
        "output_path": None,
    }

    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, ref_dicom, new_dicom, threshold),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


def _find_dicom_dir(root: str) -> str:
    """Find directory containing .dcm files (may be nested inside ZIP)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".dcm") or "." not in f:
                return dirpath
    return root


@app.get("/api/progress/{job_id}")
async def progress(job_id: str):
    """SSE endpoint streaming job progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        last_progress = -1
        while True:
            job = jobs.get(job_id)
            if job is None:
                break

            current = job.get("progress", 0)
            if current != last_progress or job["status"] in ("done", "error"):
                data = json.dumps(
                    {
                        "status": job["status"],
                        "progress": job.get("progress", 0),
                        "message": job.get("message", ""),
                    }
                )
                yield f"data: {data}\n\n"
                last_progress = current

            if job["status"] in ("done", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    """Download the generated DICOM SEG file."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete yet")
    if not job.get("output_path") or not os.path.exists(job["output_path"]):
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        job["output_path"],
        media_type="application/dicom",
        filename="ct_control_volume.dcm",
    )


# ---------------------------------------------------------------------------
# Serve static files and index
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
