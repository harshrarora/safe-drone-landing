from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline import MCDropoutPipeline

app = FastAPI()

# ✅ CORS – browser safe
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Preflight handler (THIS FIXES FRONTEND)
@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(status_code=200)

pipeline = MCDropoutPipeline()

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    image_path = f"temp_{image.filename}"

    with open(image_path, "wb") as f:
        f.write(await image.read())

    result = pipeline.process(image_path)

    safety = result.get("safety_assessment", {})
    raw_risks = safety.get("risk_factors", [])

    risk_factors = []
    for r in raw_risks:
        if isinstance(r, dict):
            risk_factors.append({
                "type": r.get("type", "unknown"),
                "severity": r.get("severity", "low"),
            })
        else:
            risk_factors.append({
                "type": str(r),
                "severity": "low",
            })

    return {
        "final_decision": result.get("final_decision", "ABORT"),
        "final_score": float(result.get("final_score", 0.0)),
        "uncertainty": {
            "consensus": float(result["uncertainty"].get("consensus", 0.0)),
            "bbox_uncertainty": float(result["uncertainty"].get("bbox_uncertainty", 0.0)),
        },
        "risk_factors": risk_factors,
    }
