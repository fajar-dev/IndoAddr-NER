from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import NerModel
from .schemas import ExtractRequest, ExtractResponse, Entity
from .normalizer import normalize_components
from .config import settings
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="IndoAddr-NER", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global model instance (simple pattern); for production, use dependency injection / lifecycle events
_model = None

def get_model():
    global _model
    if _model is None:
        _model = NerModel(model_dir=settings.model_dir, model_name=settings.model_name, device=settings.device, aggregation_strategy=settings.aggregation_strategy)
    return _model

@app.on_event("startup")
def startup_event():
    # pre-load model
    try:
        get_model()
        logger.info("Model loaded")
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)

@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    model = get_model()
    try:
        ents = model.predict(req.text)
        normalized = normalize_components(ents)
        # shape entities to response model
        entities = []
        for e in ents:
            entities.append(Entity(entity=e.get('entity_group') or e.get('entity'), text=e.get('word') or e.get('text'), score=float(e.get('score', 0.0))))
        resp = ExtractResponse(**normalized, entities=entities, raw={"pipeline": ents})
        return resp
    except Exception as e:
        logger.exception("Error during extraction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
