# IndoAddr-NER Parse
Project: NER address extraction for Indonesian addresses (provinsi → RT/RW) using IndoBERT & HuggingFace transformers.
This repository contains:
- FastAPI app with clean OOP structure to serve NER model inference
- Training script (HuggingFace Trainer) to fine-tune IndoBERT on BIO-labelled JSONL
- Utilities for tokenization, normalization, and simple post-processing
- `requirements.txt` and `Dockerfile` example
- Example usage and installation instructions

> **Note**: This package assumes you will train or provide a fine-tuned model under `models/indoaddr-ner`. If not present, the API can use a base IndoBERT checkpoint but results will be poor until fine-tuned.

## Quick start (development)

1. Create a Python virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API locally:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open docs: http://localhost:8000/docs

## Training (basic)

Place your `train.jsonl`, `valid.jsonl`, and `test.jsonl` in a folder (e.g. `data/`). They must follow the JSONL format:
```json
{"tokens": ["Jl.", "Bunga", "Mawar", "No.", "182", ",", "Kel.", "Tanah Enam Ratus", ",", "kecamatankecamatan.", "Medan", "Timur", ",", "Kota", "Medan", ",", "Sumatera", "Utara", "20114", "RT", "07", "RW", "08"], "labels": ["B-STREET", "I-STREET", "I-STREET", "I-STREET", "I-STREET", "O", "O", "B-VILLAGE", "O", "O", "B-DISTRICT", "I-DISTRICT", "O", "B-CITY", "I-CITY", "O", "B-PROVINCE", "I-PROVINCE", "B-POSTALCODE", "B-RT", "I-RT", "B-RW", "I-RW"]}

```

Train with:
```bash
python scripts/train.py --data_dir data --output_dir models/indoaddr-ner --model_name indobenchmark/indobert-base-p1 --epochs 6 --batch_size 16
```

After training, the best model will be saved at `models/indoaddr-ner` and the API will load it by default.

## Usage (API)

POST `/extract` with JSON body:
```json
{"text": "Jl. Perintis Kemerdekaan No. 37, Kel. Gaharu, Kec. Medan Timur, Kota Medan, Sumatera Utara 20235, RT 01 RW 02"}
```

Response:
```json
{
  "provinsi": "Sumatera Utara",
  "kabupaten_kota": "Kota Medan",
  "kecamatan": "Medan Timur",
  "kelurahan_desa": "Gaharu",
  "dusun": null,
  "jalan": "Jalan Perintis Kemerdekaan",
  "nomor": "37",
  "blok": null,
  "rt": "001",
  "rw": "002",
  "kodepos": "20235",
  "entities": [
    {"entity":"JALAN","text":"Jl. Perintis Kemerdekaan","score":0.98}
  ]
}
```

## Project structure
```
indoaddr-ner-fastapi/
├─ app/
│  ├─ main.py               # FastAPI entry
│  ├─ config.py             # Configuration
│  ├─ models.py             # Model wrapper (OOP)
│  ├─ schemas.py            # Pydantic schemas
│  ├─ normalizer.py         # Post-processing normalization utilities
│  └─ __init__.py
├─ scripts/
│  └─ train.py              # Training script (HuggingFace Trainer)
├─ requirements.txt
├─ Dockerfile               # optional containerization example
└─ README.md
```

## Notes & best practices
- Use GPU for training; set `--fp16` in training args if supported.
- Validate dataset labels and tokenization alignment before long training runs.
- Use monitoring tools (Weights & Biases) for larger experiments.
- This repository provides a minimal production-ready pattern but you should add authentication, rate-limiting, and secure model storage for real deployments.
