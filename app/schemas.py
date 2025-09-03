from pydantic import BaseModel
from typing import Optional, List

class ExtractRequest(BaseModel):
    text: str

class Entity(BaseModel):
    entity: str
    text: str
    score: float

class ExtractResponse(BaseModel):
    province: Optional[str]
    city: Optional[str]
    district: Optional[str]
    village: Optional[str]
    street: Optional[str]
    rt: Optional[str]
    rw: Optional[str]
    postalcode: Optional[str]
    entities: List[Entity] = []
    # raw: Dict[str, Any] = {}  # kalau mau simpan hasil full dari model
