from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    case_id: str
    status: str