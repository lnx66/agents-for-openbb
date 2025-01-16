from typing import Any
from pydantic import BaseModel


class HandledContext(BaseModel):
    context_prompt_str: str
    loaded_context: dict[str, Any]
