from typing import Any


# For this particular function, we don't need to provide a body for the
# function, since the OpenBB app will handle the function call. We just use this
# to get the underlying LLM to follow the schema.
async def llm_get_widget_data(widget_uuids: list[str]) -> Any:
    """Retrieve the data for a list of widgets by specifying their UUIDs."""
    ...
