from typing import AsyncGenerator
from openbb_ai.models import (
    QueryRequest,
    Citation,
    DataSourceRequest,
    LlmClientFunctionCallResultMessage,
    SourceInfo,
)

import logging


logger = logging.getLogger(__name__)


async def cite_widget(
    function_call_result: LlmClientFunctionCallResultMessage, request: QueryRequest
) -> AsyncGenerator[Citation, None]:
    data_source_requests = [
        DataSourceRequest(**data_source)
        for data_source in function_call_result.input_arguments.get("data_sources", [])
    ]
    all_widgets = (
        request.widgets.primary + request.widgets.secondary if request.widgets else []
    )

    for data_source_request in data_source_requests:
        widget = next(
            (
                w
                for w in all_widgets
                if str(w.uuid) == str(data_source_request.widget_uuid)
            ),
            None,
        )
        if not widget:
            logger.warning(
                f"Widget not found while trying to create citation: {data_source_request.widget_uuid}"
            )
            continue
        else:
            yield Citation(
                source_info=SourceInfo(
                    type="widget",
                    origin=widget.origin,
                    widget_id=widget.widget_id,
                    metadata={
                        "input_args": data_source_request.input_args,
                    },
                ),
                details=[
                    {
                        "Widget Origin": widget.origin,
                        "Widget Name": widget.name,
                        "Widget ID": widget.widget_id,
                        **data_source_request.input_args,
                    }
                ],
            )
