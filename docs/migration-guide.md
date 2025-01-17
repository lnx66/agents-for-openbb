# Migration Guide

## 2025-01-16

A major change has been made to the copilot protocol and API schema.

All example copiots have been updated to reflect the new schema and changes.

### Widgets are now passed in as `primary`, `secondary`, and (optionally) `extra` lists
Previously, widgets on the currently-active dashboard were passed in as a single list. This has now changed, as has the schema for the widget objects.

**OLD**
```json
// Old schema
{
    ...,
    "widgets": [
        {
            "uuid": "91ae1153-7b4d-451c-adc5-49856e90a0e6",
            "name": "Stock Price", 
            "description": "Get the stock price of a company.",
            "metadata": {"symbol": "TSLA"}
        },
        {
            "uuid": "b2c45d67-8e9f-4a12-bc3d-123456789abc",
            "name": "Company Info",
            "description": "Get company information and key metrics.",
            "metadata": {"symbol": "TSLA"}
        }
    ]
}
```

**NEW**
```json
// New schema
{
    ...,
    "widgets": {
        "primary": [ // List of explicitly-added widgets with top priority
            {
                "uuid": "91ae1153-7b4d-451c-adc5-49856e90a0e6",
                "origin": "OpenBB API",
                "widget_id": "stock_price",
                "name": "Stock Price",
                "description": "Get real-time stock price data for any ticker",
                "params": [
                    {
                        "name": "symbol",
                        "type": "string",
                        "description": "Stock ticker symbol",
                        "current_value": "TSLA"
                    }
                ],
                "metadata": {}
            }
        ],
        "secondary": [ // List of dashboard widgets with second-highest priority
            {
                "uuid": "b2c45d67-8e9f-4a12-bc3d-123456789abc",
                "origin": "OpenBB API",
                "widget_id": "company_news",
                "name": "Company News",
                "description": "Get latest news articles about a company",
                "params": [
                    {
                        "name": "symbol",
                        "type": "string",
                        "description": "Stock ticker symbol",
                        "current_value": "TSLA"
                    },
                    {
                        "name": "start_date",
                        "type": "string",
                        "description": "Start date for news articles (YYYY-MM-DD)",
                        "current_value": "2024-01-01"
                    }
                ],
                "metadata": {}
            }
        ],
        "extra": [ // All other possible widgets (only present if the global toggle is enabled)
            {
                "uuid": "f7d23e89-1a2b-4c3d-9e8f-567890abcdef",
                "origin": "OpenBB API",
                "widget_id": "historical_price",
                "name": "Historical Stock Price",
                "description": "Get historical stock price data with customizable date range",
                "params": [
                    {
                        "name": "symbol",
                        "type": "string", 
                        "description": "Stock ticker symbol",
                        "default_value": "AAPL"
                    },
                    {
                        "name": "start_date",
                        "type": "date",
                        "description": "Start date for historical data (YYYY-MM-DD)",
                        "default_value": "2024-01-01"
                    },
                    {
                        "name": "end_date", 
                        "type": "date",
                        "description": "End date for historical data (YYYY-MM-DD)",
                        "default_value": "2024-12-31"
                    }
                ],
                "metadata": {"source": "market_data"}
            }
        ]
    }
}
```

As shown above, widgets are passed in order of priority. The `primary` list is
the highest priority (explicitly added by users as context), followed by
`secondary` (widgets on the currently-active dashboard), and then `extra` (all
other possible widgets, if the user has enabled the global toggle).

Widget objects also now have the following schema:
```json
// Widget object schema
{
    // str - Origin of the widget (eg. OpenBB API, or the name given to a custom backend)
    "origin": "string",
    
    // str - Endpoint ID of the widget (eg. `earnings_transcripts`)
    "widget_id": "string", 
    
    // str - Name of the widget (eg. "Earnings Transcripts")
    "name": "string",
    
    // str - Description of the widget (eg. "Get earnings transcripts for a company")
    "description": "string",
    
    // list[WidgetParam] - List of parameters for the widget
    "params": [],
    
    // dict[str, Any] - Metadata for the widget
    "metadata": {}
}
```

Each input argument (or parameter) for a widget is passed as a list, where each param object has the following schema:

```json
// WidgetParam object schema
{
    // str - Name of the parameter (eg. "ticker")
    "name": "string",
    
    // str - Type of the parameter (eg. "string", "integer", etc.)
    "type": "string",
    
    // str - Description of the parameter (eg. "Stock ticker symbol")
    "description": "string",
    
    // Any | None - Default value of the parameter. Can be null or undefined
    "default_value": "any",
    
    // Any | None - Current value of the parameter. Will not be set for 'extra' widgets
    "current_value": "any",
    
    // list[Any] | None - Optional list of possible values for enumeration parameters
    "options": ["any"]
}
```

### In order to access widget data, you must now explicitly respond with a function call to the client.

Previously, the client app would send widget data in the `context` field of the
request. This is no longer the case. The `context` field is now reserved for
artifacts that are returned to the client as part of the copilot response.

In order to submit a function call, you must respond with a `copilotFunctionCall` SSE:

```
// Example
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"data_sources":[{"origin":"OpenBB API","id":"company_news","input_args":{"symbol":"AAPL","channels":"All","start_date":"2021-01-16","end_date":"2025-01-16","topics":"","limit":50}}]},"copilot_function_call_arguments":{"data_sources":[{"origin":"OpenBB API","widget_id":"company_news"}]}}
```

As a result, we suggest using an async `execution_loop` function that yields all
SSEs back to the client (see the `example_copilot` for an example implementation)
