import uuid
from dataclasses import dataclass

@dataclass
class ToolResult:
    tool_name: str
    output: str
    chart_spec: dict | None = None
    chart_ref: str = ""

    def __post_init__(self):
        if self.chart_spec and not self.chart_ref:
            self.chart_ref = f"chart-{uuid.uuid4()}.json"
