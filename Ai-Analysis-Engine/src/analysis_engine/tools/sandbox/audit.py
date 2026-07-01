# src/analysis_engine/tools/sandbox/audit.py
"""
Audit logging for sandbox executions.

All sandbox code executions are logged with:
- The code that was run
- Whether it passed validation
- The result (success/failure)
- Execution time
- The LLM's stated purpose

This is CRITICAL for:
1. Debugging when things go wrong
2. Security review
3. Compliance requirements
4. Improving guardrails based on attack patterns
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


@dataclass
class SandboxAuditEntry:
    """Immutable record of a sandbox execution."""
    timestamp: str
    code_hash: str
    code_length: int
    purpose: str
    validation_passed: bool
    validation_errors: Optional[str]
    execution_success: bool
    execution_error: Optional[str]
    execution_time_seconds: float
    output_length: int
    run_id: str
    agent: str  # "analysis", "cleaning", etc.


class SandboxAuditLogger:
    """Logs sandbox executions to a file (append-only for immutability)."""
    
    def __init__(self, log_dir: str = "logs/sandbox"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # One log file per day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.log_path = self.log_dir / f"sandbox_{today}.jsonl"
    
    def log(self, entry: SandboxAuditEntry):
        """Append an audit entry to the log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
    
    def log_execution(
        self,
        code: str,
        purpose: str,
        validation_passed: bool,
        validation_errors: Optional[str],
        execution_success: bool,
        execution_error: Optional[str],
        execution_time: float,
        output: str,
        run_id: str,
        agent: str = "analysis",
    ):
        """Convenience method to log a full execution."""
        entry = SandboxAuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            code_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
            code_length=len(code),
            purpose=purpose,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
            execution_success=execution_success,
            execution_error=execution_error,
            execution_time_seconds=execution_time,
            output_length=len(output),
            run_id=run_id,
            agent=agent,
        )
        self.log(entry)


# Module-level logger
_logger: Optional[SandboxAuditLogger] = None


def get_audit_logger() -> SandboxAuditLogger:
    global _logger
    if _logger is None:
        _logger = SandboxAuditLogger()
    return _logger