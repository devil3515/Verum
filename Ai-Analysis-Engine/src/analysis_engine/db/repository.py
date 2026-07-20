from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from analysis_engine.db.engine import get_session
from analysis_engine.db.models import (
    Run, RunEvent, Claim, Chart, CleaningLogEntry,
    ChatSession as DBChatSession, ChatMessage as DBChatMessage,
)


#--Runs-----------------

def persist_run_started(
        run_id: str,
        filename: str,
        question: str,
        upload_ref: str,
        user_id: Optional[str] = None,
) -> None:
    with get_session() as session:
        run = Run(
            id=run_id,
            user_id=user_id,
            filename=filename,
            question=question,
            status="running",
            uploaded_ref=upload_ref,
        )
        session.add(run)
        session.commit()


def persist_run_completed(
        run_id: str,
        cleaned_ref: str,
        report: str,
        claims: list[dict],
        chart_refs: list[str],
        cleaning_logs: list[dict],
) -> None:
    with get_session() as session:
        run = session.query(Run).filter_by(id=run_id).first()
        if not run:
            return

        run.status = "done"
        run.completed_at = datetime.now(timezone.utc)
        run.cleaned_ref = cleaned_ref

        for c in claims:
            session.add(Claim(
                run_id = run_id,
                text = c.get("text", ""),
                metric = c.get("metric", ""),
                value = c.get("value", 0.0),
                source_query = c.get("source_query", ""),
                source_columns = c.get("source_columns", []),
                verification_status = c.get("verification_status", "unverified"),
                confidence = c.get("confidence", 0.0),
                web_sources = c.get("web_sources", []),
            ))

        for ref in chart_refs:
            session.add(Chart(
                run_id = run_id,
                ref = ref,
            ))
        for entry in cleaning_logs:
            session.add(CleaningLogEntry(
                run_id=run_id,
                operation=entry.get("operation", ""),
                column=entry.get("column"),
                rows_affected=entry.get("rows_affected", 0),
                rationale=entry.get("rationale"),
            ))

        session.commit()


def persist_run_failed(run_id: str, error: str) -> None:
    with get_session() as session:
        run = session.query(Run).filter_by(id=run_id).first()
        if run:
            run.status       = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.report       = f"Pipeline failed: {error}"
            session.commit()


def persist_event(run_id: str, event_type: str, data: dict) -> None:
    try:
        with get_session() as session:
            session.add(RunEvent(
                run_id = run_id,
                event_type = event_type,
                data = data,
            ))
        session.commit()
    except Exception as e:
        print(f"[db] failed to persist event: {e}")


def get_events_for_run(run_id: str) -> list[dict]:
    with get_session() as session:
        events = (
            session.query(RunEvent)
            .filter_by(run_id=run_id)
            .order_by(RunEvent.id)
            .all()
        )
        return [
            {
                "event_type": e.event_type,
                "data": e.data,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ]


def get_run(run_id: str) -> Optional[dict]:
    with get_session() as session:
        run = session.query(Run).filter_by(id = run_id).first()
        if not run:
            return None
        return _run_to_dict(run, include_full = True)


def list_runs(user_id: Optional[str] = None) -> list[dict]:
    with get_session() as session:
        q = session.query(Run).order_by(Run.created_at.desc())
        if user_id:
            q = q.filter_by(user_id=user_id)
        return [_run_to_dict(r, include_full=False) for r in q.all()]


def _run_to_dict(run: Run, include_full: bool = False) -> dict:
    base = {
        "run_id":      run.id,
        "filename":    run.filename,
        "question":    run.question,
        "status":      run.status,
        "created_at":  run.created_at.isoformat() if run.created_at else None,
        "claim_count": len(run.claims) if run.claims else 0,
        "chart_count": len(run.charts) if run.charts else 0,
    }
    if not include_full:
        return base

    base.update({
        "report":       run.report or "",
        "cleaned_ref":  run.cleaned_ref,
        "claims": [
            {
                "id":                  c.id,
                "text":                c.text,
                "metric":              c.metric,
                "value":               c.value,
                "source_query":        c.source_query,
                "source_columns":      c.source_columns or [],
                "verification_status": c.verification_status,
                "confidence":          c.confidence,
            }
            for c in (run.claims or [])
        ],
        "chart_refs": [ch.ref for ch in (run.charts or [])],
        "cleaning_log": [
            {
                "operation":     e.operation,
                "column":        e.column,
                "rows_affected": e.rows_affected,
                "rationale":     e.rationale,
            }
            for e in (run.cleaning_log or [])
        ],
        "cleaned_refs": {
            "uploaded-file": run.cleaned_ref
        } if run.cleaned_ref else {},
    })
    return base


# ── Chat ──────────────────────────────────────────────────────────────────────

def persist_chat_session(
    session_id: str,
    run_id: str,
    user_id: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    with get_session() as s:
        s.add(DBChatSession(
            id=session_id,
            run_id=run_id,
            user_id=user_id,
            title=title,
        ))
        s.commit()


def persist_chat_message(
    session_id: str,
    role: str,
    content: str,
    chart_refs: list[str],
    citations: list[dict],
) -> str:
    """Persist one message. Returns its generated ID."""
    import uuid
    msg_id = str(uuid.uuid4())
    with get_session() as s:
        s.add(DBChatMessage(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            chart_refs=chart_refs,
            citations=citations,
        ))
        s.commit()
    return msg_id


def get_chat_history(session_id: str) -> list[dict]:
    with get_session() as s:
        msgs = (
            s.query(DBChatMessage)
            .filter_by(session_id=session_id)
            .order_by(DBChatMessage.created_at)
            .all()
        )
        return [
            {
                "id":         m.id,
                "role":       m.role,
                "content":    m.content,
                "chart_refs": m.chart_refs or [],
                "citations":  m.citations or [],
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ]
