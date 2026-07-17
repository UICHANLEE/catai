#!/usr/bin/env python3
"""Apply explicit human review decisions to pending CashLog feedback events."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


NO_REDIRECT_OPENER = build_opener(RejectRedirects)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument(
        "--service-role-key-env", default="SUPABASE_SERVICE_ROLE_KEY"
    )
    return parser.parse_args()


def read_decisions(path: Path) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    event_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = str(uuid.UUID(str(row.get("event_id") or "")))
        decision = str(row.get("decision") or "")
        if decision not in {"approved", "rejected"}:
            raise ValueError(f"line {line_number}: decision must be approved or rejected")
        if event_id in event_ids:
            raise ValueError(f"line {line_number}: duplicate event_id")
        event_ids.add(event_id)
        decisions.append({"event_id": event_id, "decision": decision})
    return decisions


def apply_decision(
    *,
    supabase_url: str,
    service_role_key: str,
    event_id: str,
    decision: str,
) -> None:
    parsed_url = urlparse(supabase_url)
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise ValueError("Supabase URL must not contain credentials, query, or fragment")
    if parsed_url.scheme != "https" and parsed_url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Supabase URL must use HTTPS outside local development")
    query = urlencode(
        {
            "event_id": f"eq.{event_id}",
            "review_status": "eq.pending",
            "select": "event_id,review_status",
        }
    )
    request = Request(
        supabase_url.rstrip("/") + "/rest/v1/cashlog_category_feedback?" + query,
        method="PATCH",
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
            "User-Agent": "CataiFeedbackReviewer/1.0",
        },
        data=json.dumps(
            {
                "review_status": decision,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
        ).encode(),
    )
    with NO_REDIRECT_OPENER.open(request, timeout=30) as response:
        if urlparse(response.geturl()).netloc != parsed_url.netloc:
            raise ValueError("Supabase review refused a cross-origin redirect")
        payload = json.loads(response.read(1024 * 1024 + 1))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"event was not pending or did not exist: {event_id}")


def main() -> None:
    args = parse_args()
    service_role_key = os.getenv(args.service_role_key_env)
    if not args.supabase_url:
        raise SystemExit("SUPABASE_URL or --supabase-url is required")
    if not service_role_key:
        raise SystemExit(f"missing required environment variable: {args.service_role_key_env}")
    decisions = read_decisions(args.decisions)
    for row in decisions:
        apply_decision(
            supabase_url=args.supabase_url,
            service_role_key=service_role_key,
            event_id=row["event_id"],
            decision=row["decision"],
        )
    print(json.dumps({"reviewed_events": len(decisions)}, sort_keys=True))


if __name__ == "__main__":
    main()
