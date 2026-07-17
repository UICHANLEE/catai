#!/usr/bin/env python3
"""Export CashLog feedback from Supabase into a de-identified release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.feedback import export_feedback_release, load_leaf_ids, read_jsonl  # noqa: E402


DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
SELECT_FIELDS = ",".join(
    [
        "schema_version",
        "event_id",
        "sample_id",
        "user_id",
        "model_version",
        "taxonomy_version",
        "model_category",
        "predicted_top3",
        "selected_leaf_id",
        "occurred_at",
        "source",
        "image_retention_consent",
        "image_object_key",
        "review_status",
        "reviewed_at",
    ]
)


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


NO_REDIRECT_OPENER = build_opener(RejectRedirects)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument(
        "--service-role-key-env", default="SUPABASE_SERVICE_ROLE_KEY"
    )
    parser.add_argument(
        "--hmac-key-env", default="CASHLOG_FEEDBACK_HMAC_KEY"
    )
    parser.add_argument(
        "--status",
        choices=["all", "pending", "approved", "rejected"],
        default="all",
    )
    parser.add_argument("--page-size", type=int, default=500)
    return parser.parse_args()


def fetch_supabase_rows(
    *,
    supabase_url: str,
    service_role_key: str,
    status: str,
    page_size: int,
) -> list[dict[str, Any]]:
    if not 1 <= page_size <= 1000:
        raise ValueError("page-size must be between 1 and 1000")
    parsed_url = urlparse(supabase_url)
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise ValueError("Supabase URL must not contain credentials, query, or fragment")
    if parsed_url.scheme != "https" and parsed_url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Supabase URL must use HTTPS outside local development")
    endpoint = supabase_url.rstrip("/") + "/rest/v1/cashlog_category_feedback"
    params = {"select": SELECT_FIELDS, "order": "occurred_at.asc,event_id.asc"}
    if status != "all":
        params["review_status"] = f"eq.{status}"
    url = endpoint + "?" + urlencode(params)
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        request = Request(
            url,
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
                "Accept": "application/json",
                "Range-Unit": "items",
                "Range": f"{start}-{start + page_size - 1}",
                "User-Agent": "CataiFeedbackExporter/1.0",
            },
        )
        with NO_REDIRECT_OPENER.open(request, timeout=30) as response:
            if urlparse(response.geturl()).netloc != parsed_url.netloc:
                raise ValueError("Supabase export refused a cross-origin redirect")
            payload = response.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("Supabase feedback page exceeded 8 MiB")
        page = json.loads(payload)
        if not isinstance(page, list):
            raise ValueError("Supabase feedback response must be an array")
        rows.extend(page)
        if len(page) < page_size:
            return rows
        start += page_size


def main() -> None:
    args = parse_args()
    hmac_key_value = os.getenv(args.hmac_key_env)
    if not hmac_key_value:
        raise SystemExit(f"missing required environment variable: {args.hmac_key_env}")
    existing_files = [
        args.output_dir / "events.jsonl",
        args.output_dir / "secure_image_index.jsonl",
        args.output_dir / "quarantine.jsonl",
        args.output_dir / "export_summary.json",
    ]
    if args.reuse_existing and all(path.is_file() for path in existing_files):
        summary = json.loads(existing_files[-1].read_text(encoding="utf-8"))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        print(args.output_dir.resolve())
        return
    if args.input_jsonl:
        rows = read_jsonl(args.input_jsonl)
    else:
        service_role_key = os.getenv(args.service_role_key_env)
        if not args.supabase_url:
            raise SystemExit("SUPABASE_URL or --supabase-url is required")
        if not service_role_key:
            raise SystemExit(
                f"missing required environment variable: {args.service_role_key_env}"
            )
        rows = fetch_supabase_rows(
            supabase_url=args.supabase_url,
            service_role_key=service_role_key,
            status=args.status,
            page_size=args.page_size,
        )
    summary = export_feedback_release(
        rows,
        output_dir=args.output_dir,
        leaf_ids=load_leaf_ids(args.categories),
        hmac_key=hmac_key_value.encode("utf-8"),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
