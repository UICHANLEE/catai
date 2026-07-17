from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "phone_gateway"
    / "proxy_server.py"
)
SPEC = importlib.util.spec_from_file_location("phone_gateway_proxy", MODULE_PATH)
assert SPEC and SPEC.loader
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


class PhoneGatewayValidationTest(unittest.TestCase):
    def test_api_key_is_required_and_compared(self) -> None:
        with patch.object(gateway, "GATEWAY_API_KEY", "expected"):
            gateway.verify_api_key("expected")
            with self.assertRaises(HTTPException) as raised:
                gateway.verify_api_key("wrong")
        self.assertEqual(raised.exception.status_code, 401)

    def test_multipart_requires_boundary(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            gateway.validate_content_type("multipart/form-data")
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(
            gateway.validate_content_type("multipart/form-data; boundary=abc"),
            "multipart/form-data; boundary=abc",
        )

    def test_invalid_and_oversized_content_lengths_are_rejected(self) -> None:
        for value, status_code in (("-1", 400), ("bad", 400)):
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as raised:
                    gateway.validate_content_length(value)
                self.assertEqual(raised.exception.status_code, status_code)

        with patch.object(gateway, "MAX_REQUEST_BYTES", 3):
            with self.assertRaises(HTTPException) as raised:
                gateway.validate_content_length("4")
        self.assertEqual(raised.exception.status_code, 413)

    def test_streaming_limit_rejects_chunked_oversize_body(self) -> None:
        chunks = iter((b"ab", b"cd"))

        async def receive() -> dict[str, object]:
            try:
                body = next(chunks)
            except StopIteration:
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": body, "more_body": True}

        request = Request({"type": "http", "method": "POST", "path": "/"}, receive)

        async def consume() -> None:
            with patch.object(gateway, "MAX_REQUEST_BYTES", 3):
                async for _ in gateway.limited_body(request):
                    pass

        with self.assertRaises(gateway.UploadTooLarge):
            asyncio.run(consume())


if __name__ == "__main__":
    unittest.main()
