#!/usr/bin/env python3
"""
Burn a compressed JSON payload into the HTML viewer.

The script gzips the provided JSON file, base64-encodes it, and injects it
into the HTML as a script block that `index.html` knows how to read.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import re
from pathlib import Path


BURNED_SCRIPT_ID = "burned-json-data"
PAYLOAD_VAR = "__BURNED_JSON_GZIP_BASE64__"
DATA_URL_VAR = "__BURNED_JSON_DATA_URL__"


def build_script_block(
    b64_payload: str,
    raw_b64: str,
    source_name: str,
    raw_size: int,
    compressed_size: int,
) -> str:
    """Return the script block that stores the compressed and raw payload on window."""
    data_url = f"data:application/json;base64,{raw_b64}"
    return (
        f"\n<script id=\"{BURNED_SCRIPT_ID}\">\n"
        f"window.__BURNED_JSON_SOURCE__ = \"{source_name}\";\n"
        f"window.__BURNED_JSON_SIZE_BYTES__ = {raw_size};\n"
        f"window.__BURNED_JSON_GZIP_SIZE_BYTES__ = {compressed_size};\n"
        f'window.{PAYLOAD_VAR} = "{b64_payload}";\n'
        f'window.{DATA_URL_VAR} = "{data_url}";\n'
        "</script>\n"
    )


def inject_script(html: str, script_block: str) -> str:
    """Insert or replace the burned script block inside the HTML."""
    pattern = re.compile(
        rf"<script id=\"{re.escape(BURNED_SCRIPT_ID)}\">.*?</script>",
        re.DOTALL,
    )
    if pattern.search(html):
        return pattern.sub(script_block, html, count=1)

    closing_body = "</body>"
    idx = html.lower().rfind(closing_body)
    if idx != -1:
        return html[:idx] + script_block + html[idx:]

    return html + script_block


def burn(html_path: Path, json_path: Path, output_path: Path) -> tuple[int, int, int]:
    raw_bytes = json_path.read_bytes()
    raw_b64 = base64.b64encode(raw_bytes).decode("ascii")
    compressed = gzip.compress(raw_bytes, compresslevel=9)
    b64_payload = base64.b64encode(compressed).decode("ascii")

    html = html_path.read_text(encoding="utf-8")
    script_block = build_script_block(
        b64_payload=b64_payload,
        raw_b64=raw_b64,
        source_name=json_path.name,
        raw_size=len(raw_bytes),
        compressed_size=len(compressed),
    )
    burned_html = inject_script(html, script_block)
    output_path.write_text(burned_html, encoding="utf-8")
    return len(raw_bytes), len(compressed), len(raw_b64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Burn compressed JSON into an HTML viewer.")
    parser.add_argument(
        "--html",
        default="index.html",
        type=Path,
        help="Path to the HTML viewer file (default: index.html).",
    )
    parser.add_argument(
        "--json",
        required=True,
        type=Path,
        help="Path to the JSON payload to embed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output HTML path (default: overwrite --html in place).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path: Path = args.html
    json_path: Path = args.json
    output_path: Path = args.out or html_path

    if not html_path.exists():
        raise SystemExit(f"HTML file not found: {html_path}")
    if not json_path.exists():
        raise SystemExit(f"JSON file not found: {json_path}")

    raw_size, compressed_size, raw_b64_size = burn(html_path, json_path, output_path)
    print(
        f"Embedded {json_path.name} into {output_path} "
        f"(raw: {raw_size} bytes, gzip: {compressed_size} bytes, base64: {raw_b64_size} chars)."
    )


if __name__ == "__main__":
    main()
