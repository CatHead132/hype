#!/usr/bin/env bash
# Local testing server for the patched game - no GitHub push needed.
# Open http://localhost:8000/index.local.html after running this.
cd "$(dirname "$0")"
python3 -m http.server 8000
