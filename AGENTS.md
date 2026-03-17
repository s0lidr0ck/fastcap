# AGENTS.md

## Cursor Cloud specific instructions

FastCaption (FastCap) is a standalone Python desktop application (PySide6/Qt6) for video captioning and sermon processing. There are no web services, databases, or containers — it is purely a desktop tool that calls external APIs.

### Running the app

- **GUI:** `source .venv/bin/activate && python FastCap.pyw`
- **CLI (captioning):** `source .venv/bin/activate && python caption_video.py <video> [options]`
- **CLI (clip extraction):** `source .venv/bin/activate && python extract_clips.py <video> <json> [options]`
- See `README.md` for full CLI options and usage.

### Qt / PySide6 in headless environments

The GUI requires a display. In headless Cloud VMs:
- Install `libxkbcommon-x11-0`, `libxcb-cursor0`, `libxcb-icccm4`, `libxcb-keysyms1`, `libegl1`, `libgl1` for xcb support.
- Start `Xvfb :99 -screen 0 1280x1024x24 &` and set `DISPLAY=:99`, or use `QT_QPA_PLATFORM=offscreen` for non-visual testing.
- PulseAudio and pipewire warnings are cosmetic and can be ignored.

### Linting

No linter config ships with the repo. Use `flake8 --max-line-length=120` for basic style checks. Existing code has some style warnings (long lines, unused imports) — do not fix these unless explicitly asked.

### Tests

There are no automated test suites in this repo. `test_wix_api.py` is a manual integration test that requires live Wix API credentials.

### External services (all optional for local dev)

- **AWS Bedrock** (LLM) — requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` in `.env`
- **Ollama** (local LLM) — run Ollama server on `http://127.0.0.1:11434`
- **Wix APIs** — requires `WIX_BEARER_TOKEN`, `WIX_SITE_ID`, `WIX_COLLECTION_ID` in `.env`
- Copy `.env.example` to `.env` and fill in values as needed.
