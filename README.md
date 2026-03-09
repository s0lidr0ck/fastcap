# FastCaption

Closed-caption video by extracting audio with **ffmpeg** and transcribing with **faster-whisper**. Outputs SRT (and optionally VTT) caption files.

## Requirements

- **Python 3.9+**
- **ffmpeg** installed and on your PATH ([download](https://ffmpeg.org/download.html))
- Optional: NVIDIA GPU + CUDA for faster transcription

## Setup

```bash
cd c:\PROJECTS\A18\FastCaption
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic: caption a video (writes video_name.srt)
python caption_video.py path/to/video.mp4

# Custom output path
python caption_video.py video.mp4 -o output/captions

# Also create .vtt
python caption_video.py video.mp4 --vtt

# Larger model (more accurate, slower)
python caption_video.py video.mp4 --model medium

# Force language (e.g. English)
python caption_video.py video.mp4 --language en

# Keep the extracted .wav after captioning
python caption_video.py video.mp4 --keep-audio
```

## Options

| Option | Description |
|--------|-------------|
| `video` | Path to input video file |
| `-o, --output` | Base path for caption files (default: same name as video) |
| `--model` | Model size: tiny, base, small, medium, large-v3, large-v3-turbo (default: base) |
| `--device` | auto, cpu, cuda (default: auto) |
| `--language` | Language code (e.g. en, es); auto-detect if omitted |
| `--vtt` | Also write a .vtt file |
| `--keep-audio` | Keep the extracted WAV file |

## Output

- **.srt** – SubRip captions (default), compatible with most players and editors.
- **.vtt** – WebVTT (when using `--vtt`), for web players.

---

## Desktop app (FastCap)

Run the unified desktop app for captioning, ranking moments, preview, clip extraction, and sermon metadata:

```bash
python extract_clips_gui.py
```

Tabs: **Caption Video**, **Rank Moments**, **Preview Report**, **Extract Clips**, **Sermon Metadata**, **Blog Post**.

### Sermon Metadata tab

1. Paste or load a sermon transcript (SRT/VTT text or plain text).
2. Choose the same AI model used elsewhere (Bedrock or Ollama). Optionally set host (e.g. `http://127.0.0.1:11434` for Ollama or leave blank for Bedrock).
3. Click **Generate Metadata**. The app calls the AI with a strict JSON prompt (SCRIBE); response is parsed and validated.
4. After success, **Send to Wix** is enabled. Click it to create a new item in your Wix CMS Sermons collection (create-only; no update/upsert).

**Speaker aliases:** Use the **Speaker** dropdown to pick a preconfigured speaker. The **Name (saved to Wix)** field stores the plain name (e.g. "Misti Sanders"); the AI prompt uses the reference name (e.g. "Sis. Misti"). Edit `speaker_aliases.json` in the project root to add or change entries: each object has `"canonical"` (saved to Wix) and `"promptName"` (how the AI refers to them).

**Required environment variables for Wix (add to `.env`):**

| Variable | Description |
|----------|-------------|
| `WIX_BEARER_TOKEN` | Pre-issued Wix API bearer token (e.g. from OAuth or token flow). |
| `WIX_COLLECTION_ID` | Your Wix Data collection ID (GUID) for the Sermons collection. |
| `WIX_SITE_ID` | Your Wix site ID (MetaSite ID). Find it in the site dashboard URL after `/dashboard/` or in Wix Developer Center. Required for site-level Data API calls. |
| `WIX_API_BASE` | Optional. Default: `https://www.wixapis.com`. |

**Strict JSON contract (AI output):** The model is prompted to return a single JSON object with: `title`, `description`, `scriptures`, `mainPoints`, `tags`, `propheticStatements`, `keyMoments`, `topics`, `teachingStatements`. Each `keyMoments` entry must have `timestamp`, `quote`, and `explanation`. Tags must be chosen from the provided tag list. The parser normalizes and validates; missing required fields raise an error and the raw response is shown so you can fix the prompt or model.

### Blog Post tab

1. Paste or load a sermon transcript (same as Sermon Metadata). Optionally set **Speaker** and **Name** / **Date** (same speaker aliases as Sermon tab).
2. Choose the AI model and host, then click **Generate post**. The app generates an NLC-style blog post (markdown) from the full transcript.
3. After success, **Post to Wix as draft** is enabled. Click it to create a draft in your Wix Blog (title + body converted to rich content). If the API returns a memberId error, add `WIX_BLOG_MEMBER_ID` to `.env` (e.g. your site owner or member GUID).
4. **Open in Wix** opens the Wix site dashboard in your browser. Go to **Blog > Posts** to find and edit the new draft.

**Blog environment:** Uses the same `WIX_BEARER_TOKEN` and `WIX_SITE_ID` as Sermon Metadata. The token must include the scope **SCOPE.DC-BLOG.MANAGE-BLOG**. Optional: `WIX_BLOG_MEMBER_ID` (see `.env.example`).
