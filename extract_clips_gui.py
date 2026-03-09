#!/usr/bin/env python3
"""
FastCap desktop app:
- Caption Video tab (faster-whisper)
- Extract Clips tab (JSON-based ffmpeg slicing)
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QDate, QEvent, QObject, QPoint, QThread, Signal, Qt, QUrl
from PySide6.QtGui import QColor, QClipboard, QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QFont, QMouseEvent, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QListWidget,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyleFactory,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from extract_clips import (
    extract_clip,
    format_seconds_to_timestamp,
    load_clips_from_data,
    parse_timestamp_to_seconds,
)
from llm_moment_ranker import (
    REQUIRED_ANALYSIS_FILES,
    call_llm_generate,
    format_ranked_clips_text,
    rank_sermon_moments,
)
from blog_post_prompt import build_blog_post_prompt
from facebook_post_prompt import build_facebook_post_prompt
from sermon_metadata import build_scribe_prompt, parse_sermon_metadata
from session_artifacts import (
    artifact_filename,
    artifact_path,
    asset_prefix as shared_asset_prefix,
    get_video_asset_dir as shared_get_video_asset_dir,
    load_session,
    now_iso,
    resolve_main_video_path,
    session_artifact_filename,
    sha256_file,
    sha256_text,
    speaker_to_slug as shared_speaker_to_slug,
    upsert_session,
    update_words_media_file,
)
from youtube_prompt import (
    build_youtube_prompt,
    build_youtube_prompt_with_chapters,
    format_youtube_chapters,
    get_chapter_segments,
    parse_srt_to_chapters,
    parse_youtube_response,
    srt_to_plain_text,
)
from wix_cms import create_sermon_item, get_wix_config
from wix_blog import create_blog_draft, get_wix_dashboard_url


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEVICE_CHOICES = ["auto", "cpu", "cuda"]
PREVIEW_PAD_SECONDS = 5.0
DISPLAY_SCORE_MAX = 115
_FAULT_LOG_HANDLE = None
SPEAKER_ALIASES_FILENAME = "speaker_aliases.json"
RECENT_PROJECTS_FILENAME = ".fastcap_recent_projects.json"
MAX_RECENT_PROJECTS = 8


def load_speaker_aliases() -> list[dict]:
    """Load speaker alias list from speaker_aliases.json (canonical name for Wix, promptName for AI).
    Uses the project directory (where this script lives) first, then cwd, so the project file wins."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    # Prefer script dir so we always use the project's speaker_aliases.json when it exists
    paths_to_try = [script_dir / SPEAKER_ALIASES_FILENAME, cwd / SPEAKER_ALIASES_FILENAME]
    for path in paths_to_try:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            out = []
            for item in data:
                if isinstance(item, dict) and "canonical" in item and "promptName" in item:
                    out.append({"canonical": str(item["canonical"]).strip(), "promptName": str(item["promptName"]).strip()})
            if out:
                logging.debug("Loaded %d speaker(s) from %s", len(out), path)
            return out
        except (OSError, json.JSONDecodeError) as e:
            logging.warning("Could not load %s: %s", path, e)
    return []


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def speaker_to_slug(speaker: str) -> str:
    """Turn a speaker display name into a safe filename segment (e.g. 'Pastor Chris' -> 'Pastor-Chris')."""
    return shared_speaker_to_slug(speaker)


def asset_prefix(date_str: str, speaker_slug: str) -> str:
    """Canonical base name for all assets: {Date}-{Speaker} (e.g. 2026-03-01-Pastor-Chris)."""
    return shared_asset_prefix(date_str, speaker_slug)


def _asset_file_path(asset_dir: Path, base_name: str) -> Path:
    """Path to a file in asset_dir; prefers prefixed name ({dir_name}-{base}) then legacy."""
    session = load_session(asset_dir)
    key_by_name = {
        "words.json": "words",
        "energy.json": "energy",
        "cadence.json": "cadence",
        "moments.json": "moments",
        "ranked_moments.json": "ranked_moments",
        "transcript.txt": "transcript",
        "subtitles.srt": "subtitles_srt",
        "subtitles.vtt": "subtitles_vtt",
        "audio.wav": "audio",
        "sermon_metadata.raw.txt": "sermon_metadata_raw",
        "sermon_metadata.json": "sermon_metadata",
        "blog.md": "blog",
        "youtube.json": "youtube",
        "facebook.txt": "facebook",
    }
    key = key_by_name.get(base_name)
    if key:
        return artifact_path(asset_dir, key, session=session)
    prefixed = asset_dir / f"{asset_dir.name}-{base_name}"
    if prefixed.is_file():
        return prefixed
    return asset_dir / base_name


def get_video_asset_dir(
    video_path: Path,
    date_str: str | None = None,
    speaker_slug: str | None = None,
) -> Path:
    """
    Resolve organized output folder for a video.
    - If date_str and speaker_slug are provided: parent / "{date_str}-{speaker_slug}" (S3-friendly).
    - Else if video is already inside a same-named folder, reuse that folder.
    - Otherwise create/use a sibling folder named after the video stem.
    """
    return shared_get_video_asset_dir(video_path, date_str=date_str, speaker_slug=speaker_slug)


def configure_runtime_logging(log_file: Path | None = None) -> Path:
    """Configure file logging for app events and crashes."""
    global _FAULT_LOG_HANDLE

    if log_file is None:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"fastcap-{stamp}.log"
    else:
        log_file = log_file.resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )

    _FAULT_LOG_HANDLE = open(log_file, "a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)
    logging.info("FastCap log started: %s", log_file)
    return log_file


def install_exception_hooks() -> None:
    """Capture uncaught exceptions to log."""

    def _handle_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return sys.__excepthook__(exc_type, exc_value, exc_tb)
        logging.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        return sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _threading_hook(args):
        logging.critical(
            "Unhandled thread exception in %s",
            args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _handle_exception
    threading.excepthook = _threading_hook


class VideoDropLineEdit(QLineEdit):
    video_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag a video file here, or click Browse...")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and is_video_path(Path(url.toLocalFile())):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                local = Path(url.toLocalFile())
                if is_video_path(local):
                    self.setText(str(local))
                    self.video_dropped.emit(str(local))
                    event.acceptProposedAction()
                    return
        event.ignore()


class ClickJumpSlider(QSlider):
    """QSlider that seeks directly to click position."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            minimum = int(self.minimum())
            maximum = int(self.maximum())
            if maximum > minimum:
                if self.orientation() == Qt.Horizontal:
                    span = max(1, self.width() - 1)
                    ratio = float(event.position().x()) / float(span)
                else:
                    span = max(1, self.height() - 1)
                    ratio = 1.0 - (float(event.position().y()) / float(span))
                ratio = max(0.0, min(1.0, ratio))
                value = int(round(minimum + ratio * (maximum - minimum)))
                self.setValue(value)
                self.sliderMoved.emit(value)
        super().mousePressEvent(event)


class WindowTitleBar(QFrame):
    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._drag_offset: QPoint | None = None
        self.setObjectName("windowTitleBar")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = QLabel(window.windowTitle())
        self.title_label.setObjectName("windowTitleLabel")
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.min_btn = QPushButton("−")
        self.min_btn.setObjectName("titleBarButton")
        self.min_btn.setToolTip("Minimize")
        self.min_btn.setFixedHeight(28)
        self.min_btn.clicked.connect(window.showMinimized)

        self.max_btn = QPushButton("□")
        self.max_btn.setObjectName("titleBarButton")
        self.max_btn.setToolTip("Maximize")
        self.max_btn.setFixedHeight(28)
        self.max_btn.clicked.connect(self._toggle_maximized)

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("titleBarCloseButton")
        self.close_btn.setToolTip("Close")
        self.close_btn.setFixedHeight(28)
        self.close_btn.clicked.connect(window.close)

        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.close_btn)
        self.sync_window_state()

    def set_window_title(self, title: str) -> None:
        self.title_label.setText(title)

    def sync_window_state(self) -> None:
        is_maximized = self._window.isMaximized()
        self.max_btn.setText("❐" if is_maximized else "□")
        self.max_btn.setToolTip("Restore" if is_maximized else "Maximize")

    def _toggle_maximized(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()
        self.sync_window_state()

    def _can_drag_from_event(self, event: QMouseEvent) -> bool:
        child = self.childAt(event.position().toPoint())
        return not isinstance(child, QPushButton)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._can_drag_from_event(event):
            self._toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._can_drag_from_event(event):
            if self._window._begin_system_move():
                event.accept()
                return
            if not self._window.isMaximized():
                self._drag_offset = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.LeftButton
            and not self._window.isMaximized()
        ):
            self._window.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class CaptionWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(bool, str)

    def __init__(
        self,
        video_path: Path,
        out_base: Path,
        model_name: str,
        device: str,
        language: str | None,
        write_vtt: bool,
        keep_audio: bool,
        max_chars_per_line: int,
        max_lines_per_cue: int,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.out_base = out_base
        self.model_name = model_name
        self.device = device
        self.language = language
        self.write_vtt = write_vtt
        self.keep_audio = keep_audio
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_cue = max_lines_per_cue

    def _expected_output_paths(self) -> list[Path]:
        out_dir = self.out_base.parent
        prefix = self.out_base.name
        paths = [
            out_dir / artifact_filename("subtitles_srt", prefix=prefix),
            out_dir / artifact_filename("transcript", prefix=prefix),
            out_dir / artifact_filename("words", prefix=prefix),
            out_dir / artifact_filename("energy", prefix=prefix),
            out_dir / artifact_filename("cadence", prefix=prefix),
            out_dir / artifact_filename("moments", prefix=prefix),
        ]
        if self.write_vtt:
            paths.append(out_dir / artifact_filename("subtitles_vtt", prefix=prefix))
        if self.keep_audio:
            paths.append(out_dir / artifact_filename("audio", prefix=prefix))
        return paths

    def _all_expected_outputs_exist(self) -> bool:
        return all(path.is_file() for path in self._expected_output_paths())

    def run(self) -> None:
        try:
            script_path = Path(__file__).resolve().parent / "caption_video.py"
            cmd = [
                sys.executable,
                "-u",
                str(script_path),
                str(self.video_path),
                "-o",
                str(self.out_base),
                "--model",
                self.model_name,
                "--device",
                self.device,
                "--max-chars-per-line",
                str(self.max_chars_per_line),
                "--max-lines-per-cue",
                str(self.max_lines_per_cue),
            ]
            if self.language:
                cmd.extend(["--language", self.language])
            if self.write_vtt:
                cmd.append("--vtt")
            if self.keep_audio:
                cmd.append("--keep-audio")

            self.log.emit(f"Loading model: {self.model_name} ({self.device})")
            self.log.emit("Starting captioning subprocess...")
            self.log.emit("Tip: model load can take 10-60s before transcribing starts.")
            self.progress.emit(0, 100, "Starting...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    self.log.emit(line)
                    match = re.search(r"Transcribing\.\.\.\s*(\d+)%", line)
                    if match:
                        pct = max(0, min(100, int(match.group(1))))
                        self.progress.emit(pct, 100, f"Transcribing... {pct}%")
                    elif "Loading faster-whisper model" in line:
                        self.progress.emit(2, 100, "Loading model...")
                    elif "Extracting audio with ffmpeg" in line:
                        self.progress.emit(8, 100, "Extracting audio...")
                    elif "Transcribing..." in line:
                        self.progress.emit(12, 100, "Transcribing...")

            return_code = process.wait()
            if return_code != 0:
                # Some native backends can crash at interpreter shutdown on Windows
                # after all files are already written. Treat that as success.
                if return_code == 3221226505 and self._all_expected_outputs_exist():
                    self.log.emit(
                        "WARNING: Caption subprocess exited with code 3221226505 after writing outputs; "
                        "treating as completed."
                    )
                    self.progress.emit(100, 100, "Completed (with warning)")
                    self.done.emit(
                        True,
                        "Captioning complete.\n"
                        f"Output base:\n{self.out_base}\n"
                        "Warning: subprocess reported Windows native crash code 3221226505 after output write.",
                    )
                    return
                raise RuntimeError(f"Caption subprocess failed with code {return_code}")

            self.progress.emit(100, 100, "Completed")
            self.done.emit(True, f"Captioning complete.\nOutput base:\n{self.out_base}")
        except Exception as exc:
            logging.error("Caption worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc))


class ClipWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(bool, str)

    def __init__(
        self,
        video_path: Path,
        out_dir: Path,
        prefix: str,
        pad_seconds: float,
        clips: list[dict],
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.out_dir = out_dir
        self.prefix = prefix
        self.pad_seconds = pad_seconds
        self.clips = clips

    def run(self) -> None:
        try:
            total = len(self.clips)
            suffix = self.video_path.suffix or ".mp4"
            self.log.emit(f"Starting extraction: {total} clips")
            self.log.emit(f"Video: {self.video_path}")
            self.log.emit(f"Output: {self.out_dir}")
            self.log.emit(f"Pad seconds: {self.pad_seconds}")

            for idx, clip in enumerate(self.clips, start=1):
                num = clip["clip_number"]
                raw_start = clip["start_time"]
                raw_end = clip["end_time"]

                start_sec = max(0.0, parse_timestamp_to_seconds(raw_start) - self.pad_seconds)
                end_sec = parse_timestamp_to_seconds(raw_end) + self.pad_seconds
                if end_sec <= start_sec:
                    raise ValueError(
                        f"Clip {num} has invalid range after padding: {raw_start} -> {raw_end}"
                    )

                start = format_seconds_to_timestamp(start_sec)
                duration = end_sec - start_sec
                out_name = f"{self.prefix}-clip_{int(num):03d}{suffix}"
                out_path = self.out_dir / out_name

                self.log.emit(
                    f"[{idx}/{total}] Clip {num}: {raw_start} -> {raw_end} "
                    f"(padded: {start} -> {format_seconds_to_timestamp(end_sec)}) -> {out_name}"
                )

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-ss",
                    start,
                    "-i",
                    str(self.video_path),
                    "-t",
                    f"{duration:.3f}",
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(out_path),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode != 0:
                    err = (result.stderr or result.stdout or "").strip()
                    if err:
                        self.log.emit("  " + err.splitlines()[-1])
                    raise RuntimeError(
                        f"ffmpeg failed on clip {num} ({out_name}) with code {result.returncode}"
                    )
                self.log.emit(f"  Wrote: {out_name}")

                self.progress.emit(idx, total, f"{idx}/{total} clips")

            self.done.emit(True, f"Extracted {total} clips to:\n{self.out_dir}")
        except Exception as exc:
            logging.error("Clip worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc))


class RankWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(bool, str, object)

    def __init__(
        self,
        asset_dir: Path,
        model_name: str,
        candidate_count: int,
        output_count: int,
    ) -> None:
        super().__init__()
        self.asset_dir = asset_dir
        self.model_name = model_name
        self.candidate_count = candidate_count
        self.output_count = output_count

    def run(self) -> None:
        try:
            self.log.emit("Starting ranking...")
            self.log.emit(f"Asset folder: {self.asset_dir}")
            self.log.emit(f"Model: {self.model_name}")
            self.log.emit(f"Candidates: {self.candidate_count} | Output clips: {self.output_count}")
            self.progress.emit(0, 100, "Starting...")

            def _on_progress(current: int, total: int, label: str) -> None:
                mapped = int(round((max(0, min(total, current)) / max(1, total)) * 100))
                self.progress.emit(mapped, 100, label)

            rank_result = rank_sermon_moments(
                asset_dir=self.asset_dir,
                model=self.model_name,
                candidate_limit=self.candidate_count,
                output_count=self.output_count,
                logger=self.log.emit,
                progress=_on_progress,
            )
            self.log.emit(
                "Pass fallback flags: "
                f"pass1={bool(rank_result.get('pass1_used_fallback'))}, "
                f"pass2={bool(rank_result.get('pass2_used_fallback'))}"
            )
            if rank_result.get("pass1_parse_error"):
                self.log.emit(f"Pass1 parse reason: {rank_result.get('pass1_parse_error')}")
            if rank_result.get("pass2_parse_error"):
                self.log.emit(f"Pass2 parse reason: {rank_result.get('pass2_parse_error')}")

            session = load_session(self.asset_dir)
            output_path = artifact_path(self.asset_dir, "ranked_moments", session=session)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(rank_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            upsert_session(
                self.asset_dir,
                {
                    "artifacts": {"ranked_moments": output_path.name},
                    "steps": {
                        "ranking": {
                            "status": "saved",
                            "generated_at": now_iso(),
                            "model": self.model_name,
                            "candidate_count": int(self.candidate_count),
                            "output_count": int(self.output_count),
                        }
                    },
                },
            )
            self.log.emit(f"Wrote: {output_path}")
            self.progress.emit(100, 100, "Completed")
            self.done.emit(True, f"Ranking complete.\nSaved:\n{output_path}", rank_result)
        except Exception as exc:
            logging.error("Rank worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc), None)


class SermonMetadataWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    chunk = Signal(str)  # streamed AI response text
    done = Signal(bool, str, str, object)  # success, message, raw_response, parsed_payload

    def __init__(
        self,
        transcript: str,
        model_name: str,
        host: str = "",
        preacher_name: str = "",
        date_preached: str = "",
    ) -> None:
        super().__init__()
        self.transcript = transcript
        self.model_name = model_name
        self.host = host.strip()
        self.preacher_name = (preacher_name or "").strip()
        self.date_preached = (date_preached or "").strip()

    def run(self) -> None:
        raw = ""
        try:
            self.log.emit("Building SCRIBE prompt...")
            self.progress.emit(0, 100, "Starting...")
            prompt = build_scribe_prompt(
                self.transcript,
                preacher_name=self.preacher_name,
                date_preached=self.date_preached,
            )
            self.log.emit("Calling AI model (read full transcript before responding)...")
            self.progress.emit(10, 100, "Calling model...")

            def _on_chunk(text: str) -> None:
                if text:
                    self.chunk.emit(text)
                self.progress.emit(50, 100, "Receiving...")

            raw = call_llm_generate(
                model=self.model_name,
                prompt=prompt,
                host=self.host or "http://127.0.0.1:11434",
                on_chunk=_on_chunk,
            )
            self.progress.emit(80, 100, "Parsing...")
            self.log.emit("Parsing and validating JSON...")
            payload, warnings = parse_sermon_metadata(raw)
            for w in warnings:
                self.log.emit(f"Warning: {w}")
            self.progress.emit(100, 100, "Completed")
            self.log.emit("Metadata parsed successfully.")
            self.done.emit(True, "Metadata generated and validated.", raw, payload)
        except Exception as exc:
            logging.error("Sermon metadata worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc), raw, None)


class BlogPostWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    chunk = Signal(str)
    done = Signal(bool, str, str)  # success, message, full_markdown

    def __init__(
        self,
        transcript: str,
        model_name: str,
        host: str = "",
        preacher_name: str = "",
        date_preached: str = "",
    ) -> None:
        super().__init__()
        self.transcript = transcript
        self.model_name = model_name
        self.host = host.strip()
        self.preacher_name = (preacher_name or "").strip()
        self.date_preached = (date_preached or "").strip()

    def run(self) -> None:
        full_markdown = ""
        try:
            self.log.emit("Building blog post prompt...")
            self.progress.emit(0, 100, "Starting...")
            prompt = build_blog_post_prompt(
                self.transcript,
                preacher_name=self.preacher_name,
                date_preached=self.date_preached,
            )
            self.log.emit("Calling AI model...")
            self.progress.emit(10, 100, "Calling model...")

            def _on_chunk(text: str) -> None:
                if text:
                    self.chunk.emit(text)
                    nonlocal full_markdown
                    full_markdown += text
                self.progress.emit(50, 100, "Receiving...")

            raw = call_llm_generate(
                model=self.model_name,
                prompt=prompt,
                host=self.host or "http://127.0.0.1:11434",
                on_chunk=_on_chunk,
            )
            full_markdown = raw or full_markdown
            self.progress.emit(100, 100, "Completed")
            self.log.emit("Blog post generated.")
            self.done.emit(True, "Blog post generated.", full_markdown)
        except Exception as exc:
            logging.error("Blog post worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc), full_markdown)


class YouTubeWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(bool, str, str, str)  # success, message, title, description

    def __init__(
        self,
        transcript: str,
        model_name: str,
        host: str = "",
        preacher_name: str = "",
        date_preached: str = "",
    ) -> None:
        super().__init__()
        self.transcript = transcript
        self.model_name = model_name
        self.host = host.strip()
        self.preacher_name = (preacher_name or "").strip()
        self.date_preached = (date_preached or "").strip()

    def run(self) -> None:
        try:
            self.log.emit("Building YouTube prompt...")
            self.progress.emit(0, 100, "Starting...")
            plain_transcript = srt_to_plain_text(self.transcript)
            chapters = parse_srt_to_chapters(self.transcript)
            segments = get_chapter_segments(chapters, interval_sec=300.0) if chapters else []

            if segments:
                prompt = build_youtube_prompt_with_chapters(
                    plain_transcript,
                    segments,
                    preacher_name=self.preacher_name,
                    date_preached=self.date_preached,
                )
            else:
                prompt = build_youtube_prompt(
                    plain_transcript,
                    preacher_name=self.preacher_name,
                    date_preached=self.date_preached,
                )

            self.log.emit("Calling AI model...")
            self.progress.emit(20, 100, "Calling model...")
            raw = call_llm_generate(
                model=self.model_name,
                prompt=prompt,
                host=self.host or "http://127.0.0.1:11434",
            )
            self.progress.emit(85, 100, "Parsing...")
            title, description, chapter_titles = parse_youtube_response(raw or "", num_segments=len(segments))

            if segments and chapter_titles and len(chapter_titles) == len(segments):
                chapters_with_titles = [(start_sec, chapter_titles[i]) for i, (start_sec, _) in enumerate(segments)]
                chapter_block = format_youtube_chapters(chapters_with_titles)
                description = f"{description.rstrip()}\n\n{chapter_block}".strip()
                self.log.emit(f"Added {len(chapters_with_titles)} chapters with summarized titles.")
            elif segments:
                chapters_with_titles = [
                    (start_sec, " ".join(combined.split()[:5])) for start_sec, combined in segments
                ]
                chapter_block = format_youtube_chapters(chapters_with_titles)
                description = f"{description.rstrip()}\n\n{chapter_block}".strip()
                self.log.emit(f"Added {len(chapters_with_titles)} chapters (fallback titles).")

            self.progress.emit(100, 100, "Completed")
            self.log.emit("YouTube title and description generated.")
            self.done.emit(True, "YouTube title and description generated.", title, description)
        except Exception as exc:
            logging.error("YouTube worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc), "", "")


class FacebookPostWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(bool, str, str)  # success, message, post_text

    def __init__(
        self,
        blog_post_markdown: str,
        model_name: str,
        host: str = "",
    ) -> None:
        super().__init__()
        self.blog_post_markdown = blog_post_markdown
        self.model_name = model_name
        self.host = host.strip()

    def run(self) -> None:
        post_text = ""
        try:
            self.log.emit("Building Facebook post prompt...")
            self.progress.emit(0, 100, "Starting...")
            prompt = build_facebook_post_prompt(self.blog_post_markdown)
            self.log.emit("Calling AI model...")
            self.progress.emit(20, 100, "Calling model...")
            raw = call_llm_generate(
                model=self.model_name,
                prompt=prompt,
                host=self.host or "http://127.0.0.1:11434",
            )
            post_text = (raw or "").strip()
            self.progress.emit(100, 100, "Completed")
            self.log.emit("Facebook post generated.")
            self.done.emit(True, "Facebook post generated.", post_text)
        except Exception as exc:
            logging.error("Facebook post worker failed: %s", traceback.format_exc())
            self.done.emit(False, str(exc), post_text)


class MainWindow(QMainWindow):
    _RESIZE_MARGIN = 8

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FastCap")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1120, 820)

        self.active_thread: QThread | None = None
        self.active_worker: QObject | None = None
        self._caption_last_video: Path | None = None
        self._caption_last_out_dir: Path | None = None
        self._caption_last_prefix = ""
        self._clips_last_video: Path | None = None
        self._clips_last_out_dir: Path | None = None
        self._clips_last_prefix = ""
        self._rank_last_dir: Path | None = None
        self._preview_rank_result: dict | None = None
        self._preview_video_path: Path | None = None
        self._preview_clip_start_ms = 0
        self._preview_clip_end_ms = 0
        self._preview_is_scrubbing = False
        self._feedback_session_keys: set[str] = set()
        self._feedback_persisted_keys: set[str] = set()
        self._feedback_total_count = 0
        self._sermon_last_parsed: dict | None = None
        self._sermon_last_transcript = ""
        self._sermon_last_raw = ""
        self._sermon_source_path: Path | None = None
        self._speaker_aliases: list[dict] = []
        self._sermon_speaker_preset: dict | None = None
        self._blog_last_markdown = ""
        self._blog_wix_post_id: str | None = None
        self._blog_speaker_preset: dict | None = None
        self._blog_source_path: Path | None = None
        self._youtube_source_path: Path | None = None
        self._caption_output_overridden = False
        self._clips_output_overridden = False
        self._preview_suppress_autoplay = False
        self._auto_run_enabled = False
        self._auto_run_next_step: str | None = None
        self._current_asset_dir: Path | None = None
        self._current_session: dict | None = None
        self._loaded_step_flags: set[str] = set()
        self._recent_projects: list[str] = []
        self._resize_cursor_active = False

        self.clips_tabs = QTabWidget()
        self.publish_tabs = QTabWidget()
        self.stage_stack = QStackedWidget()
        self.prepare_page = QWidget()
        self.clips_page = QWidget()
        self.publish_page = QWidget()
        self.caption_tab = QWidget()
        self.rank_tab = QWidget()
        self.preview_tab = QWidget()
        self.clips_tab = QWidget()
        self.sermon_tab = QWidget()
        self.blog_tab = QWidget()
        self.youtube_tab = QWidget()
        self.facebook_tab = QWidget()

        self._build_caption_tab()
        self._build_rank_tab()
        self._build_preview_tab()
        self._build_clips_tab()
        self._build_sermon_tab()
        self._build_blog_tab()
        self._build_youtube_tab()
        self._build_facebook_tab()
        self._process_buttons: list[QPushButton] = []
        self._register_process_button(self.caption_run_btn, "Generate Caption Assets", "caption")
        self._register_process_button(self.rank_run_btn, "Rank Moments", "ranking")
        self._register_process_button(self.extract_btn, "Run Extraction", "clips")
        self._register_process_button(self.sermon_run_btn, "Generate Metadata", "sermon_metadata")
        self._register_process_button(self.blog_run_btn, "Generate Post", "blog")
        self._register_process_button(self.youtube_run_btn, "Generate YouTube Copy", "youtube")
        self._register_process_button(self.facebook_run_btn, "Generate Facebook Copy", "facebook")
        self.clips_tabs.addTab(self.rank_tab, "Rank Moments")
        self.clips_tabs.addTab(self.preview_tab, "Preview Report")
        self.clips_tabs.addTab(self.clips_tab, "Extract Clips")
        self.clips_tabs.tabBar().hide()
        self.clips_tabs.currentChanged.connect(self._on_review_tab_changed)
        prepare_layout = QVBoxLayout(self.prepare_page)
        prepare_layout.setContentsMargins(0, 0, 0, 0)
        prepare_layout.setSpacing(12)
        prepare_header = QFrame()
        prepare_header.setObjectName("pageHeader")
        prepare_header.setProperty("accent", "prepare")
        prepare_header_layout = QVBoxLayout(prepare_header)
        prepare_header_layout.setContentsMargins(16, 14, 16, 14)
        prepare_header_layout.setSpacing(4)
        prepare_title = QLabel("Prepare")
        prepare_title.setObjectName("pageTitle")
        prepare_subtitle = QLabel("Load a source video, confirm the session details, and generate the caption artifacts.")
        prepare_subtitle.setObjectName("pageSubtitle")
        prepare_subtitle.setWordWrap(True)
        prepare_header_layout.addWidget(prepare_title)
        prepare_header_layout.addWidget(prepare_subtitle)
        prepare_layout.addWidget(prepare_header)
        prepare_layout.addWidget(self.caption_tab, stretch=1)
        clips_layout = QVBoxLayout(self.clips_page)
        clips_layout.setContentsMargins(0, 0, 0, 0)
        clips_layout.setSpacing(12)
        clips_header = QFrame()
        clips_header.setObjectName("pageHeader")
        clips_header.setProperty("accent", "review")
        clips_header_layout = QVBoxLayout(clips_header)
        clips_header_layout.setContentsMargins(16, 14, 16, 14)
        clips_header_layout.setSpacing(8)
        clips_title = QLabel("Review Clips")
        clips_title.setObjectName("pageTitle")
        clips_subtitle = QLabel("Rank moments, review the report cards, and send the best clips into extraction.")
        clips_subtitle.setObjectName("pageSubtitle")
        clips_subtitle.setWordWrap(True)
        clips_nav = QHBoxLayout()
        clips_nav.setSpacing(8)
        self.review_rank_btn = self._create_substage_nav_button("Rank Moments", self.clips_tabs, self.rank_tab, "review")
        self.review_preview_btn = self._create_substage_nav_button("Preview Report", self.clips_tabs, self.preview_tab, "review")
        self.review_extract_btn = self._create_substage_nav_button("Extract Clips", self.clips_tabs, self.clips_tab, "review")
        self._review_nav_buttons = {
            self.rank_tab: self.review_rank_btn,
            self.preview_tab: self.review_preview_btn,
            self.clips_tab: self.review_extract_btn,
        }
        clips_nav.addWidget(self.review_rank_btn)
        clips_nav.addWidget(self.review_preview_btn)
        clips_nav.addWidget(self.review_extract_btn)
        clips_nav.addStretch(1)
        clips_header_layout.addWidget(clips_title)
        clips_header_layout.addWidget(clips_subtitle)
        clips_header_layout.addLayout(clips_nav)
        clips_layout.addWidget(clips_header)
        clips_layout.addWidget(self.clips_tabs)
        publish_layout = QVBoxLayout(self.publish_page)
        publish_layout.setContentsMargins(0, 0, 0, 0)
        publish_layout.setSpacing(8)
        publish_header = QFrame()
        publish_header.setObjectName("pageHeader")
        publish_header.setProperty("accent", "publish")
        publish_header_layout = QVBoxLayout(publish_header)
        publish_header_layout.setContentsMargins(16, 14, 16, 14)
        publish_header_layout.setSpacing(6)
        publish_title = QLabel("Publish")
        publish_title.setObjectName("pageTitle")
        publish_subtitle = QLabel("Generate destination-ready assets for Wix, YouTube, and Facebook from the current session.")
        publish_subtitle.setObjectName("pageSubtitle")
        publish_subtitle.setWordWrap(True)
        publish_nav = QHBoxLayout()
        publish_nav.setSpacing(8)
        self.publish_sermon_btn = self._create_publish_nav_button("Wix Sermon", self.sermon_tab, "sermon")
        self.publish_blog_btn = self._create_publish_nav_button("Wix Blog", self.blog_tab, "blog")
        self.publish_youtube_btn = self._create_publish_nav_button("YouTube Copy", self.youtube_tab, "youtube")
        self.publish_facebook_btn = self._create_publish_nav_button("Facebook Copy", self.facebook_tab, "facebook")
        self._publish_nav_buttons = {
            self.sermon_tab: self.publish_sermon_btn,
            self.blog_tab: self.publish_blog_btn,
            self.youtube_tab: self.publish_youtube_btn,
            self.facebook_tab: self.publish_facebook_btn,
        }
        publish_nav.addWidget(self.publish_sermon_btn)
        publish_nav.addWidget(self.publish_blog_btn)
        publish_nav.addWidget(self.publish_youtube_btn)
        publish_nav.addWidget(self.publish_facebook_btn)
        publish_nav.addStretch(1)
        publish_header_layout.addWidget(publish_title)
        publish_header_layout.addWidget(publish_subtitle)
        publish_header_layout.addLayout(publish_nav)
        publish_layout.addWidget(publish_header)
        self.publish_stack = QStackedWidget()
        self.publish_stack.setObjectName("publishStack")
        self.publish_stack.addWidget(self.sermon_tab)
        self.publish_stack.addWidget(self.blog_tab)
        self.publish_stack.addWidget(self.youtube_tab)
        self.publish_stack.addWidget(self.facebook_tab)
        publish_layout.addWidget(self.publish_stack, stretch=1)

        self.stage_stack.addWidget(self.prepare_page)
        self.stage_stack.addWidget(self.clips_page)
        self.stage_stack.addWidget(self.publish_page)

        shell = QWidget()
        shell.setObjectName("windowShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(14, 14, 14, 14)
        shell_layout.setSpacing(0)

        self.window_surface = QFrame()
        self.window_surface.setObjectName("windowSurface")
        surface_layout = QVBoxLayout(self.window_surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.setSpacing(0)

        self.title_bar = WindowTitleBar(self)
        self.windowTitleChanged.connect(self.title_bar.set_window_title)
        surface_layout.addWidget(self.title_bar)

        root = QWidget()
        root.setObjectName("windowContent")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("sidebarFrame")
        self.sidebar_frame.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(12)

        stage_nav = QGroupBox("Workflow")
        stage_nav.setObjectName("workflowPanel")
        stage_nav_layout = QVBoxLayout(stage_nav)
        stage_nav_layout.setContentsMargins(12, 18, 12, 12)
        stage_nav_layout.setSpacing(8)
        self.prepare_nav_btn = self._create_stage_nav_button("Prepare", self.prepare_page, "prepare")
        self.clips_nav_btn = self._create_stage_nav_button("Review Clips", self.clips_page, "review")
        self.publish_nav_btn = self._create_stage_nav_button("Publish", self.publish_page, "publish")
        self._stage_nav_buttons = {
            self.prepare_page: self.prepare_nav_btn,
            self.clips_page: self.clips_nav_btn,
            self.publish_page: self.publish_nav_btn,
        }
        stage_nav_layout.addWidget(self.prepare_nav_btn)
        stage_nav_layout.addWidget(self.clips_nav_btn)
        stage_nav_layout.addWidget(self.publish_nav_btn)
        sidebar_layout.addWidget(stage_nav)

        self.session_panel = QGroupBox("Session")
        self.session_panel.setObjectName("sessionPanel")
        session_bar_layout = QGridLayout(self.session_panel)
        session_bar_layout.setContentsMargins(14, 18, 14, 14)
        session_bar_layout.setHorizontalSpacing(10)
        session_bar_layout.setVerticalSpacing(8)
        self.session_asset_label = QLabel("No asset folder loaded")
        self.session_asset_label.setWordWrap(True)
        self.session_video_label = QLabel("No source video loaded")
        self.session_video_label.setWordWrap(True)
        self.session_meta_label = QLabel("Speaker/date not set")
        self.session_meta_label.setWordWrap(True)
        self.session_status_label = QLabel("Prepare: not started\nReview: not started\nPublish: not started")
        self.session_status_label.setWordWrap(True)
        self.session_load_btn = QPushButton("Load Session")
        self.session_load_btn.clicked.connect(self.pick_session_folder)
        self.session_open_btn = QPushButton("Open Folder")
        self.session_open_btn.setEnabled(False)
        self.session_open_btn.clicked.connect(self.open_current_session_folder)
        self.recent_projects_list = QListWidget()
        self.recent_projects_list.setMaximumHeight(180)
        self.recent_projects_list.itemClicked.connect(self._open_recent_project)
        session_bar_layout.addWidget(QLabel("Asset folder"), 0, 0, 1, 4)
        session_bar_layout.addWidget(self.session_asset_label, 1, 0, 1, 4)
        session_bar_layout.addWidget(QLabel("Source video"), 2, 0, 1, 4)
        session_bar_layout.addWidget(self.session_video_label, 3, 0, 1, 4)
        session_bar_layout.addWidget(QLabel("Speaker / date"), 4, 0, 1, 4)
        session_bar_layout.addWidget(self.session_meta_label, 5, 0, 1, 4)
        session_bar_layout.addWidget(QLabel("Status"), 6, 0, 1, 4)
        session_bar_layout.addWidget(self.session_status_label, 7, 0, 1, 4)
        session_bar_layout.addWidget(self.session_load_btn, 8, 0, 1, 2)
        session_bar_layout.addWidget(self.session_open_btn, 8, 2, 1, 2)
        session_bar_layout.addWidget(QLabel("Recent projects"), 9, 0, 1, 4)
        session_bar_layout.addWidget(self.recent_projects_list, 10, 0, 1, 4)
        session_bar_layout.setColumnStretch(0, 1)
        session_bar_layout.setColumnStretch(1, 1)
        session_bar_layout.setColumnStretch(2, 1)
        session_bar_layout.setColumnStretch(3, 1)
        sidebar_layout.addWidget(self.session_panel)
        sidebar_layout.addStretch(1)

        outer.addWidget(self.sidebar_frame)
        outer.addWidget(self.stage_stack, stretch=1)
        surface_layout.addWidget(root, stretch=1)
        shell_layout.addWidget(self.window_surface)
        self.setCentralWidget(shell)

        self._apply_style()
        self.setMouseTracking(True)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._load_recent_projects()
        self.show_stage(self.prepare_page)
        self._on_review_tab_changed(self.clips_tabs.currentIndex())
        self._set_active_publish_section(self.sermon_tab)
        self._refresh_session_summary()

    def _build_caption_tab(self) -> None:
        self.caption_video_edit = VideoDropLineEdit()
        self.caption_video_edit.video_dropped.connect(self._on_caption_video_dropped)
        self.caption_output_edit = QLineEdit()
        self.caption_output_edit.setPlaceholderText("Session folder (auto-derived from date and speaker unless overridden)")

        self.caption_date_edit = QDateEdit()
        self.caption_date_edit.setCalendarPopup(True)
        self.caption_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.caption_date_edit.setDate(QDate.currentDate())
        self.caption_date_edit.dateChanged.connect(self._refresh_caption_output_dir)
        self.caption_speaker_combo = QComboBox()
        for alias in load_speaker_aliases():
            self.caption_speaker_combo.addItem(alias["promptName"])
        self.caption_speaker_combo.addItem("— Other —")
        self.caption_speaker_combo.setCurrentIndex(0)
        self.caption_speaker_combo.currentIndexChanged.connect(self._refresh_caption_output_dir)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.setCurrentText("base")

        self.device_combo = QComboBox()
        self.device_combo.addItems(DEVICE_CHOICES)
        self.device_combo.setCurrentText("auto")

        self.language_edit = QLineEdit()
        self.language_edit.setPlaceholderText("Optional language code, e.g. en")
        self.vtt_check = QCheckBox("Also write .vtt")
        self.keep_audio_check = QCheckBox("Keep extracted .wav audio")

        self.max_chars_spin = QSpinBox()
        self.max_chars_spin.setRange(0, 200)
        self.max_chars_spin.setValue(20)
        self.max_lines_spin = QSpinBox()
        self.max_lines_spin.setRange(1, 10)
        self.max_lines_spin.setValue(2)

        self.caption_log = QPlainTextEdit()
        self.caption_log.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.caption_log.setFont(mono)

        self.caption_progress = QProgressBar()
        self.caption_progress.setMinimum(0)
        self.caption_progress.setValue(0)

        self.caption_run_btn = QPushButton("Generate Caption Assets")
        self.caption_run_btn.setObjectName("primaryButton")
        self.caption_run_btn.setProperty("accent", "prepare")
        self.caption_run_btn.clicked.connect(self.start_captioning)
        self.caption_auto_btn = QPushButton("Auto Run Core Workflow")
        self.caption_auto_btn.clicked.connect(self.start_auto_run)
        self.caption_advanced_btn = QPushButton("Show Advanced Options")
        self.caption_advanced_btn.setCheckable(True)
        self.caption_advanced_btn.clicked.connect(self._toggle_caption_advanced)
        caption_clear_btn = QPushButton("Clear Log")
        caption_clear_btn.clicked.connect(self.caption_log.clear)

        layout = QVBoxLayout(self.caption_tab)
        layout.setSpacing(12)

        inputs = QGroupBox("Session Setup")
        grid = QGridLayout(inputs)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Video file:"), 0, 0)
        grid.addWidget(self.caption_video_edit, 0, 1)
        browse_video = QPushButton("Browse...")
        browse_video.clicked.connect(self.pick_caption_video)
        grid.addWidget(browse_video, 0, 2)

        grid.addWidget(QLabel("Output folder:"), 1, 0)
        grid.addWidget(self.caption_output_edit, 1, 1)
        browse_output = QPushButton("Browse...")
        browse_output.clicked.connect(self.pick_caption_output)
        grid.addWidget(browse_output, 1, 2)

        grid.addWidget(QLabel("Date:"), 2, 0)
        grid.addWidget(self.caption_date_edit, 2, 1)
        grid.addWidget(QLabel("Speaker:"), 3, 0)
        grid.addWidget(self.caption_speaker_combo, 3, 1, 1, 3)

        grid.setColumnStretch(1, 1)

        prep_note = QLabel(
            "Creates subtitles, transcript, words, energy, cadence, and moments for the current session."
        )
        prep_note.setObjectName("sectionSubtitle")
        prep_note.setWordWrap(True)

        advanced_group = QGroupBox("Advanced Caption Options")
        advanced_group.setVisible(False)
        self.caption_advanced_group = advanced_group
        advanced_grid = QGridLayout(advanced_group)
        advanced_grid.setContentsMargins(14, 18, 14, 14)
        advanced_grid.setHorizontalSpacing(10)
        advanced_grid.setVerticalSpacing(10)
        advanced_grid.addWidget(QLabel("Model:"), 0, 0)
        advanced_grid.addWidget(self.model_combo, 0, 1)
        advanced_grid.addWidget(QLabel("Device:"), 0, 2)
        advanced_grid.addWidget(self.device_combo, 0, 3)
        advanced_grid.addWidget(QLabel("Language:"), 1, 0)
        advanced_grid.addWidget(self.language_edit, 1, 1)
        advanced_grid.addWidget(self.vtt_check, 1, 2)
        advanced_grid.addWidget(self.keep_audio_check, 1, 3)
        advanced_grid.addWidget(QLabel("Max chars/line:"), 2, 0)
        advanced_grid.addWidget(self.max_chars_spin, 2, 1)
        advanced_grid.addWidget(QLabel("Max lines/cue:"), 2, 2)
        advanced_grid.addWidget(self.max_lines_spin, 2, 3)
        advanced_grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.caption_run_btn)
        controls.addWidget(self.caption_auto_btn)
        controls.addWidget(self.caption_advanced_btn)
        controls.addWidget(caption_clear_btn)
        controls.addStretch(1)

        logs = QGroupBox("Activity")
        log_layout = QVBoxLayout(logs)
        log_layout.setContentsMargins(10, 14, 10, 10)
        log_layout.addWidget(self.caption_log)

        layout.addWidget(inputs)
        layout.addWidget(prep_note)
        layout.addWidget(advanced_group)
        layout.addLayout(controls)
        layout.addWidget(self.caption_progress)
        layout.addWidget(logs, stretch=1)

    def _build_clips_tab(self) -> None:
        self.clips_video_edit = VideoDropLineEdit()
        self.clips_video_edit.video_dropped.connect(self._on_clips_video_dropped)
        self.clips_output_edit = QLineEdit()
        self.clips_output_edit.setPlaceholderText("Clip output folder (auto-derived from date and speaker unless overridden)")
        self.clips_date_edit = QDateEdit()
        self.clips_date_edit.setCalendarPopup(True)
        self.clips_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.clips_date_edit.setDate(QDate.currentDate())
        self.clips_date_edit.dateChanged.connect(self._refresh_clips_output_dir)
        self.clips_speaker_combo = QComboBox()
        for alias in load_speaker_aliases():
            self.clips_speaker_combo.addItem(alias["promptName"])
        self.clips_speaker_combo.addItem("— Other —")
        self.clips_speaker_combo.setCurrentIndex(0)
        self.clips_speaker_combo.currentIndexChanged.connect(self._refresh_clips_output_dir)
        self.clips_source_combo = QComboBox()
        self.clips_source_combo.addItems(["Use Ranked Report", "Use Kept Clips", "Paste Custom JSON"])
        self.clips_source_combo.currentIndexChanged.connect(self._on_clips_source_mode_changed)
        self.clips_pad_spin = QDoubleSpinBox()
        self.clips_pad_spin.setRange(0.0, 9999.0)
        self.clips_pad_spin.setDecimals(3)
        self.clips_pad_spin.setSingleStep(0.5)
        self.clips_pad_spin.setValue(5.0)

        self.clips_json_edit = QPlainTextEdit()
        self.clips_json_edit.setPlainText(
            '{\n  "clips": [\n'
            '    { "clip_number": 1, "start_time": "00:00:10.000", "end_time": "00:00:20.000" }\n'
            "  ]\n}\n"
        )
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.clips_json_edit.setFont(mono)

        self.clips_log = QPlainTextEdit()
        self.clips_log.setReadOnly(True)
        self.clips_log.setFont(mono)

        self.clips_progress = QProgressBar()
        self.clips_progress.setMinimum(0)
        self.clips_progress.setValue(0)

        self.extract_btn = QPushButton("Run Extraction")
        self.extract_btn.setObjectName("primaryButton")
        self.extract_btn.setProperty("accent", "review")
        self.extract_btn.clicked.connect(self.start_extraction)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clips_log.clear)

        layout = QVBoxLayout(self.clips_tab)
        layout.setSpacing(12)

        inputs = QGroupBox("Extraction Setup")
        grid = QGridLayout(inputs)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Video file:"), 0, 0)
        grid.addWidget(self.clips_video_edit, 0, 1)
        browse_video = QPushButton("Browse...")
        browse_video.clicked.connect(self.pick_clips_video)
        grid.addWidget(browse_video, 0, 2)

        grid.addWidget(QLabel("Output folder:"), 1, 0)
        grid.addWidget(self.clips_output_edit, 1, 1)
        browse_output = QPushButton("Browse...")
        browse_output.clicked.connect(self.pick_clips_output)
        grid.addWidget(browse_output, 1, 2)

        grid.addWidget(QLabel("Date:"), 2, 0)
        grid.addWidget(self.clips_date_edit, 2, 1)
        grid.addWidget(QLabel("Speaker:"), 3, 0)
        grid.addWidget(self.clips_speaker_combo, 3, 1, 1, 3)

        grid.addWidget(QLabel("Clip source:"), 4, 0)
        grid.addWidget(self.clips_source_combo, 4, 1)
        grid.addWidget(QLabel("Pad seconds:"), 4, 2)
        grid.addWidget(self.clips_pad_spin, 4, 3)
        grid.setColumnStretch(1, 1)

        extract_note = QLabel(
            "Choose whether extraction should follow the ranked report, only your kept clips, or a pasted custom JSON payload."
        )
        extract_note.setObjectName("sectionSubtitle")
        extract_note.setWordWrap(True)

        json_group = QGroupBox("Clips To Extract")
        json_layout = QVBoxLayout(json_group)
        json_layout.setContentsMargins(10, 14, 10, 10)
        json_layout.addWidget(self.clips_json_edit)

        controls = QHBoxLayout()
        controls.addWidget(self.extract_btn)
        controls.addWidget(clear_btn)
        controls.addStretch(1)

        logs = QGroupBox("Activity")
        log_layout = QVBoxLayout(logs)
        log_layout.setContentsMargins(10, 14, 10, 10)
        log_layout.addWidget(self.clips_log)

        layout.addWidget(inputs)
        layout.addWidget(extract_note)
        layout.addWidget(json_group, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self.clips_progress)
        layout.addWidget(logs, stretch=1)
        self._on_clips_source_mode_changed(self.clips_source_combo.currentIndex())

    def _build_sermon_tab(self) -> None:
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)

        self.sermon_transcript_edit = QPlainTextEdit()
        self.sermon_transcript_edit.setPlaceholderText("Paste transcript (SRT/VTT text or plain transcript)...")
        self.sermon_transcript_edit.setFont(mono)
        self._set_text_panel_height(self.sermon_transcript_edit, 120, 180)
        self.sermon_transcript_edit.hide()

        self.sermon_model_combo = QComboBox()
        self.sermon_model_combo.addItems([
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-sonnet-4-6",
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-opus-4-6-v1",
        ])
        self.sermon_host_edit = QLineEdit()
        self.sermon_host_edit.setPlaceholderText("Optional: host (e.g. http://127.0.0.1:11434 or bedrock region)")
        self.sermon_video_url_edit = QLineEdit()
        self.sermon_video_url_edit.setPlaceholderText("Optional: video URL for CMS")
        self._speaker_aliases = load_speaker_aliases()
        self.sermon_speaker_combo = QComboBox()
        for alias in self._speaker_aliases:
            self.sermon_speaker_combo.addItem(alias["promptName"])
        self.sermon_speaker_combo.addItem("— Other —")
        self.sermon_speaker_combo.setCurrentIndex(len(self._speaker_aliases))
        self.sermon_speaker_combo.currentIndexChanged.connect(self._on_sermon_speaker_changed)
        self.sermon_name_edit = QLineEdit()
        self.sermon_name_edit.setPlaceholderText("Plain name saved to Wix (e.g. Misti Sanders)")
        self.sermon_date_edit = QDateEdit()
        self.sermon_date_edit.setCalendarPopup(True)
        self.sermon_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.sermon_date_edit.setDate(QDate.currentDate())

        self.sermon_raw_edit = QPlainTextEdit()
        self.sermon_raw_edit.setReadOnly(True)
        self.sermon_raw_edit.setFont(mono)
        self._set_text_panel_height(self.sermon_raw_edit, 120, 180)
        self.sermon_parsed_edit = QPlainTextEdit()
        self.sermon_parsed_edit.setReadOnly(True)
        self.sermon_parsed_edit.setFont(mono)
        self._set_text_panel_height(self.sermon_parsed_edit, 140, 220)
        self.sermon_log = QPlainTextEdit()
        self.sermon_log.setReadOnly(True)
        self.sermon_log.setFont(mono)
        self._set_text_panel_height(self.sermon_log, 100, 160)

        self.sermon_progress = QProgressBar()
        self.sermon_progress.setMinimum(0)
        self.sermon_progress.setValue(0)

        self.sermon_run_btn = QPushButton("Generate Metadata")
        self.sermon_run_btn.setObjectName("primaryButton")
        self.sermon_run_btn.setProperty("accent", "sermon")
        self.sermon_run_btn.clicked.connect(self.start_sermon_metadata)
        self.sermon_send_wix_btn = QPushButton("Create Wix CMS Item")
        self.sermon_send_wix_btn.setEnabled(False)
        self.sermon_send_wix_btn.clicked.connect(self.send_sermon_to_wix)
        sermon_clear_btn = QPushButton("Clear")
        sermon_clear_btn.clicked.connect(self._clear_sermon_tab)

        layout = QVBoxLayout(self.sermon_tab)
        layout.setSpacing(12)

        source_group, self.sermon_source_status_label = self._build_publish_source_group(
            "This destination automatically uses the saved transcript from Prepare.",
            "sermon",
        )

        settings = QGroupBox("Settings")
        grid = QGridLayout(settings)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(QLabel("Model:"), 0, 0)
        grid.addWidget(self.sermon_model_combo, 0, 1)
        grid.addWidget(QLabel("Host / region:"), 0, 2)
        grid.addWidget(self.sermon_host_edit, 0, 3)
        grid.addWidget(QLabel("Video URL:"), 1, 0)
        grid.addWidget(self.sermon_video_url_edit, 1, 1, 1, 3)
        grid.addWidget(QLabel("Speaker:"), 2, 0)
        grid.addWidget(self.sermon_speaker_combo, 2, 1)
        grid.addWidget(QLabel("Name (saved to Wix):"), 2, 2)
        grid.addWidget(self.sermon_name_edit, 2, 3)
        grid.addWidget(QLabel("Date:"), 3, 0)
        grid.addWidget(self.sermon_date_edit, 3, 1)
        grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.sermon_run_btn)
        controls.addWidget(self.sermon_send_wix_btn)
        controls.addWidget(sermon_clear_btn)
        controls.addStretch(1)

        raw_group = QGroupBox("Raw model response")
        raw_layout = QVBoxLayout(raw_group)
        raw_layout.setContentsMargins(10, 14, 10, 10)
        raw_layout.addWidget(self.sermon_raw_edit)

        parsed_group = QGroupBox("Parsed metadata (JSON)")
        parsed_layout = QVBoxLayout(parsed_group)
        parsed_layout.setContentsMargins(10, 14, 10, 10)
        parsed_layout.addWidget(self.sermon_parsed_edit)

        logs_group = QGroupBox("Progress / Log")
        log_layout = QVBoxLayout(logs_group)
        log_layout.setContentsMargins(10, 14, 10, 10)
        log_layout.addWidget(self.sermon_log)

        layout.addWidget(source_group)
        self._add_collapsible_section(layout, "Settings", settings, expanded=False)
        layout.addLayout(controls)
        layout.addWidget(self.sermon_progress)
        self._add_collapsible_section(layout, "Parsed Metadata", parsed_group, expanded=False)
        self._add_collapsible_section(layout, "Raw Model Response", raw_group, expanded=False)
        self._add_collapsible_section(layout, "Activity Log", logs_group, expanded=False)

    def _build_blog_tab(self) -> None:
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)

        self.blog_transcript_edit = QPlainTextEdit()
        self.blog_transcript_edit.setPlaceholderText("Paste transcript (SRT/VTT text or plain transcript)...")
        self.blog_transcript_edit.setFont(mono)
        self._set_text_panel_height(self.blog_transcript_edit, 120, 180)
        self.blog_transcript_edit.hide()

        self.blog_model_combo = QComboBox()
        self.blog_model_combo.addItems([
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-sonnet-4-6",
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-opus-4-6-v1",
        ])
        self.blog_host_edit = QLineEdit()
        self.blog_host_edit.setPlaceholderText("Optional: host (e.g. http://127.0.0.1:11434)")
        self.blog_speaker_combo = QComboBox()
        for alias in self._speaker_aliases:
            self.blog_speaker_combo.addItem(alias["promptName"])
        self.blog_speaker_combo.addItem("— Other —")
        self.blog_speaker_combo.setCurrentIndex(len(self._speaker_aliases))
        self.blog_speaker_combo.currentIndexChanged.connect(self._on_blog_speaker_changed)
        self.blog_name_edit = QLineEdit()
        self.blog_name_edit.setPlaceholderText("Plain name for prompt (e.g. Misti Sanders)")
        self.blog_date_edit = QDateEdit()
        self.blog_date_edit.setCalendarPopup(True)
        self.blog_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.blog_date_edit.setDate(QDate.currentDate())

        self.blog_output_edit = QPlainTextEdit()
        self.blog_output_edit.setReadOnly(True)
        self.blog_output_edit.setFont(mono)
        self._set_text_panel_height(self.blog_output_edit, 220, None)
        self.blog_log = QPlainTextEdit()
        self.blog_log.setReadOnly(True)
        self.blog_log.setFont(mono)
        self._set_text_panel_height(self.blog_log, 100, 160)

        self.blog_progress = QProgressBar()
        self.blog_progress.setMinimum(0)
        self.blog_progress.setValue(0)

        self.blog_run_btn = QPushButton("Generate Post")
        self.blog_run_btn.setObjectName("primaryButton")
        self.blog_run_btn.setProperty("accent", "blog")
        self.blog_run_btn.clicked.connect(self.start_blog_post)
        self.blog_post_wix_btn = QPushButton("Create Wix Blog Draft")
        self.blog_post_wix_btn.setEnabled(False)
        self.blog_post_wix_btn.clicked.connect(self.post_blog_to_wix)
        self.blog_open_wix_btn = QPushButton("Open Wix Blog Dashboard")
        self.blog_open_wix_btn.setEnabled(False)
        self.blog_open_wix_btn.clicked.connect(self.open_blog_in_wix)
        blog_clear_btn = QPushButton("Clear")
        blog_clear_btn.clicked.connect(self._clear_blog_tab)

        layout = QVBoxLayout(self.blog_tab)
        layout.setSpacing(12)

        source_group, self.blog_source_status_label = self._build_publish_source_group(
            "This destination automatically uses the saved transcript from Prepare.",
            "blog",
        )

        settings = QGroupBox("Settings")
        grid = QGridLayout(settings)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(QLabel("Model:"), 0, 0)
        grid.addWidget(self.blog_model_combo, 0, 1)
        grid.addWidget(QLabel("Host:"), 0, 2)
        grid.addWidget(self.blog_host_edit, 0, 3)
        grid.addWidget(QLabel("Speaker:"), 1, 0)
        grid.addWidget(self.blog_speaker_combo, 1, 1)
        grid.addWidget(QLabel("Name:"), 1, 2)
        grid.addWidget(self.blog_name_edit, 1, 3)
        grid.addWidget(QLabel("Date:"), 2, 0)
        grid.addWidget(self.blog_date_edit, 2, 1)
        grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.blog_run_btn)
        controls.addWidget(self.blog_post_wix_btn)
        controls.addWidget(self.blog_open_wix_btn)
        controls.addWidget(blog_clear_btn)
        controls.addStretch(1)

        output_group = QGroupBox("Generated post (markdown)")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(10, 14, 10, 10)
        output_layout.addWidget(self.blog_output_edit, stretch=1)

        logs_group = QGroupBox("Progress / Log")
        log_layout = QVBoxLayout(logs_group)
        log_layout.setContentsMargins(10, 14, 10, 10)
        log_layout.addWidget(self.blog_log)

        layout.addWidget(source_group)
        self._add_collapsible_section(layout, "Settings", settings, expanded=False)
        layout.addLayout(controls)
        layout.addWidget(self.blog_progress)
        layout.addWidget(output_group, stretch=1)
        self._add_collapsible_section(layout, "Activity Log", logs_group, expanded=False)

    def _build_youtube_tab(self) -> None:
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)

        self.youtube_transcript_edit = QPlainTextEdit()
        self.youtube_transcript_edit.setPlaceholderText(
            "Paste SRT or transcript. For accurate chapters, load an SRT file (e.g. from your asset folder)."
        )
        self.youtube_transcript_edit.setFont(mono)
        self._set_text_panel_height(self.youtube_transcript_edit, 120, 180)
        self.youtube_transcript_edit.hide()

        self.youtube_model_combo = QComboBox()
        self.youtube_model_combo.addItems([
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-sonnet-4-6",
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-opus-4-6-v1",
        ])
        self.youtube_host_edit = QLineEdit()
        self.youtube_host_edit.setPlaceholderText("Optional: host (e.g. http://127.0.0.1:11434)")
        self.youtube_speaker_combo = QComboBox()
        for alias in load_speaker_aliases():
            self.youtube_speaker_combo.addItem(alias["promptName"])
        self.youtube_speaker_combo.addItem("— Other —")
        self.youtube_speaker_combo.setCurrentIndex(0)
        self.youtube_name_edit = QLineEdit()
        self.youtube_name_edit.setPlaceholderText("Speaker name (if Other)")
        self.youtube_date_edit = QDateEdit()
        self.youtube_date_edit.setCalendarPopup(True)
        self.youtube_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.youtube_date_edit.setDate(QDate.currentDate())

        self.youtube_title_edit = QLineEdit()
        self.youtube_title_edit.setReadOnly(True)
        self.youtube_title_edit.setPlaceholderText("Generated title will appear here")
        self.youtube_description_edit = QPlainTextEdit()
        self.youtube_description_edit.setReadOnly(True)
        self.youtube_description_edit.setPlaceholderText("Generated description will appear here")
        self.youtube_description_edit.setFont(mono)
        self._set_text_panel_height(self.youtube_description_edit, 220, None)
        self.youtube_log = QPlainTextEdit()
        self.youtube_log.setReadOnly(True)
        self.youtube_log.setFont(mono)
        self._set_text_panel_height(self.youtube_log, 100, 160)
        self.youtube_progress = QProgressBar()
        self.youtube_progress.setMinimum(0)
        self.youtube_progress.setValue(0)

        self.youtube_run_btn = QPushButton("Generate YouTube Copy")
        self.youtube_run_btn.setObjectName("primaryButton")
        self.youtube_run_btn.setProperty("accent", "youtube")
        self.youtube_run_btn.clicked.connect(self.start_youtube)
        youtube_copy_title_btn = QPushButton("Copy title")
        youtube_copy_title_btn.clicked.connect(self._youtube_copy_title)
        youtube_copy_desc_btn = QPushButton("Copy description")
        youtube_copy_desc_btn.clicked.connect(self._youtube_copy_description)
        youtube_copy_both_btn = QPushButton("Copy both")
        youtube_copy_both_btn.clicked.connect(self._youtube_copy_both)
        youtube_open_studio_btn = QPushButton("Open YouTube Studio")
        youtube_open_studio_btn.clicked.connect(self._youtube_open_studio)
        youtube_clear_btn = QPushButton("Clear")
        youtube_clear_btn.clicked.connect(self._clear_youtube_tab)

        layout = QVBoxLayout(self.youtube_tab)
        layout.setSpacing(12)
        source_group, self.youtube_source_status_label = self._build_publish_source_group(
            "This destination automatically uses the saved transcript from Prepare.",
            "youtube",
        )

        settings = QGroupBox("Settings")
        grid = QGridLayout(settings)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.addWidget(QLabel("Model:"), 0, 0)
        grid.addWidget(self.youtube_model_combo, 0, 1)
        grid.addWidget(QLabel("Host:"), 0, 2)
        grid.addWidget(self.youtube_host_edit, 0, 3)
        grid.addWidget(QLabel("Speaker:"), 1, 0)
        grid.addWidget(self.youtube_speaker_combo, 1, 1)
        grid.addWidget(QLabel("Name:"), 1, 2)
        grid.addWidget(self.youtube_name_edit, 1, 3)
        grid.addWidget(QLabel("Date:"), 2, 0)
        grid.addWidget(self.youtube_date_edit, 2, 1)
        grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.youtube_run_btn)
        controls.addWidget(youtube_copy_title_btn)
        controls.addWidget(youtube_copy_desc_btn)
        controls.addWidget(youtube_copy_both_btn)
        controls.addWidget(youtube_open_studio_btn)
        controls.addWidget(youtube_clear_btn)
        controls.addStretch(1)

        output_group = QGroupBox("Generated title & description")
        out_layout = QVBoxLayout(output_group)
        out_layout.addWidget(QLabel("Title:"))
        out_layout.addWidget(self.youtube_title_edit)
        out_layout.addWidget(QLabel("Description:"))
        out_layout.addWidget(self.youtube_description_edit, stretch=1)
        logs_group = QGroupBox("Progress / Log")
        logs_group.setLayout(QVBoxLayout())
        logs_group.layout().addWidget(self.youtube_log)

        layout.addWidget(source_group)
        self._add_collapsible_section(layout, "Settings", settings, expanded=False)
        layout.addLayout(controls)
        layout.addWidget(self.youtube_progress)
        layout.addWidget(output_group, stretch=1)
        self._add_collapsible_section(layout, "Activity Log", logs_group, expanded=False)

    def _build_facebook_tab(self) -> None:
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)

        self.facebook_source_edit = QPlainTextEdit()
        self.facebook_source_edit.setPlaceholderText("Paste blog post (markdown) or click Load from Blog...")
        self.facebook_source_edit.setFont(mono)
        self._set_text_panel_height(self.facebook_source_edit, 120, 180)
        self.facebook_source_edit.hide()

        self.facebook_model_combo = QComboBox()
        self.facebook_model_combo.addItems([
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-sonnet-4-6",
            "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-opus-4-6-v1",
        ])
        self.facebook_host_edit = QLineEdit()
        self.facebook_host_edit.setPlaceholderText("Optional: host (e.g. http://127.0.0.1:11434)")

        self.facebook_output_edit = QPlainTextEdit()
        self.facebook_output_edit.setReadOnly(True)
        self.facebook_output_edit.setPlaceholderText("Generated Facebook post will appear here")
        self.facebook_output_edit.setFont(mono)
        self._set_text_panel_height(self.facebook_output_edit, 220, None)
        self.facebook_log = QPlainTextEdit()
        self.facebook_log.setReadOnly(True)
        self.facebook_log.setFont(mono)
        self._set_text_panel_height(self.facebook_log, 100, 160)
        self.facebook_progress = QProgressBar()
        self.facebook_progress.setMinimum(0)
        self.facebook_progress.setValue(0)

        self.facebook_run_btn = QPushButton("Generate Facebook Copy")
        self.facebook_run_btn.setObjectName("primaryButton")
        self.facebook_run_btn.setProperty("accent", "facebook")
        self.facebook_run_btn.clicked.connect(self.start_facebook_post)
        facebook_copy_btn = QPushButton("Copy to clipboard")
        facebook_copy_btn.clicked.connect(self._facebook_copy)
        facebook_clear_btn = QPushButton("Clear")
        facebook_clear_btn.clicked.connect(self._clear_facebook_tab)

        layout = QVBoxLayout(self.facebook_tab)
        layout.setSpacing(12)
        source_group, self.facebook_source_status_label = self._build_publish_source_group(
            "This destination automatically uses the saved blog draft from Wix Blog.",
            "facebook",
        )

        settings = QGroupBox("Settings")
        grid = QGridLayout(settings)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.addWidget(QLabel("Model:"), 0, 0)
        grid.addWidget(self.facebook_model_combo, 0, 1)
        grid.addWidget(QLabel("Host:"), 0, 2)
        grid.addWidget(self.facebook_host_edit, 0, 3)
        grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.facebook_run_btn)
        controls.addWidget(facebook_copy_btn)
        controls.addWidget(facebook_clear_btn)
        controls.addStretch(1)

        output_group = QGroupBox("Generated Facebook post")
        out_layout = QVBoxLayout(output_group)
        out_layout.addWidget(self.facebook_output_edit, stretch=1)
        logs_group = QGroupBox("Progress / Log")
        logs_group.setLayout(QVBoxLayout())
        logs_group.layout().addWidget(self.facebook_log)

        layout.addWidget(source_group)
        self._add_collapsible_section(layout, "Settings", settings, expanded=False)
        layout.addLayout(controls)
        layout.addWidget(self.facebook_progress)
        layout.addWidget(output_group, stretch=1)
        self._add_collapsible_section(layout, "Activity Log", logs_group, expanded=False)

    def _reload_blog_speaker_aliases(self) -> None:
        session = self._current_session or {}
        session_speaker = session.get("speaker", {}) if isinstance(session.get("speaker", {}), dict) else {}
        current_prompt = self.blog_speaker_combo.currentText().strip()
        if current_prompt == "— Other —":
            current_prompt = str(session_speaker.get("prompt_name", "")).strip()
        current_canonical = self.blog_name_edit.text().strip() or str(session_speaker.get("canonical", "")).strip()
        self.blog_speaker_combo.blockSignals(True)
        self.blog_speaker_combo.clear()
        aliases = load_speaker_aliases()
        for alias in aliases:
            self.blog_speaker_combo.addItem(alias["promptName"])
        self.blog_speaker_combo.addItem("— Other —")
        self._select_speaker_combo(self.blog_speaker_combo, current_prompt, current_canonical)
        self.blog_speaker_combo.blockSignals(False)
        self._on_blog_speaker_changed(self.blog_speaker_combo.currentIndex())
        if self.blog_speaker_combo.currentIndex() >= len(aliases):
            self.blog_name_edit.setText(current_canonical)

    def _on_blog_speaker_changed(self, index: int) -> None:
        aliases = load_speaker_aliases()
        if 0 <= index < len(aliases):
            self._blog_speaker_preset = aliases[index]
            self.blog_name_edit.blockSignals(True)
            self.blog_name_edit.setText(self._blog_speaker_preset["canonical"])
            self.blog_name_edit.blockSignals(False)
        else:
            self._blog_speaker_preset = None

    def _set_active_publish_section(self, section: QWidget) -> None:
        for target, button in getattr(self, "_publish_nav_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(target is section)
            button.blockSignals(False)
        if hasattr(self, "publish_stack"):
            self.publish_stack.setCurrentWidget(section)
        if section is self.sermon_tab:
            self._reload_sermon_speaker_aliases()
        if section is self.blog_tab:
            self._reload_blog_speaker_aliases()

    def _scroll_publish_to(self, section: QWidget) -> None:
        self._set_active_publish_section(section)

    @staticmethod
    def _set_text_panel_height(
        widget: QPlainTextEdit,
        min_height: int = 120,
        max_height: int | None = 180,
    ) -> None:
        widget.setMinimumHeight(min_height)
        if max_height is None:
            widget.setMaximumHeight(16777215)
        else:
            widget.setMaximumHeight(max_height)

    def _build_publish_source_group(self, description: str, accent: str) -> tuple[QFrame, QLabel]:
        group = QFrame()
        group.setObjectName("publishSourceStrip")
        group.setProperty("accent", accent)
        group.setToolTip(description)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        label = QLabel("Source")
        label.setObjectName("publishSourceLabel")
        status = QLabel("Load a session to use this destination.")
        status.setObjectName("publishSourceStatus")
        status.setWordWrap(True)

        layout.addWidget(label, 0, Qt.AlignTop)
        layout.addWidget(status, stretch=1)
        return group, status

    @staticmethod
    def _short_publish_text(text: str, max_len: int = 56) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= max_len:
            return compact
        return f"{compact[: max_len - 3].rstrip()}..."

    def _read_current_artifact_text(self, key: str) -> str:
        if self._current_asset_dir is None:
            return ""
        session = self._current_session or load_session(self._current_asset_dir) or {}
        path = artifact_path(self._current_asset_dir, key, session=session)
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _get_publish_transcript(self) -> str:
        transcript = self._read_current_artifact_text("transcript")
        if transcript:
            return transcript
        if self._current_asset_dir is not None:
            return ""
        return self.sermon_transcript_edit.toPlainText().strip()

    def _get_publish_blog_markdown(self) -> str:
        if self._blog_last_markdown.strip():
            return self._blog_last_markdown.strip()
        text = self._read_current_artifact_text("blog")
        if text:
            return text
        if self._current_asset_dir is not None:
            return ""
        return self.blog_output_edit.toPlainText().strip()

    def _refresh_publish_source_panels(self) -> None:
        if not hasattr(self, "sermon_source_status_label"):
            return

        transcript = self._get_publish_transcript()
        transcript_ready = bool(transcript)
        transcript_words = len(transcript.split()) if transcript_ready else 0

        if self._current_asset_dir is None:
            transcript_message = "Load a session to use the transcript from Prepare."
            blog_message = "Load a session to use the saved blog draft."
        elif transcript_ready:
            transcript_message = f"Transcript from Prepare ready | {transcript_words} words"
            blog_text = self._get_publish_blog_markdown()
            if blog_text:
                title = blog_text.splitlines()[0].strip().lstrip("#").strip() or "Untitled post"
                blog_message = f"Wix Blog draft ready | {self._short_publish_text(title)}"
            else:
                blog_message = "Missing blog draft | Run Wix Blog first"
        else:
            transcript_message = "Missing transcript | Run Caption Video in Prepare first"
            blog_text = self._get_publish_blog_markdown()
            if blog_text:
                title = blog_text.splitlines()[0].strip().lstrip("#").strip() or "Untitled post"
                blog_message = f"Wix Blog draft ready | {self._short_publish_text(title)}"
            else:
                blog_message = "Missing blog draft | Run Wix Blog first"

        self.sermon_source_status_label.setText(transcript_message)
        self.blog_source_status_label.setText(transcript_message)
        self.youtube_source_status_label.setText(transcript_message)
        self.facebook_source_status_label.setText(blog_message)

        self.sermon_run_btn.setEnabled(transcript_ready)
        self.blog_run_btn.setEnabled(transcript_ready)
        self.youtube_run_btn.setEnabled(transcript_ready)
        self.facebook_run_btn.setEnabled(bool(self._get_publish_blog_markdown()))

    def _add_collapsible_section(
        self,
        layout: QVBoxLayout,
        title: str,
        section_widget: QWidget,
        *,
        expanded: bool = False,
    ) -> None:
        toggle = QPushButton()
        toggle.setCheckable(True)
        toggle.setObjectName("inlineToggleButton")
        toggle.setChecked(expanded)
        self._set_collapsible_section_state(toggle, section_widget, title, expanded)
        toggle.clicked.connect(
            lambda checked=False, btn=toggle, widget=section_widget, label=title: self._set_collapsible_section_state(
                btn, widget, label, checked
            )
        )
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addStretch(1)
        toggle_row.addWidget(toggle)
        layout.addLayout(toggle_row)
        layout.addWidget(section_widget)

    @staticmethod
    def _set_collapsible_section_state(
        button: QPushButton,
        widget: QWidget,
        title: str,
        expanded: bool,
    ) -> None:
        widget.setVisible(expanded)
        label_map = {
            "Settings": ("Advanced settings", "Hide settings"),
            "Activity Log": ("View activity log", "Hide activity log"),
            "Parsed Metadata": ("View parsed metadata", "Hide parsed metadata"),
            "Raw Model Response": ("View raw response", "Hide raw response"),
        }
        collapsed_label, expanded_label = label_map.get(title, (f"View {title.lower()}", f"Hide {title.lower()}"))
        button.setText(expanded_label if expanded else collapsed_label)

    def _create_stage_nav_button(self, label: str, page: QWidget, accent: str) -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setObjectName("stageNavButton")
        button.setProperty("accent", accent)
        button.clicked.connect(lambda checked=False, target=page: self.show_stage(target))
        return button

    def _create_substage_nav_button(self, label: str, tabs: QTabWidget, page: QWidget, accent: str = "review") -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setObjectName("substageNavButton")
        button.setProperty("accent", accent)
        button.clicked.connect(lambda checked=False, target=page, target_tabs=tabs: target_tabs.setCurrentWidget(target))
        return button

    def _create_publish_nav_button(self, label: str, page: QWidget, accent: str) -> QPushButton:
        button = QPushButton(label)
        button.setCheckable(True)
        button.setObjectName("substageNavButton")
        button.setProperty("accent", accent)
        button.clicked.connect(lambda checked=False, target=page: self._scroll_publish_to(target))
        return button

    def show_stage(self, page: QWidget) -> None:
        self.stage_stack.setCurrentWidget(page)
        for target, button in getattr(self, "_stage_nav_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(target is page)
            button.blockSignals(False)

    def _on_review_tab_changed(self, index: int) -> None:
        current = self.clips_tabs.widget(index)
        for target, button in getattr(self, "_review_nav_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(target is current)
            button.blockSignals(False)

    def _toggle_caption_advanced(self, checked: bool) -> None:
        self.caption_advanced_group.setVisible(checked)
        self.caption_advanced_btn.setText("Hide Advanced Options" if checked else "Show Advanced Options")

    def _register_process_button(self, button: QPushButton, base_text: str, step_key: str) -> None:
        button.setProperty("baseText", base_text)
        button.setProperty("stepKey", step_key)
        self._process_buttons.append(button)

    def _refresh_process_button_labels(self) -> None:
        label_map = {
            "saved": "Done",
            "loaded": "Done",
            "running": "Running",
            "failed": "Failed",
            "out of date": "Update",
        }
        for button in getattr(self, "_process_buttons", []):
            base_text = str(button.property("baseText") or button.text())
            step_key = str(button.property("stepKey") or "").strip()
            status = self._step_status(step_key) if step_key else "not started"
            suffix = label_map.get(status)
            button.setText(f"{base_text} [{suffix}]" if suffix else base_text)

    def start_auto_run(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return
        self.clips_source_combo.setCurrentIndex(0)
        self._auto_run_enabled = True
        self._auto_run_next_step = None
        self.statusBar().showMessage("Auto run started: caption, rank, then extract.", 6000)
        self.start_captioning()
        if self.active_thread is None:
            self._auto_run_enabled = False
            self._auto_run_next_step = None

    def _stop_auto_run(self, reason: str | None = None) -> None:
        self._auto_run_enabled = False
        self._auto_run_next_step = None
        if reason:
            self.statusBar().showMessage(reason, 7000)

    def _clips_source_mode(self) -> str:
        idx = int(self.clips_source_combo.currentIndex())
        if idx == 1:
            return "kept"
        if idx == 2:
            return "custom"
        return "ranked"

    def _on_clips_source_mode_changed(self, _index: int) -> None:
        mode = self._clips_source_mode()
        self.clips_json_edit.setReadOnly(mode != "custom")
        if mode != "custom":
            self._sync_extraction_payload_from_mode()

    def _sync_extraction_payload_from_mode(self) -> bool:
        mode = self._clips_source_mode()
        if mode == "custom":
            return False
        try:
            payload = self._build_extraction_payload_from_ranked(kept_only=(mode == "kept"))
        except ValueError:
            return False
        self.clips_json_edit.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        return True

    def pick_session_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select asset folder")
        if path:
            self.load_session_into_ui(Path(path).resolve())

    def open_current_session_folder(self) -> None:
        if self._current_asset_dir is None or not self._current_asset_dir.is_dir():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_asset_dir)))

    def _set_current_asset_dir(self, asset_dir: Path | None) -> None:
        self._current_asset_dir = asset_dir.resolve() if asset_dir is not None else None
        self._current_session = load_session(self._current_asset_dir) if self._current_asset_dir else None
        if self._current_asset_dir is not None:
            self._record_recent_project(self._current_asset_dir)
            self.session_asset_label.setText(self._format_sidebar_path(self._current_asset_dir))
            self.session_asset_label.setToolTip(str(self._current_asset_dir))
            self.session_open_btn.setEnabled(True)
            self.rank_folder_edit.setText(str(self._current_asset_dir))
            self.caption_output_edit.setText(str(self._current_asset_dir))
            self.clips_output_edit.setText(str(self._current_asset_dir))
            self._caption_output_overridden = False
            self._clips_output_overridden = False
        else:
            self.session_asset_label.setText("No asset folder loaded")
            self.session_asset_label.setToolTip("")
            self.session_open_btn.setEnabled(False)
        self._refresh_session_details()
        self._refresh_session_summary()

    @staticmethod
    def _format_sidebar_path(path: Path) -> str:
        parts = list(path.parts)
        if len(parts) <= 3:
            return str(path)
        return str(Path(parts[-3]) / parts[-2] / parts[-1])

    @staticmethod
    def _recent_projects_path() -> Path:
        return Path(__file__).resolve().with_name(RECENT_PROJECTS_FILENAME)

    def _load_recent_projects(self) -> None:
        path = self._recent_projects_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        except (OSError, json.JSONDecodeError):
            payload = []
        if not isinstance(payload, list):
            payload = []
        cleaned: list[str] = []
        for value in payload:
            candidate = Path(str(value)).resolve()
            if candidate.is_dir():
                as_text = str(candidate)
                if as_text not in cleaned:
                    cleaned.append(as_text)
            if len(cleaned) >= MAX_RECENT_PROJECTS:
                break
        self._recent_projects = cleaned
        self._save_recent_projects()
        self._refresh_recent_projects_list()

    def _save_recent_projects(self) -> None:
        path = self._recent_projects_path()
        try:
            path.write_text(json.dumps(self._recent_projects, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            logging.warning("Could not save recent projects list to %s", path)

    def _refresh_recent_projects_list(self) -> None:
        if not hasattr(self, "recent_projects_list"):
            return
        self.recent_projects_list.clear()
        if not self._recent_projects:
            placeholder = QListWidgetItem("No recent projects yet")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
            self.recent_projects_list.addItem(placeholder)
            return
        for raw_path in self._recent_projects:
            project_path = Path(raw_path)
            item = QListWidgetItem(f"{project_path.name} | {self._format_sidebar_path(project_path)}")
            item.setData(Qt.UserRole, str(project_path))
            item.setToolTip(str(project_path))
            self.recent_projects_list.addItem(item)

    def _record_recent_project(self, asset_dir: Path | None) -> None:
        if asset_dir is None or not asset_dir.is_dir():
            return
        raw_path = str(asset_dir.resolve())
        self._recent_projects = [raw_path, *[entry for entry in self._recent_projects if entry != raw_path]][
            :MAX_RECENT_PROJECTS
        ]
        self._save_recent_projects()
        self._refresh_recent_projects_list()

    def _open_recent_project(self, item: QListWidgetItem) -> None:
        target = item.data(Qt.UserRole)
        if not target:
            return
        path = Path(str(target)).resolve()
        if not path.is_dir():
            self._recent_projects = [entry for entry in self._recent_projects if entry != str(path)]
            self._save_recent_projects()
            self._refresh_recent_projects_list()
            QMessageBox.information(self, "Missing folder", f"The recent project folder is no longer available:\n{path}")
            return
        self.load_session_into_ui(path)

    def _write_step_status(
        self,
        asset_dir: Path | None,
        step_key: str,
        status: str,
        **extra: object,
    ) -> None:
        if asset_dir is None:
            return
        payload: dict[str, object] = {"status": status, "generated_at": now_iso()}
        payload.update(extra)
        upsert_session(asset_dir, {"steps": {step_key: payload}})
        if self._current_asset_dir is not None and asset_dir.resolve() == self._current_asset_dir:
            self._current_session = load_session(self._current_asset_dir)

    def _step_status(self, step_key: str) -> str:
        if self._current_asset_dir is None:
            return "not started"
        session = self._current_session or load_session(self._current_asset_dir) or {}
        step_payload = session.get("steps", {}) if isinstance(session.get("steps", {}), dict) else {}
        step_meta = step_payload.get(step_key, {}) if isinstance(step_payload.get(step_key, {}), dict) else {}
        explicit_status = str(step_meta.get("status", "")).strip().lower()
        if explicit_status in {"running", "failed"}:
            return explicit_status
        requirements = {
            "caption": ["subtitles_srt", "transcript", "words", "energy", "cadence", "moments"],
            "ranking": ["ranked_moments"],
            "clips": [],
            "sermon_metadata": ["sermon_metadata"],
            "blog": ["blog"],
            "youtube": ["youtube"],
            "facebook": ["facebook"],
        }
        required = requirements.get(step_key, [])
        if not required:
            if step_key == "clips":
                prefix = str((session.get("prefix", "") or (self._current_asset_dir.name if self._current_asset_dir else ""))).strip()
                clip_files = list(self._current_asset_dir.glob(f"{prefix}-clip_*.*")) if prefix else []
                if clip_files:
                    return "loaded" if "clips" in self._loaded_step_flags else "saved"
            return "not started"
        if not all(artifact_path(self._current_asset_dir, key, session=session).is_file() for key in required):
            return "not started"

        transcript_path = artifact_path(self._current_asset_dir, "transcript", session=session)
        transcript_hash = sha256_file(transcript_path) if transcript_path.is_file() else None
        fingerprints = step_meta.get("fingerprints", {}) if isinstance(step_meta.get("fingerprints", {}), dict) else {}
        saved_hash = str(fingerprints.get("transcript_sha256", "")).strip()
        if step_key in {"sermon_metadata", "blog", "youtube"} and saved_hash and transcript_hash and saved_hash != transcript_hash:
            return "out of date"
        if step_key == "facebook":
            saved_hash = str(fingerprints.get("blog_sha256", "")).strip()
            current_blog = self._get_publish_blog_markdown()
            blog_hash = sha256_text(current_blog) if current_blog else None
            if saved_hash and blog_hash and saved_hash != blog_hash:
                return "out of date"
        return "loaded" if step_key in self._loaded_step_flags else "saved"

    @staticmethod
    def _combine_statuses(statuses: list[str]) -> str:
        filtered = [status for status in statuses if status != "not started"]
        if not filtered:
            return "not started"
        if "failed" in filtered:
            return "failed"
        if "running" in filtered:
            return "running"
        if "out of date" in filtered:
            return "out of date"
        if "loaded" in filtered:
            return "loaded"
        if "saved" in filtered:
            return "saved"
        return filtered[0]

    def _refresh_session_summary(self) -> None:
        prepare_status = self._step_status("caption")
        clips_status = self._combine_statuses([self._step_status("ranking"), self._step_status("clips")])
        publish_status = self._combine_statuses(
            [
                self._step_status("sermon_metadata"),
                self._step_status("blog"),
                self._step_status("youtube"),
                self._step_status("facebook"),
            ]
        )
        self.session_status_label.setText(
            f"Prepare: {prepare_status}\nReview: {clips_status}\nPublish: {publish_status}"
        )
        self._refresh_process_button_labels()
        self._refresh_publish_source_panels()

    def _refresh_session_details(self) -> None:
        if self._current_asset_dir is None:
            self.session_video_label.setText("No source video loaded")
            self.session_meta_label.setText("Speaker/date not set")
            return

        session = self._current_session or load_session(self._current_asset_dir) or {}
        video_path = resolve_main_video_path(self._current_asset_dir, session=session)
        self.session_video_label.setText(video_path.name if video_path is not None else "No source video loaded")

        speaker = session.get("speaker", {}) if isinstance(session.get("speaker", {}), dict) else {}
        prompt_name = str(speaker.get("prompt_name", "")).strip()
        canonical = str(speaker.get("canonical", "")).strip()
        date_preached = str(session.get("date_preached", "")).strip()
        speaker_text = prompt_name or canonical or "Speaker not set"
        date_text = date_preached or "Date not set"
        self.session_meta_label.setText(f"{speaker_text} | {date_text}")

    def _select_speaker_combo(self, combo: QComboBox, prompt_name: str, canonical_name: str) -> None:
        targets = [prompt_name.strip(), canonical_name.strip()]
        for target in targets:
            if not target:
                continue
            for idx in range(combo.count()):
                if combo.itemText(idx).strip() == target:
                    combo.setCurrentIndex(idx)
                    return
        if combo.count():
            combo.setCurrentIndex(max(0, combo.count() - 1))

    def _apply_session_metadata(self, session: dict) -> None:
        date_preached = str(session.get("date_preached", "")).strip()
        if date_preached and re.match(r"^\d{4}-\d{2}-\d{2}$", date_preached):
            qdate = QDate.fromString(date_preached, "yyyy-MM-dd")
            if qdate.isValid():
                for widget in [
                    self.caption_date_edit,
                    self.clips_date_edit,
                    self.sermon_date_edit,
                    self.blog_date_edit,
                    self.youtube_date_edit,
                ]:
                    widget.setDate(qdate)
        speaker = session.get("speaker", {}) if isinstance(session.get("speaker", {}), dict) else {}
        canonical = str(speaker.get("canonical", "")).strip()
        prompt_name = str(speaker.get("prompt_name", "")).strip()
        if canonical or prompt_name:
            self._select_speaker_combo(self.caption_speaker_combo, prompt_name, canonical)
            self._select_speaker_combo(self.clips_speaker_combo, prompt_name, canonical)
            self._select_speaker_combo(self.sermon_speaker_combo, prompt_name, canonical)
            self._select_speaker_combo(self.blog_speaker_combo, prompt_name, canonical)
            self._select_speaker_combo(self.youtube_speaker_combo, prompt_name, canonical)
        if canonical:
            self.sermon_name_edit.setText(canonical)
            self.blog_name_edit.setText(canonical)
            self.youtube_name_edit.setText(canonical)

    def load_session_into_ui(self, asset_dir: Path) -> None:
        asset_dir = asset_dir.resolve()
        if not asset_dir.is_dir():
            QMessageBox.critical(self, "Missing folder", f"Asset folder not found:\n{asset_dir}")
            return
        self._loaded_step_flags.clear()
        self._set_current_asset_dir(asset_dir)
        session = self._current_session or {}
        self._apply_session_metadata(session)
        self.sermon_transcript_edit.clear()
        self.blog_transcript_edit.clear()
        self.youtube_transcript_edit.clear()
        self.sermon_raw_edit.clear()
        self.sermon_parsed_edit.clear()
        self.blog_output_edit.clear()
        self.youtube_title_edit.clear()
        self.youtube_description_edit.clear()
        self.facebook_output_edit.clear()
        self._sermon_last_transcript = ""
        self._sermon_last_raw = ""
        self._sermon_last_parsed = None
        self._blog_last_markdown = ""
        self.sermon_send_wix_btn.setEnabled(False)
        self.blog_post_wix_btn.setEnabled(False)
        self.blog_open_wix_btn.setEnabled(False)

        video_path = resolve_main_video_path(asset_dir, session=session)
        if video_path is not None and video_path.is_file():
            self.caption_video_edit.setText(str(video_path))
            self.clips_video_edit.setText(str(video_path))
            self._preview_video_path = video_path
            self.preview_player.setSource(QUrl.fromLocalFile(str(video_path)))
            self.preview_player.setPosition(0)

        transcript_path = artifact_path(asset_dir, "transcript", session=session)
        if transcript_path.is_file():
            try:
                transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                transcript_text = ""
            if transcript_text:
                self.sermon_transcript_edit.setPlainText(transcript_text)
                self.blog_transcript_edit.setPlainText(transcript_text)
                self.youtube_transcript_edit.setPlainText(transcript_text)
                self._sermon_last_transcript = transcript_text
                self._loaded_step_flags.add("caption")

        ranked_path = artifact_path(asset_dir, "ranked_moments", session=session)
        if ranked_path.is_file() and self._load_preview_report_from_path(ranked_path):
            self._loaded_step_flags.add("ranking")
            self._sync_extraction_payload_from_mode()

        prefix = str((session.get("prefix", "") or asset_dir.name)).strip()
        if prefix and list(asset_dir.glob(f"{prefix}-clip_*.*")):
            self._loaded_step_flags.add("clips")

        sermon_path = artifact_path(asset_dir, "sermon_metadata", session=session)
        if sermon_path.is_file():
            try:
                payload = json.loads(sermon_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                transcript = str(payload.get("transcript", "")).strip()
                raw_response = str(payload.get("raw_response", "")).strip()
                parsed_payload = payload.get("parsed_payload")
                if transcript:
                    self.sermon_transcript_edit.setPlainText(transcript)
                    self._sermon_last_transcript = transcript
                if raw_response:
                    self.sermon_raw_edit.setPlainText(raw_response)
                    self._sermon_last_raw = raw_response
                if isinstance(parsed_payload, dict):
                    self._sermon_last_parsed = parsed_payload
                    self.sermon_parsed_edit.setPlainText(json.dumps(parsed_payload, ensure_ascii=False, indent=2))
                    self.sermon_send_wix_btn.setEnabled(True)
                    self._loaded_step_flags.add("sermon_metadata")
                self.sermon_video_url_edit.setText(str(payload.get("video_url", "")).strip())
                if str(payload.get("name", "")).strip():
                    self.sermon_name_edit.setText(str(payload.get("name", "")).strip())

        blog_path = artifact_path(asset_dir, "blog", session=session)
        if blog_path.is_file():
            try:
                self._blog_last_markdown = blog_path.read_text(encoding="utf-8")
            except OSError:
                self._blog_last_markdown = ""
            self.blog_output_edit.setPlainText(self._blog_last_markdown)
            self.blog_post_wix_btn.setEnabled(bool(self._blog_last_markdown.strip()))
            if self._blog_last_markdown.strip():
                self._loaded_step_flags.add("blog")

        youtube_path = artifact_path(asset_dir, "youtube", session=session)
        if youtube_path.is_file():
            try:
                payload = json.loads(youtube_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                self.youtube_title_edit.setText(str(payload.get("title", "")).strip())
                self.youtube_description_edit.setPlainText(str(payload.get("description", "")).strip())
                transcript = str(payload.get("transcript", "")).strip()
                if transcript:
                    self.youtube_transcript_edit.setPlainText(transcript)
                self._loaded_step_flags.add("youtube")

        facebook_path = artifact_path(asset_dir, "facebook", session=session)
        if facebook_path.is_file():
            try:
                text = facebook_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            self.facebook_output_edit.setPlainText(text)
            if text.strip():
                self._loaded_step_flags.add("facebook")
        if self._blog_last_markdown.strip():
            self.facebook_source_edit.setPlainText(self._blog_last_markdown)

        self._refresh_session_summary()
        self.statusBar().showMessage(f"Loaded session: {asset_dir.name}", 6000)

    def _resolve_current_asset_dir(self) -> Path | None:
        candidates = [
            self._current_asset_dir,
            Path(self.rank_folder_edit.text().strip()).resolve() if self.rank_folder_edit.text().strip() else None,
            Path(self.caption_output_edit.text().strip()).resolve() if self.caption_output_edit.text().strip() else None,
            Path(self.clips_output_edit.text().strip()).resolve() if self.clips_output_edit.text().strip() else None,
            self._sermon_source_path.parent.resolve() if self._sermon_source_path and self._sermon_source_path.exists() else None,
            self._blog_source_path.parent.resolve() if self._blog_source_path and self._blog_source_path.exists() else None,
            self._youtube_source_path.parent.resolve() if self._youtube_source_path and self._youtube_source_path.exists() else None,
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_dir():
                return candidate
        return None

    def _reload_sermon_speaker_aliases(self) -> None:
        """Reload speaker_aliases.json and repopulate the Speaker dropdown."""
        session = self._current_session or {}
        session_speaker = session.get("speaker", {}) if isinstance(session.get("speaker", {}), dict) else {}
        current_prompt = self.sermon_speaker_combo.currentText().strip()
        if current_prompt == "— Other —":
            current_prompt = str(session_speaker.get("prompt_name", "")).strip()
        current_canonical = self.sermon_name_edit.text().strip() or str(session_speaker.get("canonical", "")).strip()
        self.sermon_speaker_combo.blockSignals(True)
        self.sermon_speaker_combo.clear()
        self._speaker_aliases = load_speaker_aliases()
        for alias in self._speaker_aliases:
            self.sermon_speaker_combo.addItem(alias["promptName"])
        self.sermon_speaker_combo.addItem("— Other —")
        self._select_speaker_combo(self.sermon_speaker_combo, current_prompt, current_canonical)
        self.sermon_speaker_combo.blockSignals(False)
        self._on_sermon_speaker_changed(self.sermon_speaker_combo.currentIndex())
        if self.sermon_speaker_combo.currentIndex() >= len(self._speaker_aliases):
            self.sermon_name_edit.setText(current_canonical)

    def _on_sermon_speaker_changed(self, index: int) -> None:
        if 0 <= index < len(self._speaker_aliases):
            alias = self._speaker_aliases[index]
            self._sermon_speaker_preset = alias
            self.sermon_name_edit.blockSignals(True)
            self.sermon_name_edit.setText(alias["canonical"])
            self.sermon_name_edit.blockSignals(False)
        else:
            self._sermon_speaker_preset = None

    def _clear_sermon_tab(self) -> None:
        self.sermon_raw_edit.clear()
        self.sermon_parsed_edit.clear()
        self.sermon_log.clear()
        self.sermon_progress.setValue(0)
        self._sermon_last_parsed = None
        self._sermon_last_raw = ""
        self._sermon_source_path = None
        self.sermon_speaker_combo.setCurrentIndex(len(self._speaker_aliases))
        self._sermon_speaker_preset = None
        self.sermon_name_edit.clear()
        self.sermon_send_wix_btn.setEnabled(False)
        self._refresh_publish_source_panels()

    def pick_sermon_transcript_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select transcript file",
            "",
            "Text / subtitles (*.txt *.srt *.vtt);;All files (*.*)",
        )
        if path:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
                self.sermon_transcript_edit.setPlainText(text)
                self._sermon_source_path = Path(path).resolve()
                self.load_session_into_ui(self._sermon_source_path.parent)
            except Exception as exc:
                QMessageBox.critical(self, "Read failed", str(exc))

    def start_sermon_metadata(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return
        transcript = self._get_publish_transcript()
        if not transcript:
            QMessageBox.critical(self, "Missing input", "Run Caption Video in Prepare first to create the saved transcript.")
            return
        model_name = self.sermon_model_combo.currentText().strip()
        host = self.sermon_host_edit.text().strip()
        if self._sermon_speaker_preset:
            preacher_name = self._sermon_speaker_preset["promptName"]
        else:
            preacher_name = self.sermon_name_edit.text().strip()
        date_preached = self.sermon_date_edit.date().toString("yyyy-MM-dd")
        asset_dir = self._resolve_current_asset_dir()
        if asset_dir is None:
            QMessageBox.critical(self, "Missing session", "Load or create an asset folder before generating metadata.")
            return
        self._set_current_asset_dir(asset_dir)
        canonical_name = self.sermon_name_edit.text().strip()
        upsert_session(
            asset_dir,
            {
                "date_preached": date_preached,
                "speaker": {
                    "canonical": canonical_name,
                    "prompt_name": preacher_name,
                    "slug": speaker_to_slug(canonical_name or preacher_name),
                },
                "steps": {
                    "sermon_metadata": {
                        "status": "running",
                        "generated_at": now_iso(),
                        "model": model_name,
                        "host": host,
                        "fingerprints": {"transcript_sha256": sha256_text(transcript)},
                    }
                },
            },
        )

        self._sermon_last_transcript = transcript
        self.sermon_log.clear()
        self.sermon_raw_edit.clear()
        self.sermon_parsed_edit.clear()
        self.sermon_progress.setMaximum(100)
        self.sermon_progress.setValue(0)
        self.sermon_send_wix_btn.setEnabled(False)
        self._set_busy(True)

        worker = SermonMetadataWorker(
            transcript=transcript,
            model_name=model_name,
            host=host,
            preacher_name=preacher_name,
            date_preached=date_preached,
        )
        worker.log.connect(self._on_sermon_log)
        worker.progress.connect(self._on_sermon_progress)
        worker.chunk.connect(self._on_sermon_chunk)
        self._run_worker(worker, self._on_sermon_done)

    def _on_sermon_log(self, text: str) -> None:
        self._append_log(self.sermon_log, text)

    def _on_sermon_chunk(self, text: str) -> None:
        """Append streamed AI response to raw panel and scroll to end."""
        self.sermon_raw_edit.insertPlainText(text)
        self.sermon_raw_edit.verticalScrollBar().setValue(
            self.sermon_raw_edit.verticalScrollBar().maximum()
        )

    def _on_sermon_progress(self, current: int, total: int, label: str) -> None:
        self.sermon_progress.setMaximum(max(total, 1))
        self.sermon_progress.setValue(min(current, max(total, 1)))
        self.sermon_progress.setFormat(label)

    def _on_sermon_done(self, success: bool, message: str, raw_response: str, parsed_payload: object) -> None:
        self._set_busy(False)
        self._sermon_last_raw = raw_response or ""
        self.sermon_raw_edit.setPlainText(self._sermon_last_raw)
        asset_dir = self._resolve_current_asset_dir()
        if success and isinstance(parsed_payload, dict):
            self._sermon_last_parsed = parsed_payload
            self.sermon_parsed_edit.setPlainText(
                json.dumps(parsed_payload, ensure_ascii=False, indent=2)
            )
            if asset_dir is not None:
                raw_path = artifact_path(asset_dir, "sermon_metadata_raw", session=load_session(asset_dir))
                data_path = artifact_path(asset_dir, "sermon_metadata", session=load_session(asset_dir))
                raw_path.write_text(self._sermon_last_raw, encoding="utf-8")
                payload = {
                    "transcript": self._sermon_last_transcript,
                    "raw_response": self._sermon_last_raw,
                    "parsed_payload": parsed_payload,
                    "video_url": self.sermon_video_url_edit.text().strip(),
                    "name": self.sermon_name_edit.text().strip(),
                    "speaker_prompt_name": self._sermon_speaker_preset["promptName"] if self._sermon_speaker_preset else self.sermon_name_edit.text().strip(),
                    "date_preached": self.sermon_date_edit.date().toString("yyyy-MM-dd"),
                    "model": self.sermon_model_combo.currentText().strip(),
                    "host": self.sermon_host_edit.text().strip(),
                    "generated_at": now_iso(),
                }
                data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                upsert_session(
                    asset_dir,
                    {
                        "artifacts": {
                            "sermon_metadata_raw": raw_path.name,
                            "sermon_metadata": data_path.name,
                        },
                        "steps": {
                            "sermon_metadata": {
                                "status": "saved",
                                "generated_at": now_iso(),
                                "model": self.sermon_model_combo.currentText().strip(),
                                "host": self.sermon_host_edit.text().strip(),
                                "fingerprints": {"transcript_sha256": sha256_text(self._sermon_last_transcript)},
                            }
                        },
                    },
                )
                self._loaded_step_flags.add("sermon_metadata")
            self.sermon_send_wix_btn.setEnabled(True)
            self._append_log(self.sermon_log, message)
            self.statusBar().showMessage("Sermon metadata ready. You can Send to Wix.", 6000)
        else:
            self._sermon_last_parsed = None
            self.sermon_send_wix_btn.setEnabled(False)
            self._write_step_status(asset_dir, "sermon_metadata", "failed", error=message)
            self._append_log(self.sermon_log, f"ERROR: {message}")
            self.statusBar().showMessage("Metadata generation failed. See log.", 8000)
        self._refresh_session_summary()

    def send_sermon_to_wix(self) -> None:
        if not self._sermon_last_parsed:
            QMessageBox.warning(self, "No metadata", "Generate metadata first, then Send to Wix.")
            return
        try:
            get_wix_config()
        except ValueError as e:
            QMessageBox.critical(self, "Wix config", str(e))
            return
        transcript = self._get_publish_transcript()
        video_url = self.sermon_video_url_edit.text().strip()
        name = self.sermon_name_edit.text().strip()
        date_str = self.sermon_date_edit.date().toString("yyyy-MM-dd")
        self._append_log(self.sermon_log, "Sending to Wix CMS...")
        try:
            result = create_sermon_item(
                payload=self._sermon_last_parsed,
                transcript=transcript,
                video_url=video_url,
                name=name,
                date_preached=date_str,
            )
            self._append_log(self.sermon_log, "Wix create succeeded.")
            item = (result or {}).get("dataItem") or result
            self._append_log(self.sermon_log, json.dumps(item, ensure_ascii=False, indent=2))
            self.statusBar().showMessage("Sermon created in Wix CMS.", 6000)
        except ValueError as e:
            self._append_log(self.sermon_log, f"Config/auth error: {e}")
            QMessageBox.critical(self, "Wix error", str(e))
        except RuntimeError as e:
            self._append_log(self.sermon_log, str(e))
            QMessageBox.critical(self, "Wix API error", str(e))

    def _clear_blog_tab(self) -> None:
        self.blog_output_edit.clear()
        self.blog_log.clear()
        self.blog_progress.setValue(0)
        self._blog_wix_post_id = None
        self._blog_source_path = None
        self.blog_post_wix_btn.setEnabled(False)
        self.blog_open_wix_btn.setEnabled(False)
        aliases = load_speaker_aliases()
        self.blog_speaker_combo.setCurrentIndex(len(aliases))
        self._blog_speaker_preset = None
        self.blog_name_edit.clear()
        self._refresh_publish_source_panels()

    def pick_blog_transcript_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select transcript file",
            "",
            "Text / subtitles (*.txt *.srt *.vtt);;All files (*.*)",
        )
        if path:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
                self.blog_transcript_edit.setPlainText(text)
                self._blog_source_path = Path(path).resolve()
                self.load_session_into_ui(self._blog_source_path.parent)
            except Exception as exc:
                QMessageBox.critical(self, "Read failed", str(exc))

    def start_blog_post(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return
        transcript = self._get_publish_transcript()
        if not transcript:
            QMessageBox.critical(self, "Missing input", "Run Caption Video in Prepare first to create the saved transcript.")
            return
        model_name = self.blog_model_combo.currentText().strip()
        host = self.blog_host_edit.text().strip()
        if self._blog_speaker_preset:
            preacher_name = self._blog_speaker_preset["promptName"]
        else:
            preacher_name = self.blog_name_edit.text().strip()
        date_preached = self.blog_date_edit.date().toString("yyyy-MM-dd")
        asset_dir = self._resolve_current_asset_dir()
        if asset_dir is None:
            QMessageBox.critical(self, "Missing session", "Load or create an asset folder before generating a blog post.")
            return
        self._set_current_asset_dir(asset_dir)
        canonical_name = self.blog_name_edit.text().strip()
        upsert_session(
            asset_dir,
            {
                "date_preached": date_preached,
                "speaker": {
                    "canonical": canonical_name,
                    "prompt_name": preacher_name,
                    "slug": speaker_to_slug(canonical_name or preacher_name),
                },
                "steps": {
                    "blog": {
                        "status": "running",
                        "generated_at": now_iso(),
                        "model": model_name,
                        "host": host,
                        "fingerprints": {"transcript_sha256": sha256_text(transcript)},
                    }
                },
            },
        )

        self.blog_log.clear()
        self.blog_output_edit.clear()
        self.blog_progress.setMaximum(100)
        self.blog_progress.setValue(0)
        self.blog_post_wix_btn.setEnabled(False)
        self.blog_open_wix_btn.setEnabled(False)
        self._set_busy(True)

        worker = BlogPostWorker(
            transcript=transcript,
            model_name=model_name,
            host=host,
            preacher_name=preacher_name,
            date_preached=date_preached,
        )
        worker.log.connect(self._on_blog_log)
        worker.progress.connect(self._on_blog_progress)
        worker.chunk.connect(self._on_blog_chunk)
        self._run_worker(worker, self._on_blog_done)

    def _on_blog_log(self, text: str) -> None:
        self._append_log(self.blog_log, text)

    def _on_blog_chunk(self, text: str) -> None:
        self.blog_output_edit.insertPlainText(text)
        self.blog_output_edit.verticalScrollBar().setValue(
            self.blog_output_edit.verticalScrollBar().maximum()
        )

    def _on_blog_progress(self, current: int, total: int, label: str) -> None:
        self.blog_progress.setMaximum(max(total, 1))
        self.blog_progress.setValue(min(current, max(total, 1)))
        self.blog_progress.setFormat(label)

    def _on_blog_done(self, success: bool, message: str, full_markdown: str) -> None:
        self._set_busy(False)
        raw = full_markdown or ""
        # Inject italic "adapted from {speaker}'s sermon on {date}" with dividers before and after
        speaker_display = (self._blog_speaker_preset["promptName"] if self._blog_speaker_preset else self.blog_name_edit.text().strip()) or "the speaker"
        date_display = self.blog_date_edit.date().toString("yyyy-MM-dd")
        adapted_line = f"*This post was adapted from {speaker_display}'s sermon on {date_display}.*"
        lines = raw.strip().split("\n")
        title = lines[0].strip() if lines else ""
        rest = "\n".join(lines[1:]).lstrip() if len(lines) > 1 else ""
        self._blog_last_markdown = f"{title}\n\n---\n\n{adapted_line}\n\n---\n\n{rest}".strip() if title else raw
        self.blog_output_edit.setPlainText(self._blog_last_markdown)
        if success:
            asset_dir = self._resolve_current_asset_dir()
            if asset_dir is not None:
                blog_path = artifact_path(asset_dir, "blog", session=load_session(asset_dir))
                blog_path.write_text(self._blog_last_markdown, encoding="utf-8")
                upsert_session(
                    asset_dir,
                    {
                        "artifacts": {"blog": blog_path.name},
                        "steps": {
                            "blog": {
                                "status": "saved",
                                "generated_at": now_iso(),
                                "model": self.blog_model_combo.currentText().strip(),
                                "host": self.blog_host_edit.text().strip(),
                                "fingerprints": {"transcript_sha256": sha256_text(self._get_publish_transcript())},
                            }
                        },
                    },
                )
                self._loaded_step_flags.add("blog")
            self._append_log(self.blog_log, message)
            self.blog_post_wix_btn.setEnabled(bool(self._blog_last_markdown.strip()))
            self.statusBar().showMessage("Blog post ready. You can Post to Wix as draft.", 6000)
        else:
            self.blog_post_wix_btn.setEnabled(False)
            asset_dir = self._resolve_current_asset_dir()
            self._write_step_status(asset_dir, "blog", "failed", error=message)
            self._append_log(self.blog_log, f"ERROR: {message}")
            self.statusBar().showMessage("Blog post generation failed. See log.", 8000)
        self._refresh_session_summary()

    def post_blog_to_wix(self) -> None:
        text = (self._blog_last_markdown or self.blog_output_edit.toPlainText()).strip()
        if not text:
            QMessageBox.warning(self, "No post", "Generate a post first, then Post to Wix as draft.")
            return
        lines = text.split("\n")
        title = (lines[0].strip() or "Untitled").lstrip("#").strip()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        excerpt = ""
        if body:
            first_para = body.split("\n\n")[0].strip() if "\n\n" in body else body.split("\n")[0].strip()
            excerpt = first_para[:300] if len(first_para) > 300 else first_para
        try:
            result = create_blog_draft(title=title, markdown_body=body, excerpt=excerpt)
            self._blog_wix_post_id = None
            draft = (result or {}).get("draftPost") or (result or {}).get("post") or result
            if isinstance(draft, dict) and draft.get("id"):
                self._blog_wix_post_id = str(draft["id"])
            self._append_log(self.blog_log, "Draft created in Wix Blog.")
            self.blog_open_wix_btn.setEnabled(True)
            self.statusBar().showMessage("Draft posted. Use Open in Wix to go to the dashboard.", 6000)
        except ValueError as e:
            self._append_log(self.blog_log, str(e))
            QMessageBox.critical(self, "Wix config", str(e))
        except RuntimeError as e:
            self._append_log(self.blog_log, str(e))
            QMessageBox.critical(self, "Wix API error", str(e))

    def open_blog_in_wix(self) -> None:
        try:
            url = get_wix_dashboard_url()
            QDesktopServices.openUrl(QUrl(url))
            self.statusBar().showMessage("Dashboard opened. Go to Blog > Posts to edit your draft.", 6000)
        except ValueError as e:
            QMessageBox.critical(self, "Wix config", str(e))

    def _clear_youtube_tab(self) -> None:
        self.youtube_title_edit.clear()
        self.youtube_description_edit.clear()
        self.youtube_log.clear()
        self.youtube_progress.setValue(0)
        self._youtube_source_path = None
        self._refresh_publish_source_panels()

    def _youtube_load_transcript(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load transcript", "", "Text / subtitles (*.txt *.srt *.vtt);;All files (*.*)"
        )
        if path:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
                self.youtube_transcript_edit.setPlainText(text)
                self._youtube_source_path = Path(path).resolve()
                self.load_session_into_ui(self._youtube_source_path.parent)
            except OSError as e:
                QMessageBox.critical(self, "Load failed", str(e))

    def start_youtube(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return
        transcript = self._get_publish_transcript()
        if not transcript:
            QMessageBox.critical(self, "Missing input", "Run Caption Video in Prepare first to create the saved transcript.")
            return
        aliases = load_speaker_aliases()
        idx = self.youtube_speaker_combo.currentIndex()
        preacher_name = aliases[idx]["promptName"] if 0 <= idx < len(aliases) else self.youtube_name_edit.text().strip()
        date_preached = self.youtube_date_edit.date().toString("yyyy-MM-dd")
        model_name = self.youtube_model_combo.currentText().strip()
        host = self.youtube_host_edit.text().strip()
        asset_dir = self._resolve_current_asset_dir()
        if asset_dir is None:
            QMessageBox.critical(self, "Missing session", "Load or create an asset folder before generating YouTube copy.")
            return
        self._set_current_asset_dir(asset_dir)
        canonical_name = self.youtube_name_edit.text().strip()
        upsert_session(
            asset_dir,
            {
                "date_preached": date_preached,
                "speaker": {
                    "canonical": canonical_name,
                    "prompt_name": preacher_name,
                    "slug": speaker_to_slug(canonical_name or preacher_name),
                },
                "steps": {
                    "youtube": {
                        "status": "running",
                        "generated_at": now_iso(),
                        "model": model_name,
                        "host": host,
                        "fingerprints": {"transcript_sha256": sha256_text(transcript)},
                    }
                },
            },
        )
        self.youtube_log.clear()
        self.youtube_title_edit.clear()
        self.youtube_description_edit.clear()
        self.youtube_progress.setMaximum(100)
        self.youtube_progress.setValue(0)
        self._set_busy(True)
        worker = YouTubeWorker(
            transcript=transcript,
            model_name=model_name,
            host=host,
            preacher_name=preacher_name or "",
            date_preached=date_preached,
        )
        worker.log.connect(self._on_youtube_log)
        worker.progress.connect(self._on_youtube_progress)
        self._run_worker(worker, self._on_youtube_done)

    def _on_youtube_log(self, text: str) -> None:
        self._append_log(self.youtube_log, text)

    def _on_youtube_progress(self, current: int, total: int, label: str) -> None:
        self.youtube_progress.setMaximum(max(total, 1))
        self.youtube_progress.setValue(min(current, max(total, 1)))
        self.youtube_progress.setFormat(label)

    def _on_youtube_done(self, success: bool, message: str, title: str, description: str) -> None:
        self._set_busy(False)
        if success:
            self.youtube_title_edit.setText(title)
            self.youtube_description_edit.setPlainText(description)
            asset_dir = self._resolve_current_asset_dir()
            if asset_dir is not None:
                youtube_path = artifact_path(asset_dir, "youtube", session=load_session(asset_dir))
                payload = {
                    "title": title,
                    "description": description,
                    "transcript": self._get_publish_transcript(),
                    "name": self.youtube_name_edit.text().strip(),
                    "speaker_prompt_name": self.youtube_speaker_combo.currentText().strip(),
                    "date_preached": self.youtube_date_edit.date().toString("yyyy-MM-dd"),
                    "model": self.youtube_model_combo.currentText().strip(),
                    "host": self.youtube_host_edit.text().strip(),
                    "generated_at": now_iso(),
                }
                youtube_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                upsert_session(
                    asset_dir,
                    {
                        "artifacts": {"youtube": youtube_path.name},
                        "steps": {
                            "youtube": {
                                "status": "saved",
                                "generated_at": now_iso(),
                                "model": self.youtube_model_combo.currentText().strip(),
                                "host": self.youtube_host_edit.text().strip(),
                                "fingerprints": {"transcript_sha256": sha256_text(self._get_publish_transcript())},
                            }
                        },
                    },
                )
                self._loaded_step_flags.add("youtube")
            self._append_log(self.youtube_log, message)
            self.statusBar().showMessage("YouTube title and description ready. Copy and upload in YouTube Studio.", 6000)
        else:
            asset_dir = self._resolve_current_asset_dir()
            self._write_step_status(asset_dir, "youtube", "failed", error=message)
            self._append_log(self.youtube_log, f"ERROR: {message}")
            self.statusBar().showMessage("YouTube generation failed.", 6000)
        self._refresh_session_summary()

    def _youtube_copy_title(self) -> None:
        title = self.youtube_title_edit.text().strip()
        if title:
            QApplication.clipboard().setText(title)
            self.statusBar().showMessage("Title copied to clipboard.", 3000)

    def _youtube_copy_description(self) -> None:
        desc = self.youtube_description_edit.toPlainText().strip()
        if desc:
            QApplication.clipboard().setText(desc)
            self.statusBar().showMessage("Description copied to clipboard.", 3000)

    def _youtube_copy_both(self) -> None:
        title = self.youtube_title_edit.text().strip()
        desc = self.youtube_description_edit.toPlainText().strip()
        if title or desc:
            QApplication.clipboard().setText(f"{title}\n\n{desc}" if title and desc else (title or desc))
            self.statusBar().showMessage("Title and description copied to clipboard.", 3000)

    def _youtube_open_studio(self) -> None:
        QDesktopServices.openUrl(QUrl("https://studio.youtube.com"))
        self.statusBar().showMessage("Open YouTube Studio, then paste title/description and upload your video.", 8000)

    def _clear_facebook_tab(self) -> None:
        self.facebook_output_edit.clear()
        self.facebook_log.clear()
        self.facebook_progress.setValue(0)
        self._refresh_publish_source_panels()

    def _facebook_load_from_blog(self) -> None:
        text = (self._blog_last_markdown or self.blog_output_edit.toPlainText()).strip()
        if text:
            self.facebook_source_edit.setPlainText(text)
            self.statusBar().showMessage("Blog post loaded from Blog tab.", 4000)
        else:
            QMessageBox.information(
                self, "No blog post", "Generate a blog post first in the Blog Post tab, or paste markdown here."
            )

    def start_facebook_post(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return
        blog_md = self._get_publish_blog_markdown()
        if not blog_md:
            QMessageBox.critical(self, "Missing input", "Run Wix Blog first to create the saved blog draft for Facebook.")
            return
        model_name = self.facebook_model_combo.currentText().strip()
        host = self.facebook_host_edit.text().strip()
        asset_dir = self._resolve_current_asset_dir()
        if asset_dir is None:
            QMessageBox.critical(self, "Missing session", "Load or create an asset folder before generating Facebook copy.")
            return
        self._set_current_asset_dir(asset_dir)
        upsert_session(
            asset_dir,
            {
                "steps": {
                    "facebook": {
                        "status": "running",
                        "generated_at": now_iso(),
                        "model": model_name,
                        "host": host,
                        "fingerprints": {"blog_sha256": sha256_text(blog_md)},
                    }
                }
            },
        )
        self.facebook_log.clear()
        self.facebook_output_edit.clear()
        self.facebook_progress.setMaximum(100)
        self.facebook_progress.setValue(0)
        self._set_busy(True)
        worker = FacebookPostWorker(blog_post_markdown=blog_md, model_name=model_name, host=host)
        worker.log.connect(self._on_facebook_log)
        worker.progress.connect(self._on_facebook_progress)
        self._run_worker(worker, self._on_facebook_done)

    def _on_facebook_log(self, text: str) -> None:
        self._append_log(self.facebook_log, text)

    def _on_facebook_progress(self, current: int, total: int, label: str) -> None:
        self.facebook_progress.setMaximum(max(total, 1))
        self.facebook_progress.setValue(min(current, max(total, 1)))
        self.facebook_progress.setFormat(label)

    def _on_facebook_done(self, success: bool, message: str, post_text: str) -> None:
        self._set_busy(False)
        if success:
            self.facebook_output_edit.setPlainText(post_text)
            asset_dir = self._resolve_current_asset_dir()
            if asset_dir is not None:
                facebook_path = artifact_path(asset_dir, "facebook", session=load_session(asset_dir))
                facebook_path.write_text(post_text, encoding="utf-8")
                upsert_session(
                    asset_dir,
                    {
                        "artifacts": {"facebook": facebook_path.name},
                        "steps": {
                            "facebook": {
                                "status": "saved",
                                "generated_at": now_iso(),
                                "model": self.facebook_model_combo.currentText().strip(),
                                "host": self.facebook_host_edit.text().strip(),
                                "fingerprints": {"blog_sha256": sha256_text(self._get_publish_blog_markdown())},
                            }
                        },
                    },
                )
                self._loaded_step_flags.add("facebook")
            self._append_log(self.facebook_log, message)
            self.statusBar().showMessage("Facebook post ready. Copy to clipboard to paste on Facebook.", 6000)
        else:
            asset_dir = self._resolve_current_asset_dir()
            self._write_step_status(asset_dir, "facebook", "failed", error=message)
            self._append_log(self.facebook_log, f"ERROR: {message}")
            self.statusBar().showMessage("Facebook post generation failed.", 6000)
        self._refresh_session_summary()

    def _facebook_copy(self) -> None:
        text = self.facebook_output_edit.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Facebook post copied to clipboard.", 3000)
        else:
            QMessageBox.information(self, "No post", "Generate a Facebook post first.")

    def _build_rank_tab(self) -> None:
        self.rank_folder_edit = QLineEdit()
        self.rank_folder_edit.setPlaceholderText("Folder containing words.json/energy.json/cadence.json/moments.json")
        self.rank_model_combo = QComboBox()
        self.rank_model_combo.addItems(
            [
                "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-sonnet-4-6",
                "arn:aws:bedrock:us-east-1:644190502535:inference-profile/us.anthropic.claude-opus-4-6-v1",
            ]
        )

        self.rank_candidate_spin = QSpinBox()
        self.rank_candidate_spin.setRange(1, 100)
        self.rank_candidate_spin.setValue(20)

        self.rank_output_spin = QSpinBox()
        self.rank_output_spin.setRange(1, 20)
        self.rank_output_spin.setValue(10)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)

        self.rank_results = QPlainTextEdit()
        self.rank_results.setReadOnly(True)
        self.rank_results.setFont(mono)

        self.rank_log = QPlainTextEdit()
        self.rank_log.setReadOnly(True)
        self.rank_log.setFont(mono)

        self.rank_progress = QProgressBar()
        self.rank_progress.setMinimum(0)
        self.rank_progress.setValue(0)

        self.rank_run_btn = QPushButton("Rank Moments")
        self.rank_run_btn.setObjectName("primaryButton")
        self.rank_run_btn.setProperty("accent", "review")
        self.rank_run_btn.clicked.connect(self.start_ranking)
        rank_clear_btn = QPushButton("Clear")
        rank_clear_btn.clicked.connect(self._clear_rank_panels)

        layout = QVBoxLayout(self.rank_tab)
        layout.setSpacing(12)

        inputs = QGroupBox("Ranking Setup")
        grid = QGridLayout(inputs)
        grid.setContentsMargins(14, 18, 14, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Asset folder:"), 0, 0)
        grid.addWidget(self.rank_folder_edit, 0, 1)
        browse_folder = QPushButton("Browse...")
        browse_folder.clicked.connect(self.pick_rank_folder)
        grid.addWidget(browse_folder, 0, 2)

        grid.addWidget(QLabel("Model:"), 1, 0)
        grid.addWidget(self.rank_model_combo, 1, 1)
        grid.addWidget(QLabel("Candidates:"), 1, 2)
        grid.addWidget(self.rank_candidate_spin, 1, 3)

        grid.addWidget(QLabel("Output clips:"), 2, 0)
        grid.addWidget(self.rank_output_spin, 2, 1)
        grid.addWidget(
            QLabel(
                "Model format: bedrock/<model-id> (streaming) or Ollama model id (local)."
            ),
            2,
            2,
            1,
            2,
        )
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.rank_run_btn)
        controls.addWidget(rank_clear_btn)
        controls.addStretch(1)

        results_group = QGroupBox("Ranked Clips")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(10, 14, 10, 10)
        results_layout.addWidget(self.rank_results)

        logs = QGroupBox("Activity")
        log_layout = QVBoxLayout(logs)
        log_layout.setContentsMargins(10, 14, 10, 10)
        log_layout.addWidget(self.rank_log)

        layout.addWidget(inputs)
        layout.addLayout(controls)
        layout.addWidget(self.rank_progress)
        layout.addWidget(results_group, stretch=2)
        layout.addWidget(logs, stretch=1)

    def _build_preview_tab(self) -> None:
        self.preview_clips_list = QListWidget()
        self.preview_details = QPlainTextEdit()
        self.preview_details.setReadOnly(True)
        self.preview_header = QLabel("Run Rank Moments to load candidates into the review workspace.")
        self.preview_header.setStyleSheet("color: #c8d0df;")
        self.preview_header.setWordWrap(True)

        self.preview_video_widget = QVideoWidget()
        self.preview_video_widget.setMinimumHeight(280)
        self.preview_video_widget.setStyleSheet(
            "background: #10131a; border: 1px solid #2b303b; border-radius: 10px;"
        )

        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_audio.setVolume(0.8)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.preview_video_widget)
        self.preview_player.positionChanged.connect(self._on_preview_position_changed)

        self.preview_play_btn = QPushButton("Play / Pause Clip")
        self.preview_play_btn.setObjectName("primaryButton")
        self.preview_play_btn.setProperty("accent", "review")
        self.preview_play_btn.clicked.connect(self.toggle_preview_playback)

        self.preview_open_btn = QPushButton("Open Source")
        self.preview_open_btn.clicked.connect(self.pick_preview_video)

        self.preview_stop_btn = QPushButton("Stop")
        self.preview_stop_btn.clicked.connect(self.stop_preview_playback)

        self.preview_keep_btn = QPushButton("\U0001F44D")
        self.preview_keep_btn.setToolTip("Keep")
        self.preview_keep_btn.clicked.connect(lambda: self._record_clip_feedback("up"))
        self.preview_skip_btn = QPushButton("\U0001F44E")
        self.preview_skip_btn.setToolTip("Skip")
        self.preview_skip_btn.clicked.connect(lambda: self._record_clip_feedback("down"))
        self._feedback_labels: dict[str, str] = {}
        self.preview_feedback_note = QLineEdit()
        self.preview_feedback_note.setPlaceholderText("Optional note (why keep/skip)")

        self.preview_extend_start_btn = QPushButton("+5s Start")
        self.preview_extend_start_btn.setToolTip("Extend clip start by 5 seconds earlier")
        self.preview_extend_start_btn.clicked.connect(lambda: self._extend_clip_boundary("start", 5.0))

        self.preview_extend_end_btn = QPushButton("+5s End")
        self.preview_extend_end_btn.setToolTip("Extend clip end by 5 seconds later")
        self.preview_extend_end_btn.clicked.connect(lambda: self._extend_clip_boundary("end", 5.0))

        self.preview_export_btn = QPushButton("Export Clip")
        self.preview_export_btn.clicked.connect(self._export_current_clip)
        self.preview_send_extract_btn = QPushButton("Queue All")
        self.preview_send_extract_btn.clicked.connect(self._load_extraction_from_ranked)
        self.preview_send_kept_btn = QPushButton("Queue Kept")
        self.preview_send_kept_btn.clicked.connect(lambda: self._load_extraction_from_ranked(kept_only=True))

        self.preview_seek_current = QLabel("00:00")
        self.preview_seek_total = QLabel("00:00")
        self.preview_seek_current.setStyleSheet("color: #c8d0df;")
        self.preview_seek_total.setStyleSheet("color: #c8d0df;")
        self.preview_seek_slider = ClickJumpSlider()
        self.preview_seek_slider.setOrientation(Qt.Horizontal)
        self.preview_seek_slider.setRange(0, 0)
        self.preview_seek_slider.setEnabled(False)
        self.preview_seek_slider.sliderPressed.connect(self._on_preview_seek_pressed)
        self.preview_seek_slider.sliderReleased.connect(self._on_preview_seek_released)
        self.preview_seek_slider.valueChanged.connect(self._on_preview_seek_value_changed)

        self.preview_score_widgets: dict[str, dict[str, object]] = {}
        self.preview_scores_group = QGroupBox("Editorial Scores (LLM)")
        score_layout = QGridLayout(self.preview_scores_group)
        score_layout.setContentsMargins(10, 14, 10, 10)
        score_layout.setHorizontalSpacing(8)
        score_layout.setVerticalSpacing(8)
        for col, key in enumerate(["editor", "hook", "cadence", "standalone", "emotion"]):
            title = "Editor" if key == "editor" else key.capitalize()
            frame = QFrame()
            frame.setObjectName("scoreCard")
            frame.setStyleSheet(
                "QFrame#scoreCard { border: 1px solid #384053; border-radius: 8px; background: #171b23; }"
            )
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(8, 6, 8, 6)
            frame_layout.setSpacing(4)
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #c8d0df;")
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: 700; color: #f0f3fa;")
            bar = QProgressBar()
            bar.setRange(0, DISPLAY_SCORE_MAX)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            frame_layout.addWidget(title_label)
            frame_layout.addWidget(value_label)
            frame_layout.addWidget(bar)
            score_layout.addWidget(frame, 0, col)
            self.preview_score_widgets[key] = {"value_label": value_label, "bar": bar, "frame": frame}

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.preview_clips_list.setFont(mono)
        self.preview_details.setFont(mono)
        self.preview_clips_list.currentRowChanged.connect(self._on_preview_clip_selected)

        layout = QVBoxLayout(self.preview_tab)
        layout.setSpacing(12)

        player_group = QGroupBox("Clip Preview")
        player_layout = QVBoxLayout(player_group)
        player_layout.setContentsMargins(10, 14, 10, 10)
        player_transport = QHBoxLayout()
        player_transport.addWidget(self.preview_play_btn)
        player_transport.addWidget(self.preview_open_btn)
        player_transport.addWidget(self.preview_stop_btn)
        player_transport.addStretch(1)
        player_actions = QHBoxLayout()
        player_actions.addWidget(self.preview_extend_start_btn)
        player_actions.addWidget(self.preview_extend_end_btn)
        player_actions.addWidget(self.preview_export_btn)
        player_actions.addWidget(self.preview_send_extract_btn)
        player_actions.addWidget(self.preview_send_kept_btn)
        player_actions.addStretch(1)
        seek_row = QHBoxLayout()
        seek_row.addWidget(self.preview_seek_current)
        seek_row.addWidget(self.preview_seek_slider, stretch=1)
        seek_row.addWidget(self.preview_seek_total)
        player_layout.addWidget(self.preview_video_widget)
        player_layout.addLayout(seek_row)
        player_layout.addLayout(player_transport)
        player_layout.addLayout(player_actions)

        clips_group = QGroupBox("Ranked Clips")
        clips_layout = QVBoxLayout(clips_group)
        clips_layout.setContentsMargins(10, 14, 10, 10)
        clips_layout.addWidget(self.preview_clips_list)

        detail_group = QGroupBox("Selected Clip Details")
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.setContentsMargins(10, 14, 10, 10)
        self.preview_reasoning_flag = QLabel("")
        self.preview_reasoning_flag.setWordWrap(True)
        self.preview_reasoning_flag.setVisible(False)
        self.preview_reasoning_flag.setStyleSheet("color: #f59f00;")
        detail_layout.addWidget(self.preview_reasoning_flag)
        feedback_row = QHBoxLayout()
        feedback_row.addWidget(self.preview_keep_btn)
        feedback_row.addWidget(self.preview_skip_btn)
        feedback_row.addWidget(self.preview_feedback_note, stretch=1)
        detail_layout.addLayout(feedback_row)
        detail_layout.addWidget(self.preview_details)

        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        left_column.addWidget(player_group, stretch=4)
        left_column.addWidget(clips_group, stretch=2)

        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        right_column.addWidget(self.preview_scores_group, stretch=1)
        right_column.addWidget(detail_group, stretch=5)

        content = QHBoxLayout()
        content.setSpacing(12)
        content.addLayout(left_column, stretch=5)
        content.addLayout(right_column, stretch=7)

        layout.addWidget(self.preview_header)
        layout.addLayout(content, stretch=1)
        self._clear_preview_score_cards()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 13px;
                color: #e7e9ee;
                background: #181a1f;
            }
            QMainWindow {
                background: transparent;
            }
            QWidget#windowShell {
                background: transparent;
            }
            QWidget#windowContent {
                background: transparent;
            }
            QFrame#windowSurface {
                border: 1px solid #2b303b;
                border-radius: 16px;
                background: #181a1f;
            }
            QFrame#windowTitleBar {
                border-bottom: 1px solid #2b303b;
                background: #141821;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
            }
            QLabel#windowTitleLabel {
                font-size: 14px;
                font-weight: 700;
                color: #f0f3fa;
                background: transparent;
            }
            QFrame#sidebarFrame {
                border: 1px solid #2b303b;
                border-radius: 14px;
                background: #15181f;
            }
            QGroupBox#workflowPanel {
                border-color: #32506f;
                background: #18222f;
            }
            QGroupBox#sessionPanel {
                border-color: #2f5d58;
                background: #172221;
            }
            QLabel#sidebarTitle {
                font-size: 18px;
                font-weight: 700;
                color: #f0f3fa;
                padding: 2px 4px 10px 4px;
                background: transparent;
            }
            QFrame#pageHeader {
                border: 1px solid #2b303b;
                border-radius: 12px;
                background: #1a1e26;
            }
            QFrame#pageHeader[accent="prepare"] {
                border-color: #2c7b6d;
                background: #162523;
            }
            QFrame#pageHeader[accent="review"] {
                border-color: #9b6f2f;
                background: #272014;
            }
            QFrame#pageHeader[accent="publish"] {
                border-color: #6e4aa8;
                background: #211a2e;
            }
            QLabel#pageTitle {
                font-size: 20px;
                font-weight: 700;
                color: #f0f3fa;
                background: transparent;
            }
            QLabel#pageSubtitle {
                color: #b7c0d3;
                background: transparent;
            }
            QLabel#sectionSubtitle {
                color: #b7c0d3;
                background: transparent;
            }
            QFrame#publishSourceStrip {
                border: 1px solid #2b303b;
                border-radius: 10px;
                background: #1a1e26;
            }
            QFrame#publishSourceStrip[accent="sermon"] {
                border-color: #3f6dcf;
                background: #182134;
            }
            QFrame#publishSourceStrip[accent="blog"] {
                border-color: #8a57c8;
                background: #221a34;
            }
            QFrame#publishSourceStrip[accent="youtube"] {
                border-color: #c95656;
                background: #321c22;
            }
            QFrame#publishSourceStrip[accent="facebook"] {
                border-color: #4d87d7;
                background: #18263a;
            }
            QLabel#publishSourceLabel {
                color: #9aa4b8;
                font-weight: 700;
                background: transparent;
                min-width: 48px;
            }
            QLabel#publishSourceStatus {
                color: #d8deea;
                background: transparent;
            }
            QTabWidget::pane {
                border: 1px solid #2b303b;
                border-radius: 10px;
                top: -1px;
                background: #1a1e26;
            }
            QTabBar::tab {
                background: #20242d;
                border: 1px solid #2b303b;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #2a3040;
                border-color: #3a7afe;
            }
            QGroupBox {
                border: 1px solid #2b303b;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 10px;
                background: #20242d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #c8d0df;
            }
            QLineEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                border: 1px solid #343a46;
                border-radius: 8px;
                padding: 6px;
                background: #15181f;
                color: #f0f3fa;
                selection-background-color: #2f6feb;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #4f8cff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                border: 1px solid #3a4150;
                border-radius: 8px;
                padding: 8px 14px;
                background: #2a3040;
                color: #e5e9f4;
            }
            QPushButton:hover {
                background: #323a4d;
            }
            QPushButton:disabled {
                color: #8a90a0;
                background: #252a35;
                border-color: #2b303b;
            }
            QPushButton#titleBarButton {
                min-width: 34px;
                max-width: 34px;
                padding: 0;
                background: #202633;
                border-color: #354053;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton#titleBarButton:hover {
                background: #2a3241;
            }
            QPushButton#titleBarCloseButton {
                min-width: 34px;
                max-width: 34px;
                padding: 0;
                background: #5a2a34;
                border-color: #874552;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#titleBarCloseButton:hover {
                background: #723341;
            }
            QPushButton#primaryButton {
                background: #3a7afe;
                border-color: #3a7afe;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover {
                background: #4a87ff;
            }
            QPushButton#primaryButton[accent="prepare"] {
                background: #1f9d82;
                border-color: #29ba9b;
            }
            QPushButton#primaryButton[accent="prepare"]:hover {
                background: #27b694;
            }
            QPushButton#primaryButton[accent="review"] {
                background: #c08829;
                border-color: #d89d3b;
            }
            QPushButton#primaryButton[accent="review"]:hover {
                background: #d29431;
            }
            QPushButton#primaryButton[accent="sermon"] {
                background: #3f6dcf;
                border-color: #5a88ea;
            }
            QPushButton#primaryButton[accent="sermon"]:hover {
                background: #4a79de;
            }
            QPushButton#primaryButton[accent="blog"] {
                background: #8a57c8;
                border-color: #a270e1;
            }
            QPushButton#primaryButton[accent="blog"]:hover {
                background: #9864d8;
            }
            QPushButton#primaryButton[accent="youtube"] {
                background: #c95656;
                border-color: #e26f6f;
            }
            QPushButton#primaryButton[accent="youtube"]:hover {
                background: #da6262;
            }
            QPushButton#primaryButton[accent="facebook"] {
                background: #4d87d7;
                border-color: #69a0eb;
            }
            QPushButton#primaryButton[accent="facebook"]:hover {
                background: #5a94e6;
            }
            QPushButton#stageNavButton {
                text-align: left;
                padding: 10px 12px;
                background: #1f2430;
                border-color: #2f3542;
                font-weight: 600;
            }
            QPushButton#stageNavButton[accent="prepare"] {
                background: #172524;
                border-color: #2b5c56;
                color: #c6ece6;
            }
            QPushButton#stageNavButton[accent="review"] {
                background: #272013;
                border-color: #6f5628;
                color: #f0ddb0;
            }
            QPushButton#stageNavButton[accent="publish"] {
                background: #231c30;
                border-color: #5b4480;
                color: #ddc8ff;
            }
            QPushButton#stageNavButton:checked {
                background: #2f4b80;
                border-color: #4f8cff;
                color: #ffffff;
            }
            QPushButton#stageNavButton[accent="prepare"]:checked {
                background: #1f6e61;
                border-color: #34b49a;
            }
            QPushButton#stageNavButton[accent="review"]:checked {
                background: #8b6524;
                border-color: #d7a24b;
            }
            QPushButton#stageNavButton[accent="publish"]:checked {
                background: #5f3f92;
                border-color: #9d73db;
            }
            QPushButton#substageNavButton {
                text-align: left;
                padding: 8px 12px;
                background: #1f2430;
                border-color: #2f3542;
                font-weight: 600;
            }
            QPushButton#substageNavButton[accent="review"] {
                background: #272013;
                border-color: #6f5628;
                color: #f0ddb0;
            }
            QPushButton#substageNavButton[accent="sermon"] {
                background: #1a2130;
                border-color: #4068b8;
                color: #cfe0ff;
            }
            QPushButton#substageNavButton[accent="blog"] {
                background: #241b31;
                border-color: #7951b3;
                color: #e1d0ff;
            }
            QPushButton#substageNavButton[accent="youtube"] {
                background: #2d1a20;
                border-color: #b65656;
                color: #ffd3d3;
            }
            QPushButton#substageNavButton[accent="facebook"] {
                background: #192433;
                border-color: #4d87d7;
                color: #d3e4ff;
            }
            QPushButton#substageNavButton:checked {
                background: #273246;
                border-color: #3a7afe;
                color: #ffffff;
            }
            QPushButton#substageNavButton[accent="review"]:checked {
                background: #8b6524;
                border-color: #d7a24b;
            }
            QPushButton#substageNavButton[accent="sermon"]:checked {
                background: #3f6dcf;
                border-color: #6e97f0;
            }
            QPushButton#substageNavButton[accent="blog"]:checked {
                background: #8a57c8;
                border-color: #b588ef;
            }
            QPushButton#substageNavButton[accent="youtube"]:checked {
                background: #c95656;
                border-color: #ee8d8d;
            }
            QPushButton#substageNavButton[accent="facebook"]:checked {
                background: #4d87d7;
                border-color: #85b5ff;
            }
            QPushButton#inlineToggleButton {
                border: none;
                border-radius: 0;
                padding: 2px 0;
                background: transparent;
                color: #9ebeff;
                font-weight: 600;
            }
            QPushButton#inlineToggleButton:hover {
                background: transparent;
                color: #bfd3ff;
            }
            QPushButton#inlineToggleButton:checked {
                color: #d7e3ff;
            }
            QProgressBar {
                border: 1px solid #2f3542;
                border-radius: 10px;
                text-align: center;
                background: #12151b;
                color: #f4f6fb;
                min-height: 24px;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background: #3a7afe;
                border-radius: 9px;
            }
            """
        )

    @staticmethod
    def _append_log(widget: QPlainTextEdit, text: str) -> None:
        logging.info(text)
        widget.appendPlainText(text)
        sb = widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_caption_log(self, text: str) -> None:
        self._append_log(self.caption_log, text)

    def _on_clips_log(self, text: str) -> None:
        self._append_log(self.clips_log, text)

    def _on_rank_log(self, text: str) -> None:
        self._append_log(self.rank_log, text)

    def _set_busy(self, busy: bool) -> None:
        self.caption_run_btn.setDisabled(busy)
        self.caption_auto_btn.setDisabled(busy)
        self.rank_run_btn.setDisabled(busy)
        self.extract_btn.setDisabled(busy)
        self.sermon_run_btn.setDisabled(busy)
        self.blog_run_btn.setDisabled(busy)
        self.youtube_run_btn.setDisabled(busy)
        self.facebook_run_btn.setDisabled(busy)

    def _move_source_video_if_needed(
        self,
        video_path: Path | None,
        target_dir: Path | None,
        prefix: str | None = None,
    ) -> Path | None:
        """Move source video into target asset folder after successful processing."""
        if video_path is None or target_dir is None:
            return video_path

        source = video_path.resolve()
        destination_dir = target_dir.resolve()
        desired_name = source.name
        if prefix:
            desired_name = artifact_filename("video", prefix=prefix, suffix=source.suffix or ".mp4")
        destination = destination_dir / desired_name
        if source == destination:
            return source

        destination_dir.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            logging.warning("Destination video already exists, skipping move: %s", destination)
            self.statusBar().showMessage(
                f"Skipped moving source video (already exists): {destination.name}",
                7000,
            )
            return source

        try:
            moved = Path(shutil.move(str(source), str(destination))).resolve()
            if prefix:
                session_updates = {
                    "prefix": prefix,
                    "source_video_name": moved.name,
                    "source_video_original_name": source.name,
                    "artifacts": {"video": moved.name},
                    "steps": {"caption": {"status": "saved"}},
                }
                upsert_session(destination_dir, session_updates)
                words_path = artifact_path(destination_dir, "words", session=load_session(destination_dir))
                if words_path.is_file():
                    update_words_media_file(words_path, moved.name)
            logging.info("Moved source video to %s", moved)
            self.statusBar().showMessage(f"Moved source video to: {moved}", 8000)
            return moved
        except Exception:
            logging.error("Failed moving source video: %s", traceback.format_exc())
            self.statusBar().showMessage("Could not move source video. See log.", 8000)
            return source

    def _run_worker(self, worker: QObject, on_done_slot) -> None:
        self.active_thread = QThread()
        self.active_worker = worker
        worker.moveToThread(self.active_thread)
        self.active_thread.started.connect(worker.run)
        worker.done.connect(on_done_slot)
        worker.done.connect(self.active_thread.quit)
        worker.done.connect(worker.deleteLater)
        self.active_thread.finished.connect(self.active_thread.deleteLater)
        self.active_thread.finished.connect(self._on_worker_thread_finished)
        self.active_thread.start()

    def _on_worker_thread_finished(self) -> None:
        self.active_worker = None
        self.active_thread = None
        if self._auto_run_enabled and self._auto_run_next_step:
            next_step = self._auto_run_next_step
            self._auto_run_next_step = None
            if next_step == "ranking":
                self.start_ranking()
                if self.active_thread is None:
                    self._stop_auto_run("Auto run stopped before ranking could start.")
            elif next_step == "extraction":
                self.start_extraction()
                if self.active_thread is None:
                    self._stop_auto_run("Auto run stopped before extraction could start.")

    def pick_caption_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if path:
            self.caption_video_edit.setText(path)
            self._on_caption_video_dropped(path)

    def pick_caption_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.caption_output_edit.setText(path)
            self._caption_output_overridden = True

    def _on_caption_video_dropped(self, path: str) -> None:
        self._caption_output_overridden = False
        self._refresh_caption_output_dir()

    def _refresh_caption_output_dir(self, *_args) -> None:
        if self._caption_output_overridden:
            return
        video_raw = self.caption_video_edit.text().strip()
        if not video_raw:
            return
        video_path = Path(video_raw).resolve()
        date_str = self.caption_date_edit.date().toString("yyyy-MM-dd")
        aliases = load_speaker_aliases()
        idx = self.caption_speaker_combo.currentIndex()
        if 0 <= idx < len(aliases):
            speaker_slug = speaker_to_slug(aliases[idx]["canonical"])
        else:
            speaker_slug = speaker_to_slug(self.caption_speaker_combo.currentText().replace("— Other —", "").strip() or "Speaker")
        asset_dir = get_video_asset_dir(video_path, date_str=date_str, speaker_slug=speaker_slug)
        self.caption_output_edit.setText(str(asset_dir))

    def start_captioning(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return

        video_raw = self.caption_video_edit.text().strip()
        out_raw = self.caption_output_edit.text().strip()
        language = self.language_edit.text().strip() or None

        if not video_raw:
            QMessageBox.critical(self, "Missing input", "Please select or drop a video file.")
            return

        video_path = Path(video_raw).resolve()
        if not video_path.is_file():
            QMessageBox.critical(self, "Missing file", f"Video file not found:\n{video_path}")
            return

        date_str = self.caption_date_edit.date().toString("yyyy-MM-dd")
        aliases = load_speaker_aliases()
        idx = self.caption_speaker_combo.currentIndex()
        if 0 <= idx < len(aliases):
            speaker_slug = speaker_to_slug(aliases[idx]["canonical"])
        else:
            speaker_slug = speaker_to_slug(self.caption_speaker_combo.currentText().replace("— Other —", "").strip() or "Speaker")
        prefix = asset_prefix(date_str, speaker_slug)
        asset_dir = Path(out_raw).resolve() if out_raw else get_video_asset_dir(video_path, date_str=date_str, speaker_slug=speaker_slug)
        out_base = asset_dir / prefix
        out_base.parent.mkdir(parents=True, exist_ok=True)
        self._caption_last_video = video_path
        self._caption_last_out_dir = out_base.parent
        self._caption_last_prefix = prefix
        canonical_name = artifact_filename("video", prefix=prefix, suffix=video_path.suffix or ".mp4")
        upsert_session(
            asset_dir,
            {
                "prefix": prefix,
                "date_preached": date_str,
                "source_video_name": canonical_name,
                "source_video_original_name": video_path.name,
                "speaker": {
                    "canonical": aliases[idx]["canonical"] if 0 <= idx < len(aliases) else "",
                    "prompt_name": aliases[idx]["promptName"] if 0 <= idx < len(aliases) else self.caption_speaker_combo.currentText().replace("— Other —", "").strip(),
                    "slug": speaker_slug,
                },
                "artifacts": {"video": canonical_name},
                "steps": {
                    "caption": {
                        "status": "running",
                        "generated_at": now_iso(),
                    }
                },
            },
        )
        self._set_current_asset_dir(asset_dir)

        self.caption_log.clear()
        self.caption_progress.setMaximum(100)
        self.caption_progress.setValue(0)
        self.caption_progress.setFormat("Starting...")
        self._set_busy(True)

        worker = CaptionWorker(
            video_path=video_path,
            out_base=out_base,
            model_name=self.model_combo.currentText(),
            device=self.device_combo.currentText(),
            language=language,
            write_vtt=self.vtt_check.isChecked(),
            keep_audio=self.keep_audio_check.isChecked(),
            max_chars_per_line=self.max_chars_spin.value(),
            max_lines_per_cue=self.max_lines_spin.value(),
        )
        worker.log.connect(self._on_caption_log)
        worker.progress.connect(self._on_caption_progress)
        self._run_worker(worker, self._on_caption_done)

    def _on_caption_progress(self, current: int, total: int, label: str) -> None:
        self.caption_progress.setMaximum(max(total, 1))
        self.caption_progress.setValue(min(current, max(total, 1)))
        self.caption_progress.setFormat(label)

    def _on_caption_done(self, success: bool, message: str) -> None:
        self._set_busy(False)
        asset_dir = self._caption_last_out_dir
        if success:
            self._append_log(self.caption_log, "Done.")
            self.caption_progress.setValue(self.caption_progress.maximum())
            self.caption_progress.setFormat("Completed")
            self.statusBar().showMessage("Captioning complete.", 6000)
            self._append_log(self.caption_log, message)
            moved = self._move_source_video_if_needed(
                self._caption_last_video,
                self._caption_last_out_dir,
                prefix=self._caption_last_prefix or None,
            )
            if moved is not None:
                self.caption_video_edit.setText(str(moved))
                self.clips_video_edit.setText(str(moved))
                self.caption_output_edit.setText(str(self._caption_last_out_dir))
            if asset_dir is not None:
                self._set_current_asset_dir(asset_dir)
                self._loaded_step_flags.add("caption")
            self._try_auto_load_preview_after_caption(asset_dir)
            if self._auto_run_enabled:
                self._auto_run_next_step = "ranking"
        else:
            if self._auto_run_enabled:
                self._stop_auto_run("Auto run stopped because captioning failed.")
            self._write_step_status(asset_dir, "caption", "failed", error=message)
            self._append_log(self.caption_log, f"ERROR: {message}")
            self.statusBar().showMessage("Captioning failed. See log panel.", 8000)
        self._caption_last_video = None
        self._caption_last_out_dir = None
        self._caption_last_prefix = ""

    def _try_auto_load_preview_after_caption(self, asset_dir: Path | None) -> None:
        if asset_dir is None or not asset_dir.is_dir():
            return
        self.rank_folder_edit.setText(str(asset_dir))
        self._set_current_asset_dir(asset_dir)
        self.show_stage(self.clips_page)
        self.clips_tabs.setCurrentWidget(self.rank_tab)
        self.statusBar().showMessage(
            "Captioning complete. Run Rank Moments to score clips.",
            8000,
        )

    def _clear_rank_panels(self) -> None:
        self.rank_log.clear()
        self.rank_results.clear()

    def pick_rank_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select asset folder")
        if path:
            self.load_session_into_ui(Path(path).resolve())

    def _resolve_preview_source_video(self, rank_result: dict) -> Path | None:
        if self._preview_video_path is not None and self._preview_video_path.is_file():
            return self._preview_video_path

        asset_dir_raw = str(rank_result.get("asset_dir", "")).strip()
        if not asset_dir_raw:
            return None
        asset_dir = Path(asset_dir_raw)
        if not asset_dir.is_dir():
            return None
        session = load_session(asset_dir)
        return resolve_main_video_path(asset_dir, session=session)

    def pick_preview_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source video",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if not path:
            return
        video_path = Path(path).resolve()
        self._preview_video_path = video_path
        self._set_preview_clip_window(0, 0)
        self.preview_player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.preview_player.setPosition(0)
        self.preview_header.setText(f"Loaded video: {video_path.name}")
        self.preview_reasoning_flag.setVisible(False)

        self._set_current_asset_dir(video_path.parent)
        ranked_path = artifact_path(video_path.parent, "ranked_moments", session=self._current_session)
        if ranked_path.is_file():
            if self._load_preview_report_from_path(ranked_path):
                self._loaded_step_flags.add("ranking")
                self._refresh_session_summary()
                self.statusBar().showMessage(
                    f"Loaded video and report: {ranked_path.name}",
                    6000,
                )
            return

        self._preview_rank_result = None
        self.preview_clips_list.clear()
        self.preview_details.clear()
        self._clear_preview_score_cards()
        self.preview_reasoning_flag.setVisible(False)
        QMessageBox.information(
            self,
            "Missing ranked report",
            "No saved ranked report was found next to this video.\n\n"
            "Run Rank Moments first, then return to Preview Report.",
        )
        self.statusBar().showMessage(
            "No ranked report found beside selected video.",
            7000,
        )

    def _load_preview_report_from_path(self, ranked_path: Path) -> bool:
        try:
            rank_result = json.loads(ranked_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Invalid ranked report", f"Could not parse ranked_moments.json:\n{exc}")
            return False

        clips = rank_result.get("clips", [])
        if not isinstance(clips, list) or not clips:
            QMessageBox.critical(self, "Missing clips", "Ranked report does not contain any clips.")
            return False

        self._preview_rank_result = rank_result
        asset_dir_raw = str(rank_result.get("asset_dir", "")).strip()
        if asset_dir_raw:
            self._set_current_asset_dir(Path(asset_dir_raw))
        self._preview_suppress_autoplay = True
        self._populate_preview_clips(clips)
        self._preview_suppress_autoplay = False
        self._set_preview_header_for_result(rank_result)
        return True

    @staticmethod
    def _friendly_model_name(model_name: str) -> str:
        value = model_name.strip()
        if not value:
            return "(unknown model)"
        if "/inference-profile/" in value:
            value = value.rsplit("/inference-profile/", 1)[-1]
        if value.startswith("arn:"):
            value = value.rsplit("/", 1)[-1]
        return value

    @staticmethod
    def _phase_balance_text(rank_result: dict) -> str:
        diag = (
            rank_result.get("phase_balance_diagnostics", {})
            if isinstance(rank_result.get("phase_balance_diagnostics", {}), dict)
            else {}
        )
        counts = diag.get("counts", {}) if isinstance(diag.get("counts", {}), dict) else {}
        if not counts:
            return "phase: n/a"
        return (
            f"phase E/M/L {int(counts.get('early', 0))}/"
            f"{int(counts.get('mid', 0))}/{int(counts.get('late', 0))}"
        )

    @staticmethod
    def _feedback_file_path() -> Path:
        return Path.home() / ".fastcap" / "clip_feedback.jsonl"

    @staticmethod
    def _clip_feedback_key(clip: dict, rank_result: dict | None) -> str:
        asset_dir = ""
        if isinstance(rank_result, dict):
            asset_dir = str(rank_result.get("asset_dir", "")).strip()
        return "|".join(
            [
                asset_dir,
                str(int(clip.get("candidate_id", 0) or 0)),
                str(clip.get("start_time", "")).strip(),
                str(clip.get("end_time", "")).strip(),
            ]
        )

    @staticmethod
    def _feedback_row_key(row: dict) -> str:
        return "|".join(
            [
                str(row.get("asset_dir", "")).strip(),
                str(int(row.get("candidate_id", 0) or 0)),
                str(row.get("start_time", "")).strip(),
                str(row.get("end_time", "")).strip(),
            ]
        )

    def _upsert_feedback_payload(self, feedback_key: str, payload: dict) -> bool:
        feedback_path = self._feedback_file_path()
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        prior_exists = False
        temp_path = Path(
            tempfile.mkstemp(prefix="clip_feedback_", suffix=".jsonl.tmp", dir=str(feedback_path.parent))[1]
        )
        try:
            with temp_path.open("w", encoding="utf-8") as out_handle:
                if feedback_path.is_file():
                    with feedback_path.open("r", encoding="utf-8") as in_handle:
                        for raw in in_handle:
                            line = raw.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:  # noqa: BLE001
                                out_handle.write(raw if raw.endswith("\n") else (raw + "\n"))
                                continue
                            if isinstance(row, dict) and self._feedback_row_key(row) == feedback_key:
                                prior_exists = True
                                continue
                            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            try:
                temp_path.replace(feedback_path)
            except OSError:
                import shutil
                shutil.copy2(str(temp_path), str(feedback_path))
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        except OSError:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
        return not prior_exists

    def _refresh_feedback_cache(self) -> None:
        feedback_path = self._feedback_file_path()
        self._feedback_total_count = 0
        self._feedback_persisted_keys = set()
        self._feedback_labels = {}
        if not feedback_path.is_file():
            return
        try:
            with feedback_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if isinstance(row, dict):
                        key = self._feedback_row_key(row)
                        if key.strip("|"):
                            self._feedback_persisted_keys.add(key)
                            label = str(row.get("label", "")).strip().lower()
                            if label in {"up", "down"}:
                                self._feedback_labels[key] = label
        except OSError:
            return
        self._feedback_total_count = len(self._feedback_persisted_keys)

    _FEEDBACK_BTN_ACTIVE = (
        "QPushButton { background: #2a6e3f; border-color: #3aaf5c; color: #ffffff; font-weight: 700; }"
    )
    _FEEDBACK_BTN_ACTIVE_DOWN = (
        "QPushButton { background: #8b2020; border-color: #c44040; color: #ffffff; font-weight: 700; }"
    )
    _FEEDBACK_BTN_NORMAL = ""

    def _update_feedback_button_state(self, clip: dict) -> None:
        key = self._clip_feedback_key(clip, self._preview_rank_result)
        label = self._feedback_labels.get(key, "")
        if label == "up":
            self.preview_keep_btn.setStyleSheet(self._FEEDBACK_BTN_ACTIVE)
            self.preview_skip_btn.setStyleSheet(self._FEEDBACK_BTN_NORMAL)
        elif label == "down":
            self.preview_keep_btn.setStyleSheet(self._FEEDBACK_BTN_NORMAL)
            self.preview_skip_btn.setStyleSheet(self._FEEDBACK_BTN_ACTIVE_DOWN)
        else:
            self.preview_keep_btn.setStyleSheet(self._FEEDBACK_BTN_NORMAL)
            self.preview_skip_btn.setStyleSheet(self._FEEDBACK_BTN_NORMAL)

    def _feedback_progress_text(self) -> str:
        target = 50
        count = int(self._feedback_total_count)
        return f"{count}/{target} reviews toward next calibration"

    def _set_preview_header_for_result(self, rank_result: dict) -> None:
        clips = list(rank_result.get("clips", []) or [])
        self._refresh_feedback_cache()
        model_name = self._friendly_model_name(str(rank_result.get("model", "")).strip())
        self.preview_header.setText(
            f"Loaded report for {len(clips)} clips | model: {model_name} | "
            f"{self._phase_balance_text(rank_result)} | {self._feedback_progress_text()}"
        )

    @staticmethod
    def _format_millis(ms: int) -> str:
        sec = max(0, int(ms // 1000))
        hours = sec // 3600
        minutes = (sec % 3600) // 60
        seconds = sec % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _set_preview_clip_window(self, start_ms: int, end_ms: int) -> None:
        self._preview_clip_start_ms = max(0, int(start_ms))
        self._preview_clip_end_ms = max(self._preview_clip_start_ms, int(end_ms))
        window_ms = max(0, self._preview_clip_end_ms - self._preview_clip_start_ms)
        self.preview_seek_slider.setEnabled(window_ms > 0)
        self.preview_seek_slider.setRange(0, window_ms)
        self.preview_seek_slider.blockSignals(True)
        self.preview_seek_slider.setValue(0)
        self.preview_seek_slider.blockSignals(False)
        self.preview_seek_current.setText("00:00")
        self.preview_seek_total.setText(self._format_millis(window_ms))

    def _on_preview_seek_pressed(self) -> None:
        self._preview_is_scrubbing = True

    def _on_preview_seek_released(self) -> None:
        if self._preview_clip_end_ms <= self._preview_clip_start_ms:
            self._preview_is_scrubbing = False
            return
        target = self._preview_clip_start_ms + int(self.preview_seek_slider.value())
        target = max(self._preview_clip_start_ms, min(self._preview_clip_end_ms, target))
        self.preview_player.setPosition(target)
        self._preview_is_scrubbing = False

    def _on_preview_seek_value_changed(self, value: int) -> None:
        if self._preview_clip_end_ms <= self._preview_clip_start_ms:
            return
        current_ms = max(0, min(self._preview_clip_end_ms - self._preview_clip_start_ms, int(value)))
        self.preview_seek_current.setText(self._format_millis(current_ms))

    @staticmethod
    def _editorial_scores_from_clip(clip: dict) -> dict[str, int]:
        editorial = clip.get("editorial_scores", {}) if isinstance(clip.get("editorial_scores", {}), dict) else {}
        sub = clip.get("sub_scores", {}) if isinstance(clip.get("sub_scores", {}), dict) else {}
        return {
            "editor": int(editorial.get("editor", clip.get("editor_score", 0))),
            "hook": int(editorial.get("hook", sub.get("hook", 0))),
            "cadence": int(editorial.get("cadence", sub.get("cadence", 0))),
            "standalone": int(editorial.get("standalone", sub.get("standalone", 0))),
            "emotion": int(editorial.get("emotion", sub.get("emotion", 0))),
        }

    def _reasoning_with_consistency_guard(self, clip: dict) -> tuple[str, bool]:
        reason = str(clip.get("editor_reason", "")).strip()
        if not reason:
            return "", False
        consistency = (
            clip.get("reasoning_consistency", {})
            if isinstance(clip.get("reasoning_consistency", {}), dict)
            else {}
        )
        if bool(consistency.get("normalized", False)):
            return reason, True

        editorial = self._editorial_scores_from_clip(clip)
        feature = clip.get("feature_scores", {}) if isinstance(clip.get("feature_scores", {}), dict) else {}
        expected = {
            "editor": int(editorial.get("editor", 0)),
            "hook": int(editorial.get("hook", 0)),
            "cadence": int(editorial.get("cadence", 0)),
            "standalone": int(editorial.get("standalone", 0)),
            "emotion": int(editorial.get("emotion", 0)),
            "energy": int(feature.get("energy", editorial.get("emotion", 0))),
            "contrast": int(feature.get("contrast", editorial.get("editor", 0))),
            "overall": int(feature.get("overall_candidate", editorial.get("editor", 0))),
        }
        aliases = {
            "editor": "editor",
            "editor score": "editor",
            "overall": "overall",
            "overall candidate": "overall",
            "hook": "hook",
            "cadence": "cadence",
            "standalone": "standalone",
            "emotion": "emotion",
            "energy": "energy",
            "contrast": "contrast",
        }

        mismatch = False

        def _metric_repl(match: re.Match) -> str:
            nonlocal mismatch
            label = str(match.group(1)).strip()
            key = aliases.get(label.lower())
            actual = int(match.group(2))
            want = expected.get(key)
            if want is None or actual == want:
                return match.group(0)
            mismatch = True
            if match.group(0).find("(") >= 0:
                return f"{label} score ({want})"
            if "score" in match.group(0).lower():
                return f"{label} score of {want}"
            return f"{label} {want}"

        reason = re.sub(
            r"\b(Editor|Editor score|Overall candidate|Overall|Hook|Cadence|Standalone|Emotion|Energy|Contrast)\b"
            r"(?:\s+score)?(?:\s+of|\s*[:=]|\s+is)?\s*(?:\(\s*)?(\d{1,3})(?:\s*\))?",
            _metric_repl,
            reason,
            flags=re.IGNORECASE,
        )

        def _overall_paren_repl(match: re.Match) -> str:
            nonlocal mismatch
            actual = int(match.group(1))
            want = expected["overall"]
            if actual == want:
                return match.group(0)
            mismatch = True
            return f"overall candidate score ({want})"

        reason = re.sub(
            r"overall candidate score\s*\((\d+)\)",
            _overall_paren_repl,
            reason,
            flags=re.IGNORECASE,
        )

        return reason, mismatch

    @staticmethod
    def _score_tier(value: int) -> tuple[str, str]:
        if value >= 85:
            return "#2f9e44", "#14321e"
        if value >= 70:
            return "#f59f00", "#3a2a10"
        return "#e03131", "#3a1517"

    def _set_preview_score_card(self, key: str, value: int) -> None:
        widget = self.preview_score_widgets.get(key)
        if not widget:
            return
        value = max(0, min(DISPLAY_SCORE_MAX, int(value)))
        fg, bg = self._score_tier(value)
        value_label = widget["value_label"]
        bar = widget["bar"]
        frame = widget["frame"]
        if isinstance(value_label, QLabel):
            value_label.setText(str(value))
            value_label.setStyleSheet(f"font-weight: 700; color: {fg};")
        if isinstance(bar, QProgressBar):
            bar.setValue(value)
            bar.setStyleSheet(
                "QProgressBar { border: 1px solid #2f3542; border-radius: 5px; background: #12151b; }"
                f"QProgressBar::chunk {{ background: {fg}; border-radius: 4px; }}"
            )
        if isinstance(frame, QFrame):
            frame.setStyleSheet(
                "QFrame#scoreCard { border: 1px solid #384053; border-radius: 8px; "
                f"background: {bg}; }}"
            )

    def _clear_preview_score_cards(self) -> None:
        for key in ("editor", "hook", "cadence", "standalone", "emotion"):
            widget = self.preview_score_widgets.get(key)
            if not widget:
                continue
            value_label = widget["value_label"]
            bar = widget["bar"]
            frame = widget["frame"]
            if isinstance(value_label, QLabel):
                value_label.setText("--")
                value_label.setStyleSheet("font-weight: 700; color: #f0f3fa;")
            if isinstance(bar, QProgressBar):
                bar.setValue(0)
                bar.setStyleSheet(
                    "QProgressBar { border: 1px solid #2f3542; border-radius: 5px; background: #12151b; }"
                    "QProgressBar::chunk { background: #3a7afe; border-radius: 4px; }"
                )
            if isinstance(frame, QFrame):
                frame.setStyleSheet(
                    "QFrame#scoreCard { border: 1px solid #384053; border-radius: 8px; background: #171b23; }"
                )

    def _update_preview_score_cards(self, clip: dict) -> None:
        editorial = self._editorial_scores_from_clip(clip)
        self._set_preview_score_card("editor", int(editorial.get("editor", 0)))
        self._set_preview_score_card("hook", int(editorial.get("hook", 0)))
        self._set_preview_score_card("cadence", int(editorial.get("cadence", 0)))
        self._set_preview_score_card("standalone", int(editorial.get("standalone", 0)))
        self._set_preview_score_card("emotion", int(editorial.get("emotion", 0)))

    def _render_clip_report_card(self, clip: dict) -> str:
        reason_text, _ = self._reasoning_with_consistency_guard(clip)
        editorial = self._editorial_scores_from_clip(clip)
        feature = clip.get("feature_scores", {}) if isinstance(clip.get("feature_scores", {}), dict) else {}
        signals = clip.get("feature_signals", {}) if isinstance(clip.get("feature_signals", {}), dict) else {}
        confidence = (
            clip.get("assessment_confidence", {})
            if isinstance(clip.get("assessment_confidence", {}), dict)
            else {}
        )
        evidence = clip.get("reason_evidence", {}) if isinstance(clip.get("reason_evidence", {}), dict) else {}
        diag = clip.get("selection_diagnostics", {}) if isinstance(clip.get("selection_diagnostics", {}), dict) else {}
        consistency = (
            clip.get("reasoning_consistency", {})
            if isinstance(clip.get("reasoning_consistency", {}), dict)
            else {}
        )
        personal_fit_applied = bool(clip.get("personal_fit_applied", False))
        personal_fit_score = int(clip.get("personal_fit_score", 0) or 0)
        final_rank_score = int(clip.get("final_rank_score", clip.get("editor_score", 0)) or 0)
        profile_version = int(clip.get("preference_profile_version", 0) or 0)
        editorial_lines = [
            "Editorial scores (LLM-facing):",
            "  "
            f"Editor {int(editorial.get('editor', 0))}, "
            f"Hook {int(editorial.get('hook', 0))}, "
            f"Cadence {int(editorial.get('cadence', 0))}, "
            f"Standalone {int(editorial.get('standalone', 0))}, "
            f"Emotion {int(editorial.get('emotion', 0))}",
        ]
        feature_lines = []
        if feature:
            feature_lines.append("Feature scores (deterministic):")
            feature_lines.append(
                "  "
                f"Overall {int(feature.get('overall_candidate', 0))}, "
                f"Cadence {int(feature.get('cadence', 0))}, "
                f"Energy {int(feature.get('energy', 0))}, "
                f"Contrast {int(feature.get('contrast', 0))}"
            )
            markers = signals.get("markers", [])
            reasons = signals.get("reasons", [])
            if isinstance(markers, list) and markers:
                feature_lines.append("  Markers: " + ", ".join(str(x) for x in markers))
            if isinstance(reasons, list) and reasons:
                feature_lines.append("  Reasons: " + ", ".join(str(x) for x in reasons))
        else:
            feature_lines.append("Feature scores (deterministic): (not present in this ranked file)")

        confidence_line = (
            f"Assessment confidence: {str(confidence.get('level', 'n/a')).capitalize()} "
            f"({int(confidence.get('score', 0))}/100)"
        )
        if personal_fit_applied:
            personalization_line = (
                f"Personalization: applied (profile v{profile_version}) | "
                f"personal fit {personal_fit_score}/100 | final rank {final_rank_score}/{DISPLAY_SCORE_MAX}"
            )
        else:
            personalization_line = "Personalization: not applied for this ranking run"
        evidence_lines = []
        cited = evidence.get("cited_metrics", [])
        if isinstance(cited, list) and cited:
            evidence_lines.append("Reason evidence metrics: " + ", ".join(str(x) for x in cited))
        phase = str(diag.get("phase", "")).strip()
        if phase:
            evidence_lines.append(
                f"Selection diagnostics: phase={phase}, "
                f"overlap_prev={float(diag.get('overlap_prev_ratio', 0.0)):.2f}, "
                f"overlap_next={float(diag.get('overlap_next_ratio', 0.0)):.2f}"
            )
        if consistency:
            evidence_lines.append(
                f"Reasoning consistency: severity={consistency.get('severity', 'none')}, "
                f"minor={int(consistency.get('minor_count', 0))}, severe={int(consistency.get('severe_count', 0))}"
            )

        return "\n".join(
            [
                f"Clip #{int(clip.get('clip_number', 0))}",
                f"Window: {clip.get('start_time', '')} - {clip.get('end_time', '')} "
                f"({float(clip.get('duration_sec', 0.0)):.1f}s)",
                f"Type: {clip.get('clip_type', '')}",
                f"Cadence marker: {clip.get('cadence_marker', '')}",
                f"Scroll-stopping: {clip.get('scroll_stopping_strength', '')}",
                f"Best platform: {clip.get('best_platform_fit', '')}",
                "",
                *editorial_lines,
                "",
                *feature_lines,
                "",
                confidence_line,
                personalization_line,
                *evidence_lines,
                "",
                f"Opening hook:\n{clip.get('opening_hook', '')}",
                "",
                f"Editor reason:\n{reason_text}",
            ]
        )

    def _populate_preview_clips(self, clips: list[dict]) -> None:
        self.preview_clips_list.clear()
        reviewed_keys = self._feedback_persisted_keys | self._feedback_session_keys
        for clip in clips:
            num = int(clip.get("clip_number", 0))
            start = str(clip.get("start_time", ""))
            end = str(clip.get("end_time", ""))
            score = int(clip.get("editor_score", 0))
            ctype = str(clip.get("clip_type", "")).strip()
            base_text = f"{num:02d} | {start} - {end} | score {score} | {ctype}"
            key = self._clip_feedback_key(clip, self._preview_rank_result)
            label = f"✓ {base_text}" if key in reviewed_keys else base_text
            self.preview_clips_list.addItem(label)
        if clips:
            self.preview_clips_list.setCurrentRow(0)
        else:
            self.preview_details.clear()
            self._clear_preview_score_cards()
            self.preview_reasoning_flag.setVisible(False)
            self._set_preview_clip_window(0, 0)
            self.stop_preview_playback()

    def _on_preview_clip_selected(self, row: int) -> None:
        clips = list(self._preview_rank_result.get("clips", []) or []) if isinstance(self._preview_rank_result, dict) else []
        if row < 0 or row >= len(clips):
            self.preview_details.clear()
            self._clear_preview_score_cards()
            self.preview_reasoning_flag.setVisible(False)
            return
        self._update_preview_score_cards(clips[row])
        self._update_feedback_button_state(clips[row])
        _, mismatch = self._reasoning_with_consistency_guard(clips[row])
        consistency = (
            clips[row].get("reasoning_consistency", {})
            if isinstance(clips[row].get("reasoning_consistency", {}), dict)
            else {}
        )
        if mismatch:
            self.preview_reasoning_flag.setText(
                "Reasoning mismatch normalized in ranking output to match persisted scores."
            )
            sev = str(consistency.get("severity", "none")).lower()
            if sev == "severe":
                self.preview_reasoning_flag.setText(
                    "Severe reasoning conflicts were detected; output was normalized and flagged."
                )
            self.preview_reasoning_flag.setVisible(True)
        else:
            self.preview_reasoning_flag.setVisible(False)
        self.preview_details.setPlainText(self._render_clip_report_card(clips[row]))
        if not self._preview_suppress_autoplay:
            self._play_preview_clip(row)

    def _record_clip_feedback(self, label: str) -> None:
        if label not in {"up", "down"}:
            return
        if not isinstance(self._preview_rank_result, dict):
            QMessageBox.information(self, "No report loaded", "Load a ranked report before saving feedback.")
            return
        clips = list(self._preview_rank_result.get("clips", []) or [])
        row = int(self.preview_clips_list.currentRow())
        if row < 0 or row >= len(clips):
            QMessageBox.information(self, "No clip selected", "Select a clip first.")
            return
        clip = clips[row]
        feedback_key = self._clip_feedback_key(clip, self._preview_rank_result)
        note = self.preview_feedback_note.text().strip()
        source_video = str(self._preview_video_path) if self._preview_video_path else ""
        asset_dir = str(self._preview_rank_result.get("asset_dir", "")).strip()
        editorial_scores = (
            clip.get("editorial_scores", {})
            if isinstance(clip.get("editorial_scores", {}), dict)
            else {}
        )
        feature_scores = (
            clip.get("feature_scores", {})
            if isinstance(clip.get("feature_scores", {}), dict)
            else {}
        )
        feature_signals = (
            clip.get("feature_signals", {})
            if isinstance(clip.get("feature_signals", {}), dict)
            else {}
        )
        assessment_confidence = (
            clip.get("assessment_confidence", {})
            if isinstance(clip.get("assessment_confidence", {}), dict)
            else {}
        )
        payload = {
            "feedback_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "label": label,
            "note": note,
            "source_video": source_video,
            "asset_dir": asset_dir,
            "clip_number": int(clip.get("clip_number", 0) or 0),
            "candidate_id": int(clip.get("candidate_id", 0) or 0),
            "start_time": str(clip.get("start_time", "")).strip(),
            "end_time": str(clip.get("end_time", "")).strip(),
            "duration_sec": float(clip.get("duration_sec", 0.0) or 0.0),
            "opening_hook": str(clip.get("opening_hook", "")).strip(),
            "editor_score": int(clip.get("editor_score", 0) or 0),
            "editorial_scores": editorial_scores,
            "feature_scores": feature_scores,
            "feature_signals": feature_signals,
            "clip_type": str(clip.get("clip_type", "")).strip(),
            "cadence_marker": str(clip.get("cadence_marker", "")).strip(),
            "scroll_stopping_strength": str(clip.get("scroll_stopping_strength", "")).strip(),
            "best_platform_fit": str(clip.get("best_platform_fit", "")).strip(),
            "assessment_confidence": assessment_confidence,
            "model": str(self._preview_rank_result.get("model", "")).strip(),
        }
        try:
            is_new_feedback = self._upsert_feedback_payload(feedback_key, payload)
        except OSError as exc:
            QMessageBox.critical(self, "Feedback save failed", f"Could not save feedback:\n{exc}")
            return

        self._feedback_session_keys.add(feedback_key)
        self._feedback_persisted_keys.add(feedback_key)
        self._feedback_labels[feedback_key] = label
        if is_new_feedback:
            self._feedback_total_count = int(self._feedback_total_count) + 1
        self.preview_feedback_note.clear()
        item = self.preview_clips_list.item(row)
        if item is not None and not item.text().startswith("✓ "):
            item.setText(f"✓ {item.text()}")
        self._update_feedback_button_state(clip)
        self._set_preview_header_for_result(self._preview_rank_result)
        self._sync_extraction_payload_from_mode()
        self.statusBar().showMessage("Feedback saved." if is_new_feedback else "Feedback updated.", 4000)

    def _ranked_moments_path(self) -> Path | None:
        if not isinstance(self._preview_rank_result, dict):
            return None
        asset_dir_raw = str(self._preview_rank_result.get("asset_dir", "")).strip()
        if not asset_dir_raw:
            return None
        asset_dir = Path(asset_dir_raw)
        p = artifact_path(asset_dir, "ranked_moments", session=load_session(asset_dir))
        return p if p.is_file() else None

    def _build_extraction_payload_from_ranked(self, kept_only: bool = False) -> dict[str, list[dict]]:
        if not isinstance(self._preview_rank_result, dict):
            raise ValueError("Load a ranked report before populating extraction.")
        clips = list(self._preview_rank_result.get("clips", []) or [])
        payload_clips: list[dict] = []
        for clip in clips:
            start_time = str(clip.get("start_time", "")).strip()
            end_time = str(clip.get("end_time", "")).strip()
            if not start_time or not end_time:
                continue
            if kept_only:
                key = self._clip_feedback_key(clip, self._preview_rank_result)
                if self._feedback_labels.get(key, "") != "up":
                    continue
            payload_clips.append(
                {
                    "clip_number": int(clip.get("clip_number", len(payload_clips) + 1) or len(payload_clips) + 1),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
        if not payload_clips:
            label = "kept" if kept_only else "ranked"
            raise ValueError(f"No {label} clips are available for extraction.")
        return {"clips": payload_clips}

    def _load_extraction_from_ranked(self, kept_only: bool = False) -> None:
        self.clips_source_combo.blockSignals(True)
        self.clips_source_combo.setCurrentIndex(1 if kept_only else 0)
        self.clips_source_combo.blockSignals(False)
        try:
            payload = self._build_extraction_payload_from_ranked(kept_only=kept_only)
        except ValueError as exc:
            QMessageBox.information(self, "Nothing to load", str(exc))
            return
        self.clips_json_edit.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self.clips_tabs.setCurrentWidget(self.clips_tab)
        scope = "kept clips" if kept_only else "ranked clips"
        self.statusBar().showMessage(f"Loaded {scope} into Extract Clips.", 5000)

    def _persist_ranked_moments(self) -> bool:
        if not isinstance(self._preview_rank_result, dict):
            return False
        path = self._ranked_moments_path()
        if path is None:
            asset_dir_raw = str(self._preview_rank_result.get("asset_dir", "")).strip()
            if asset_dir_raw:
                asset_dir = Path(asset_dir_raw)
                path = artifact_path(asset_dir, "ranked_moments", session=load_session(asset_dir))
            else:
                return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._preview_rank_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            upsert_session(
                path.parent,
                {
                    "artifacts": {"ranked_moments": path.name},
                    "steps": {"ranking": {"status": "saved", "generated_at": now_iso()}},
                },
            )
            return True
        except OSError:
            return False

    def _extend_clip_boundary(self, edge: str, seconds: float) -> None:
        if not isinstance(self._preview_rank_result, dict):
            QMessageBox.information(self, "No report", "Load a ranked report first.")
            return
        clips = list(self._preview_rank_result.get("clips", []) or [])
        row = int(self.preview_clips_list.currentRow())
        if row < 0 or row >= len(clips):
            QMessageBox.information(self, "No clip selected", "Select a clip first.")
            return
        clip = clips[row]
        start_sec = parse_timestamp_to_seconds(str(clip.get("start_time", "00:00:00.000")))
        end_sec = parse_timestamp_to_seconds(str(clip.get("end_time", "00:00:00.000")))
        jump_target_sec = start_sec
        if edge == "start":
            start_sec = max(0.0, start_sec - seconds)
            jump_target_sec = start_sec
        elif edge == "end":
            jump_target_sec = end_sec
            end_sec = end_sec + seconds
        else:
            return
        clip["start_time"] = format_seconds_to_timestamp(start_sec)
        clip["end_time"] = format_seconds_to_timestamp(end_sec)
        clip["duration_sec"] = round(end_sec - start_sec, 3)
        if self._persist_ranked_moments():
            self.statusBar().showMessage(
                f"Clip {int(clip.get('clip_number', 0))}: {edge} extended by {seconds:.0f}s and saved.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                f"Clip {int(clip.get('clip_number', 0))}: {edge} extended (could not save to file).",
                5000,
            )
        self._update_preview_score_cards(clip)
        self.preview_details.setPlainText(self._render_clip_report_card(clip))
        self._populate_preview_clips(clips)
        self.preview_clips_list.setCurrentRow(row)
        self._sync_extraction_payload_from_mode()
        self._play_preview_clip(row)
        jump_ms = int(round(jump_target_sec * 1000.0))
        jump_ms = max(self._preview_clip_start_ms, min(self._preview_clip_end_ms, jump_ms))
        self.preview_player.setPosition(jump_ms)

    @staticmethod
    def _clip_export_filename(clip: dict) -> str:
        num = int(clip.get("clip_number", 0) or 0)
        hook = str(clip.get("opening_hook", "")).strip()
        words = [w for w in hook.split() if w][:5]
        slug = "_".join(
            "".join(ch for ch in w if ch.isalnum() or ch in "-_") for w in words
        )
        if not slug:
            slug = "clip"
        return f"{num:02d}-{slug}"

    def _export_current_clip(self) -> None:
        if not isinstance(self._preview_rank_result, dict):
            QMessageBox.information(self, "No report", "Load a ranked report first.")
            return
        clips = list(self._preview_rank_result.get("clips", []) or [])
        row = int(self.preview_clips_list.currentRow())
        if row < 0 or row >= len(clips):
            QMessageBox.information(self, "No clip selected", "Select a clip first.")
            return
        clip = clips[row]
        video_path = self._resolve_preview_source_video(self._preview_rank_result)
        if video_path is None or not video_path.is_file():
            QMessageBox.critical(self, "Missing video", "Source video not found.")
            return
        start_time = str(clip.get("start_time", "")).strip()
        end_time = str(clip.get("end_time", "")).strip()
        if not start_time or not end_time:
            QMessageBox.critical(self, "Missing timestamps", "Clip has no start/end time.")
            return
        start_sec = max(0.0, parse_timestamp_to_seconds(start_time) - PREVIEW_PAD_SECONDS)
        end_sec = parse_timestamp_to_seconds(end_time) + PREVIEW_PAD_SECONDS
        duration_ms = int(self.preview_player.duration())
        if duration_ms > 0:
            end_sec = min(end_sec, duration_ms / 1000.0)
        duration = max(0.001, end_sec - start_sec)
        export_start_time = format_seconds_to_timestamp(start_sec)
        base_name = self._clip_export_filename(clip)
        ext = video_path.suffix or ".mp4"
        asset_dir_raw = str(self._preview_rank_result.get("asset_dir", "")).strip()
        default_dir = Path(asset_dir_raw) if asset_dir_raw else video_path.parent
        default_path = str(default_dir / f"{base_name}{ext}")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Clip",
            default_path,
            f"Video files (*{ext});;All files (*.*)",
        )
        if not save_path:
            return
        try:
            extract_clip(
                video_path=video_path,
                output_path=Path(save_path),
                start_time=export_start_time,
                duration_seconds=duration,
            )
            self.statusBar().showMessage(f"Exported: {Path(save_path).name}", 6000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"ffmpeg error:\n{exc}")

    def _play_preview_clip(self, row: int) -> None:
        if not isinstance(self._preview_rank_result, dict):
            return
        clips = list(self._preview_rank_result.get("clips", []) or [])
        if row < 0 or row >= len(clips):
            return
        clip = clips[row]
        video_path = self._resolve_preview_source_video(self._preview_rank_result)
        if video_path is None or not video_path.is_file():
            self.statusBar().showMessage("Preview source video not found in asset folder.", 7000)
            return

        start_time = str(clip.get("start_time", "")).strip()
        end_time = str(clip.get("end_time", "")).strip()
        if not start_time or not end_time:
            return

        start_ms = int(round((parse_timestamp_to_seconds(start_time) - PREVIEW_PAD_SECONDS) * 1000.0))
        end_ms = int(round((parse_timestamp_to_seconds(end_time) + PREVIEW_PAD_SECONDS) * 1000.0))
        start_ms = max(0, start_ms)
        duration_ms = int(self.preview_player.duration())
        if duration_ms > 0:
            end_ms = min(duration_ms, end_ms)
        if end_ms <= start_ms:
            return
        self._set_preview_clip_window(start_ms, end_ms)

        if self._preview_video_path != video_path:
            self._preview_video_path = video_path
            self.preview_player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.preview_player.setPosition(start_ms)
        self.preview_player.play()
        self.statusBar().showMessage(f"Previewing clip {int(clip.get('clip_number', 0))}", 5000)

    def _on_preview_position_changed(self, position_ms: int) -> None:
        if self._preview_clip_end_ms <= 0:
            return
        if not self._preview_is_scrubbing:
            offset_ms = max(0, min(self._preview_clip_end_ms - self._preview_clip_start_ms, position_ms - self._preview_clip_start_ms))
            self.preview_seek_slider.blockSignals(True)
            self.preview_seek_slider.setValue(offset_ms)
            self.preview_seek_slider.blockSignals(False)
            self.preview_seek_current.setText(self._format_millis(offset_ms))
        if position_ms >= self._preview_clip_end_ms:
            self.preview_player.pause()
            self.preview_player.setPosition(self._preview_clip_end_ms)

    def toggle_preview_playback(self) -> None:
        if self.preview_player.source().isEmpty():
            row = int(self.preview_clips_list.currentRow())
            self._play_preview_clip(row)
            return
        if self.preview_player.playbackState() == QMediaPlayer.PlayingState:
            self.preview_player.pause()
        else:
            self.preview_player.play()

    def stop_preview_playback(self) -> None:
        self.preview_player.stop()

    def start_ranking(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return

        folder_raw = self.rank_folder_edit.text().strip()
        model_name = self.rank_model_combo.currentText().strip()
        candidate_count = int(self.rank_candidate_spin.value())
        output_count = int(self.rank_output_spin.value())

        if not folder_raw:
            QMessageBox.critical(self, "Missing input", "Please select the asset folder.")
            return

        asset_dir = Path(folder_raw).resolve()
        if not asset_dir.is_dir():
            QMessageBox.critical(self, "Missing folder", f"Asset folder not found:\n{asset_dir}")
            return

        missing_files = [
            name
            for name in REQUIRED_ANALYSIS_FILES
            if not _asset_file_path(asset_dir, name).is_file()
        ]
        if missing_files:
            QMessageBox.critical(
                self,
                "Missing analysis files",
                "This folder is missing required files:\n"
                + "\n".join(missing_files)
                + "\n\nRun Caption Video first for this sermon.",
            )
            return

        self._rank_last_dir = asset_dir
        self._set_current_asset_dir(asset_dir)
        self.rank_log.clear()
        self.rank_results.clear()
        self.rank_progress.setMaximum(100)
        self.rank_progress.setValue(0)
        self.rank_progress.setFormat("Starting...")
        self._write_step_status(
            asset_dir,
            "ranking",
            "running",
            model=model_name,
            candidate_count=candidate_count,
            output_count=output_count,
        )
        self._set_busy(True)

        worker = RankWorker(
            asset_dir=asset_dir,
            model_name=model_name,
            candidate_count=candidate_count,
            output_count=output_count,
        )
        worker.log.connect(self._on_rank_log)
        worker.progress.connect(self._on_rank_progress)
        self._run_worker(worker, self._on_rank_done)

    def _on_rank_progress(self, current: int, total: int, label: str) -> None:
        self.rank_progress.setMaximum(max(total, 1))
        self.rank_progress.setValue(min(current, max(total, 1)))
        self.rank_progress.setFormat(label)

    def _on_rank_done(self, success: bool, message: str, result: object) -> None:
        self._set_busy(False)
        if success:
            self._append_log(self.rank_log, "Done.")
            self.rank_progress.setValue(self.rank_progress.maximum())
            self.rank_progress.setFormat("Completed")
            self.statusBar().showMessage("Ranking complete.", 6000)
            self._append_log(self.rank_log, message)
            if isinstance(result, dict):
                self.rank_results.setPlainText(format_ranked_clips_text(result))
                self._preview_rank_result = result
                clips = list(result.get("clips", []) or [])
                self._preview_suppress_autoplay = True
                self._populate_preview_clips(clips)
                self._preview_suppress_autoplay = False
                self._set_preview_header_for_result(result)
                self._sync_extraction_payload_from_mode()
                self._loaded_step_flags.add("ranking")
                if self._rank_last_dir is not None:
                    self._set_current_asset_dir(self._rank_last_dir)
                self.show_stage(self.clips_page)
                self.clips_tabs.setCurrentWidget(self.preview_tab)
                if self._auto_run_enabled:
                    self._auto_run_next_step = "extraction"
        else:
            if self._auto_run_enabled:
                self._stop_auto_run("Auto run stopped because ranking failed.")
            self._write_step_status(self._rank_last_dir, "ranking", "failed", error=message)
            self._append_log(self.rank_log, f"ERROR: {message}")
            self.statusBar().showMessage("Ranking failed. See log panel.", 8000)
        self._rank_last_dir = None
        self._refresh_session_summary()

    def pick_clips_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;All files (*.*)",
        )
        if path:
            self.clips_video_edit.setText(path)
            self._on_clips_video_dropped(path)

    def pick_clips_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.clips_output_edit.setText(path)
            self._clips_output_overridden = True

    def _on_clips_video_dropped(self, path: str) -> None:
        self._clips_output_overridden = False
        self._refresh_clips_output_dir()

    def _refresh_clips_output_dir(self, *_args) -> None:
        if self._clips_output_overridden:
            return
        video_raw = self.clips_video_edit.text().strip()
        if not video_raw:
            return
        video_path = Path(video_raw).resolve()
        date_str = self.clips_date_edit.date().toString("yyyy-MM-dd")
        aliases = load_speaker_aliases()
        idx = self.clips_speaker_combo.currentIndex()
        if 0 <= idx < len(aliases):
            speaker_slug = speaker_to_slug(aliases[idx]["canonical"])
        else:
            speaker_slug = speaker_to_slug(self.clips_speaker_combo.currentText().replace("— Other —", "").strip() or "Speaker")
        asset_dir = get_video_asset_dir(video_path, date_str=date_str, speaker_slug=speaker_slug)
        self.clips_output_edit.setText(str(asset_dir))

    def start_extraction(self) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another job is already in progress.")
            return

        video_raw = self.clips_video_edit.text().strip()
        out_raw = self.clips_output_edit.text().strip()
        pad_seconds = float(self.clips_pad_spin.value())

        if not video_raw:
            QMessageBox.critical(self, "Missing input", "Please select or drop a video file.")
            return

        video_path = Path(video_raw).resolve()
        if not video_path.is_file():
            QMessageBox.critical(self, "Missing file", f"Video file not found:\n{video_path}")
            return

        date_str = self.clips_date_edit.date().toString("yyyy-MM-dd")
        aliases = load_speaker_aliases()
        idx = self.clips_speaker_combo.currentIndex()
        if 0 <= idx < len(aliases):
            speaker_slug = speaker_to_slug(aliases[idx]["canonical"])
        else:
            speaker_slug = speaker_to_slug(self.clips_speaker_combo.currentText().replace("— Other —", "").strip() or "Speaker")
        asset_dir = get_video_asset_dir(video_path, date_str=date_str, speaker_slug=speaker_slug)
        prefix = asset_prefix(date_str, speaker_slug)
        out_dir = Path(out_raw).resolve() if out_raw else asset_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self._clips_last_video = video_path
        self._clips_last_out_dir = out_dir
        self._clips_last_prefix = prefix
        upsert_session(
            out_dir,
            {
                "prefix": prefix,
                "date_preached": date_str,
                "source_video_name": artifact_filename("video", prefix=prefix, suffix=video_path.suffix or ".mp4"),
                "source_video_original_name": video_path.name,
                "speaker": {
                    "canonical": aliases[idx]["canonical"] if 0 <= idx < len(aliases) else "",
                    "prompt_name": aliases[idx]["promptName"] if 0 <= idx < len(aliases) else self.clips_speaker_combo.currentText().replace("— Other —", "").strip(),
                    "slug": speaker_slug,
                },
                "artifacts": {"video": artifact_filename("video", prefix=prefix, suffix=video_path.suffix or ".mp4")},
                "steps": {"clips": {"status": "running", "generated_at": now_iso()}},
            },
        )
        self._set_current_asset_dir(out_dir)

        mode = self._clips_source_mode()
        try:
            if mode == "custom":
                json_raw = self.clips_json_edit.toPlainText().strip()
                if not json_raw:
                    QMessageBox.critical(self, "Missing JSON", "Please paste your clips JSON.")
                    return
                clips = load_clips_from_data(json.loads(json_raw))
            else:
                payload = self._build_extraction_payload_from_ranked(kept_only=(mode == "kept"))
                self.clips_json_edit.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
                clips = load_clips_from_data(payload)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid extraction source", str(exc))
            return

        self.clips_log.clear()
        self.clips_progress.setMaximum(len(clips))
        self.clips_progress.setValue(0)
        self.clips_progress.setFormat(f"0/{len(clips)} clips")
        self._set_busy(True)

        worker = ClipWorker(
            video_path=video_path,
            out_dir=out_dir,
            prefix=prefix,
            pad_seconds=pad_seconds,
            clips=clips,
        )
        worker.log.connect(self._on_clips_log)
        worker.progress.connect(self._on_clips_progress)
        self._run_worker(worker, self._on_clips_done)

    def _on_clips_progress(self, current: int, total: int, label: str) -> None:
        self.clips_progress.setMaximum(max(total, 1))
        self.clips_progress.setValue(min(current, max(total, 1)))
        self.clips_progress.setFormat(label)

    def _on_clips_done(self, success: bool, message: str) -> None:
        self._set_busy(False)
        if success:
            self._append_log(self.clips_log, "Done.")
            self.clips_progress.setValue(self.clips_progress.maximum())
            self.statusBar().showMessage("Extraction complete.", 6000)
            self._append_log(self.clips_log, message)
            moved = self._move_source_video_if_needed(
                self._clips_last_video,
                self._clips_last_out_dir,
                prefix=self._clips_last_prefix or None,
            )
            if moved is not None:
                self.clips_video_edit.setText(str(moved))
                self.clips_output_edit.setText(str(self._clips_last_out_dir))
            if self._clips_last_out_dir is not None:
                upsert_session(
                    self._clips_last_out_dir,
                    {
                        "steps": {
                            "clips": {
                                "status": "saved",
                                "generated_at": now_iso(),
                            }
                        }
                    },
                )
                self._loaded_step_flags.add("clips")
                self._set_current_asset_dir(self._clips_last_out_dir)
            if self._auto_run_enabled:
                self._stop_auto_run("Auto run complete.")
        else:
            if self._auto_run_enabled:
                self._stop_auto_run("Auto run stopped because extraction failed.")
            self._write_step_status(self._clips_last_out_dir, "clips", "failed", error=message)
            self._append_log(self.clips_log, f"ERROR: {message}")
            self.statusBar().showMessage("Extraction failed. See log panel.", 8000)
        self._clips_last_video = None
        self._clips_last_out_dir = None
        self._clips_last_prefix = ""
        self._refresh_session_summary()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.active_thread is not None and self.active_thread.isRunning():
            QMessageBox.warning(
                self,
                "Job running",
                "Please wait for the current job to finish before closing FastCap.",
            )
            event.ignore()
            return
        self.stop_preview_playback()
        super().closeEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.WindowStateChange:
            self.title_bar.sync_window_state()
            self.unsetCursor()
            self._resize_cursor_active = False
        super().changeEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if not self.isMaximized():
            self.unsetCursor()
            self._resize_cursor_active = False
        super().leaveEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if self.isMaximized() or not isinstance(event, QMouseEvent):
            return super().eventFilter(watched, event)
        if not isinstance(watched, QWidget):
            return super().eventFilter(watched, event)
        if watched is not self and not self.isAncestorOf(watched):
            return super().eventFilter(watched, event)

        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
            resize_edges = self._resize_edges_at(event.globalPosition().toPoint())
            if resize_edges is not None and self._begin_system_resize(resize_edges):
                return True

        if event.type() == QEvent.Type.MouseMove:
            self._update_resize_cursor(event.globalPosition().toPoint())

        return super().eventFilter(watched, event)

    def _window_handle(self):
        handle = self.windowHandle()
        if handle is None:
            self.winId()
            handle = self.windowHandle()
        return handle

    def _begin_system_move(self) -> bool:
        handle = self._window_handle()
        return bool(handle is not None and handle.startSystemMove())

    def _begin_system_resize(self, edges) -> bool:
        handle = self._window_handle()
        return bool(handle is not None and handle.startSystemResize(edges))

    def _resize_edges_at(self, global_pos: QPoint):
        local_pos = self.mapFromGlobal(global_pos)
        rect = self.rect()
        if not rect.adjusted(0, 0, -1, -1).contains(local_pos):
            return None

        edges = Qt.Edge(0)
        if local_pos.x() <= self._RESIZE_MARGIN:
            edges |= Qt.LeftEdge
        elif local_pos.x() >= rect.width() - self._RESIZE_MARGIN:
            edges |= Qt.RightEdge

        if local_pos.y() <= self._RESIZE_MARGIN:
            edges |= Qt.TopEdge
        elif local_pos.y() >= rect.height() - self._RESIZE_MARGIN:
            edges |= Qt.BottomEdge

        return edges if edges else None

    def _update_resize_cursor(self, global_pos: QPoint) -> None:
        resize_edges = self._resize_edges_at(global_pos)
        if resize_edges is None:
            if self._resize_cursor_active:
                self.unsetCursor()
                self._resize_cursor_active = False
            return

        if resize_edges in (Qt.TopEdge | Qt.LeftEdge, Qt.BottomEdge | Qt.RightEdge):
            self.setCursor(Qt.SizeFDiagCursor)
        elif resize_edges in (Qt.TopEdge | Qt.RightEdge, Qt.BottomEdge | Qt.LeftEdge):
            self.setCursor(Qt.SizeBDiagCursor)
        elif resize_edges in (Qt.LeftEdge, Qt.RightEdge):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.SizeVerCursor)
        self._resize_cursor_active = True


def main() -> None:
    parser = argparse.ArgumentParser(description="FastCap unified desktop app.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. If omitted, writes to ./logs/fastcap-<timestamp>.log",
    )
    args = parser.parse_args()

    log_file = configure_runtime_logging(args.log_file)
    install_exception_hooks()

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#181a1f"))
    palette.setColor(QPalette.WindowText, QColor("#e7e9ee"))
    palette.setColor(QPalette.Base, QColor("#15181f"))
    palette.setColor(QPalette.AlternateBase, QColor("#20242d"))
    palette.setColor(QPalette.ToolTipBase, QColor("#20242d"))
    palette.setColor(QPalette.ToolTipText, QColor("#e7e9ee"))
    palette.setColor(QPalette.Text, QColor("#f0f3fa"))
    palette.setColor(QPalette.Button, QColor("#2a3040"))
    palette.setColor(QPalette.ButtonText, QColor("#e5e9f4"))
    palette.setColor(QPalette.Highlight, QColor("#3a7afe"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setApplicationName("FastCap")
    win = MainWindow()
    win.show()
    logging.info("FastCap UI launched")
    exit_code = app.exec()
    logging.info("FastCap exiting with code %s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
