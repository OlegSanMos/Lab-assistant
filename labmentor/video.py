from __future__ import annotations

import math
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .speech import AUDIO_EXTENSIONS, SAMPLE_RATE

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".wmv",
}
FRAMES = 6
TILE = 480
GAP = 6
AUDIO_SECONDS = 180


def is_video(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    try:
        import av

        with av.open(str(path)) as container:
            return bool(container.streams.video)
    except Exception:
        return path.suffix.lower() not in AUDIO_EXTENSIONS


def is_voice(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return (suffix in AUDIO_EXTENSIONS or suffix in VIDEO_EXTENSIONS) and not is_video(path)


def clock(seconds: float, tenths: bool = False) -> str:
    if tenths:
        minutes, rest = divmod(round(max(0.0, seconds) * 10), 600)
        return f"{minutes}:{rest // 10:02d}.{rest % 10}"
    minutes, rest = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes}:{rest:02d}"


def _length(container, stream) -> float:
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / 1_000_000
    last = 0.0
    for packet in container.demux(stream):
        if packet.pts is not None and packet.time_base:
            last = max(last, float((packet.pts + (packet.duration or 0)) * packet.time_base))
    return last


def _picture(frame) -> Image.Image:
    scale = min(1.0, TILE / max(frame.width, frame.height))
    width, height = max(2, round(frame.width * scale)), max(2, round(frame.height * scale))
    image = frame.to_image(width=width, height=height)
    rotation = getattr(frame, "rotation", 0) or 0
    return image.rotate(rotation, expand=True) if rotation else image


def _first_after(frames, target: float):
    picked = None
    for frame in frames:
        picked = frame
        if frame.time is None or frame.time >= target - 0.001:
            break
    return picked


def _shots(path: str | Path, length: float, count: int, seek: bool) -> list:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        frames = container.decode(stream)
        shots, seen = [], set()
        for index in range(count):
            target = start + length * (index + 0.5) / count
            if seek:
                container.seek(int(target / stream.time_base), stream=stream)
                frames = container.decode(stream)
            frame = _first_after(frames, target)
            if frame is None:
                continue
            moment = (frame.time if frame.time is not None else target) - start
            if round(moment, 3) not in seen:
                seen.add(round(moment, 3))
                shots.append((moment, _picture(frame)))
    return shots


def key_frames(path: str | Path, count: int = FRAMES) -> tuple[float, list]:
    import av

    with av.open(str(path)) as probe:
        length = _length(probe, probe.streams.video[0])
    try:
        return length, _shots(path, length, count, seek=True)
    except Exception:
        return length, _shots(path, length, count, seek=False)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def contact_sheet(shots: list, target: Path, tenths: bool = False) -> Path:
    columns = 2 if len(shots) == 4 else min(3, len(shots))
    rows = math.ceil(len(shots) / columns)
    width = max(image.width for _, image in shots)
    height = max(image.height for _, image in shots)
    size = max(14, min(26, width // 20))
    cell = height + size + 12
    sheet = Image.new(
        "RGB",
        (columns * width + (columns + 1) * GAP, rows * cell + (rows + 1) * GAP),
        (32, 32, 32),
    )
    draw, font = ImageDraw.Draw(sheet), _font(size)
    for index, (moment, image) in enumerate(shots):
        row, column = divmod(index, columns)
        left, top = GAP + column * (width + GAP), GAP + row * (cell + GAP)
        place = (left + (width - image.width) // 2, top + (height - image.height) // 2)
        sheet.paste(image.convert("RGB"), place)
        label = f"{index + 1}    {clock(moment, tenths)}"
        draw.text((left + 4, top + height + 5), label, fill=(235, 235, 235), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, "JPEG", quality=90)
    return target


def extract_audio(path: str | Path, target: Path, seconds: int = AUDIO_SECONDS) -> Path | None:
    import av

    limit, written = seconds * SAMPLE_RATE, 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            frames = container.decode(audio=0)
            while written < limit:
                try:
                    frame = next(frames)
                except StopIteration:
                    frame = None
                except av.error.InvalidDataError:
                    continue
                for part in resampler.resample(frame):
                    samples = part.to_ndarray().reshape(-1)[: limit - written]
                    wav.writeframes(samples.tobytes())
                    written += len(samples)
                if frame is None:
                    break
    if not written:
        target.unlink(missing_ok=True)
        return None
    return target
