import shutil
import subprocess
import wave

import pytest
from PIL import Image

from conftest import make_video, make_wav
from labmentor import video
from labmentor.assistant import VIDEO_REQUEST, Options
from labmentor.speech import NO_SPEECH, SpeechError
from labmentor.video import clock, contact_sheet, extract_audio, is_video, is_voice, key_frames

SAID = "Включаю генератор, частота один килогерц, на экране синусоида"


def encoded(path, **kwargs):
    try:
        return make_video(path, **kwargs)
    except Exception as exc:
        pytest.skip(f"кодек недоступен: {exc}")


def red_and_blue(image):
    red, _, blue = image.convert("RGB").getpixel((image.width - 5, image.height - 5))
    return red, blue


def test_key_frames_cover_the_whole_video(tmp_path):
    clip = make_video(tmp_path / "clip.mp4")
    length, shots = key_frames(clip)
    moments = [round(moment, 1) for moment, _ in shots]
    assert round(length, 1) == 3.0 and moments == [0.3, 0.8, 1.3, 1.8, 2.3, 2.8]
    first, last = red_and_blue(shots[0][1]), red_and_blue(shots[-1][1])
    assert first[0] > 200 > first[1] and last[1] > 200 > last[0]
    sheet = Image.open(contact_sheet(shots, tmp_path / "sheet.jpg", tenths=True))
    assert sheet.width == 3 * 160 + 4 * 6 and sheet.height > 2 * 120
    assert clock(2.84, tenths=True) == "0:02.8" and clock(125.9) == "2:05"


def test_browser_recording_without_duration(tmp_path):
    clip = encoded(tmp_path / "record.webm", video="libvpx", audio=None, live=True)
    length, shots = key_frames(clip)
    assert round(length, 1) == 3.0 and len(shots) == 6


def test_unseekable_video_is_read_in_one_pass(monkeypatch, tmp_path):
    clip = make_video(tmp_path / "clip.avi", audio=None)
    original = video._shots

    def shots(path, length, count, seek):
        if seek:
            raise OSError("перемотка не поддерживается")
        return original(path, length, count, seek)

    monkeypatch.setattr(video, "_shots", shots)
    _, frames = key_frames(clip)
    assert [round(moment, 1) for moment, _ in frames] == [0.3, 0.8, 1.3, 1.8, 2.3, 2.8]


def test_large_frames_are_scaled_down(tmp_path):
    clip = make_video(tmp_path / "wide.mp4", seconds=1, audio=None, size=(1280, 720))
    _, shots = key_frames(clip, count=2)
    assert [image.size for _, image in shots] == [(480, 270), (480, 270)]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg не установлен")
def test_phone_video_is_turned_upright(tmp_path):
    clip = make_video(tmp_path / "land.mp4", audio=None)
    turned = tmp_path / "IMG_0001.MOV"
    command = ["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(clip)]
    subprocess.run([*command, "-c", "copy", str(turned)], check=True, timeout=60)
    _, shots = key_frames(turned)
    image = shots[0][1].convert("RGB")
    assert image.size == (120, 160)
    green = image.getpixel((5, image.height - 5))
    assert green[1] > 200 and green[0] < 100


def test_sound_is_extracted_for_speech_recognition(tmp_path):
    clip = make_video(tmp_path / "clip.mp4")
    wav = extract_audio(clip, tmp_path / "sound.wav")
    with wave.open(str(wav)) as sound:
        assert sound.getframerate() == 16000 and sound.getnchannels() == 1
        assert abs(sound.getnframes() / 16000 - 3.0) < 0.1
    short = extract_audio(clip, tmp_path / "short.wav", seconds=1)
    with wave.open(str(short)) as sound:
        assert sound.getnframes() == 16000
    silent = make_video(tmp_path / "silent.mp4", audio=None)
    assert extract_audio(silent, tmp_path / "none.wav") is None


def test_voice_recordings_and_videos_are_told_apart(tmp_path):
    film = encoded(tmp_path / "film.webm", video="libvpx", audio="libopus")
    voice = encoded(tmp_path / "voice.webm", video=None, audio="libopus")
    memo = make_video(tmp_path / "memo.mp4", video=None)
    clip = make_video(tmp_path / "clip.mov")
    broken_clip, broken_voice = tmp_path / "broken.mp4", tmp_path / "broken.webm"
    broken_clip.write_bytes(b"not a video")
    broken_voice.write_bytes(b"not a voice")
    wav = make_wav(tmp_path / "q.wav")
    photo = tmp_path / "photo.png"
    Image.new("RGB", (8, 8)).save(photo)
    assert [is_video(p) for p in (film, clip, broken_clip)] == [True] * 3
    assert [is_voice(p) for p in (voice, memo, broken_voice, wav)] == [True] * 4
    assert not any(is_video(p) for p in (voice, memo, broken_voice, wav, photo))
    assert not any(is_voice(p) for p in (film, clip, broken_clip, photo))


def test_video_is_watched_heard_and_remembered(assistant, scripted_llm, monkeypatch, tmp_path):
    heard = []

    def transcribe(path, settings):
        heard.append(path)
        assert path.suffix == ".wav" and path.exists()
        return SAID

    monkeypatch.setattr("labmentor.assistant.transcribe", transcribe)
    clip = make_video(tmp_path / "opyt.mp4")
    session = assistant.new_session()
    got = assistant.ingest(session, [str(clip)])
    assert got.videos == ["opyt.mp4"] and len(got.images) == 1 and not got.speech
    assert Image.open(got.images[0]).size[0] == 3 * 160 + 4 * 6
    assert got.notes == [
        f"Видео opyt.mp4: 0:03, раскадровка из 6 кадров отправлена модели; речь: «{SAID}»"
    ]
    assert not heard[0].exists()
    record = session.files[-1]
    assert (record.name, record.kind) == ("opyt.mp4", "видео") and SAID in record.info
    question = VIDEO_REQUEST.format(names="opyt.mp4")
    reply = list(assistant.respond(session, question, Options(), got.images))[-1]
    parts = scripted_llm.calls[-1][-1].content
    text = parts[0]["text"]
    assert reply.route == "vision" and text.startswith("Посмотри видео (opyt.mp4)")
    assert "раскадровка видео opyt.mp4 длительностью 0:03: 6 кадров" in text and SAID in text
    assert [p["type"] for p in parts].count("image_url") == 1
    system = scripted_llm.calls[-1][0].content
    assert "раскадровка видео" in system and "opyt.mp4 (видео)" in system
    reply = list(assistant.respond(session, "Что на фото в кадре 3?", Options()))[-1]
    parts = scripted_llm.calls[-1][-1].content
    assert reply.route == "vision" and SAID in parts[0]["text"]
    assert session.memory.messages[0].content == f"[Загружены файлы: opyt.mp4] {question}"
    session.forget_files()
    assert session.media == {} and session.images == []


def test_video_without_speech_still_goes_to_the_model(assistant, monkeypatch, tmp_path):
    def fail(path, settings):
        raise AssertionError("в видео без звука нечего распознавать")

    monkeypatch.setattr("labmentor.assistant.transcribe", fail)
    session = assistant.new_session()
    got = assistant.ingest(session, [str(make_video(tmp_path / "silent.mp4", audio=None))])
    assert got.notes[0].endswith("звуковой дорожки нет") and len(got.images) == 1
    assert "Звуковой дорожки в видео нет." in session.media[got.images[0]]

    def quiet(path, settings):
        raise SpeechError(NO_SPEECH)

    monkeypatch.setattr("labmentor.assistant.transcribe", quiet)
    got = assistant.ingest(session, [str(make_video(tmp_path / "noise.mp4"))])
    assert (
        got.notes[0].endswith("; речи нет") and "Речи в видео нет." in session.media[got.images[0]]
    )

    def missing(path, settings):
        raise SpeechError("Модель Whisper не загружена.")

    monkeypatch.setattr("labmentor.assistant.transcribe", missing)
    got = assistant.ingest(session, [str(make_video(tmp_path / "talk.mp4", seconds=2))])
    assert got.notes[0].endswith("речь не распознана (Модель Whisper не загружена.)")
    assert len(got.images) == 1 and got.videos == ["talk.mp4"]


def test_broken_video_is_explained(assistant, tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00" * 64)
    session = assistant.new_session()
    got = assistant.ingest(session, [str(broken)])
    assert got.images == [] and got.videos == [] and session.files == []
    assert got.notes[0].startswith("Файл broken.mp4: ошибка обработки")


def test_audio_only_mp4_is_a_voice_question(assistant, monkeypatch, tmp_path):
    monkeypatch.setattr("labmentor.assistant.transcribe", lambda path, settings: "Цель работы?")
    got = assistant.ingest(
        assistant.new_session(), [str(make_video(tmp_path / "memo.mp4", video=None))]
    )
    assert got.speech == "Цель работы?" and got.images == [] and got.videos == []
