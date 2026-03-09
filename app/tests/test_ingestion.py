from __future__ import annotations

from app.services import ingestion


def test_yt_dlp_prefix_uses_path_lookup(monkeypatch):
    monkeypatch.setattr(ingestion.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")
    prefix = ingestion._yt_dlp_cmd_prefix()
    assert prefix == ["/usr/local/bin/yt-dlp"]


def test_yt_dlp_prefix_uses_venv_neighbor(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.shutil, "which", lambda _: None)
    fake_python = tmp_path / "python3.12"
    fake_python.write_text("", encoding="utf-8")
    neighbor = tmp_path / "yt-dlp"
    neighbor.write_text("", encoding="utf-8")
    monkeypatch.setattr(ingestion.sys, "executable", str(fake_python))

    prefix = ingestion._yt_dlp_cmd_prefix()
    assert prefix == [str(neighbor)]


def test_yt_dlp_prefix_falls_back_to_module(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.shutil, "which", lambda _: None)
    fake_python = tmp_path / "python3.12"
    fake_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(ingestion.sys, "executable", str(fake_python))

    prefix = ingestion._yt_dlp_cmd_prefix()
    assert prefix == [str(fake_python), "-m", "yt_dlp"]


def test_youtube_cache_key_normalizes_watch_variants():
    a = "https://www.youtube.com/watch?v=Ylgvl8d2NX4"
    b = "https://www.youtube.com/watch?v=Ylgvl8d2NX4&list=RDYlgvl8d2NX4&start_radio=1"
    c = "https://youtu.be/Ylgvl8d2NX4"
    assert ingestion._youtube_cache_key(a) == ingestion._youtube_cache_key(b)
    assert ingestion._youtube_cache_key(a) == ingestion._youtube_cache_key(c)


def test_derive_artist_title_uses_topic_channel_as_artist_when_title_is_song_only():
    title, artist = ingestion._derive_artist_title("Some Song", "Some Artist - Topic")
    assert title == "Some Song"
    assert artist == "Some Artist"


def test_derive_artist_title_strips_topic_case_insensitive():
    title, artist = ingestion._derive_artist_title("Another Song", "Another Artist - TOPIC")
    assert title == "Another Song"
    assert artist == "Another Artist"


def test_metadata_selection_prefers_title_and_topic_over_fallback_artist():
    data = {
        "title": "Primary Artist - Clean Title",
        "uploader": "Wrong Artist - Topic",
        "artist": "Description Artist",
    }
    title, artist = ingestion._select_artist_title_from_metadata(data)
    assert title == "Clean Title"
    assert artist == "Primary Artist"


def test_metadata_selection_uses_fallback_artist_when_topic_missing():
    data = {
        "title": "Only Song Name",
        "uploader": None,
        "channel": None,
        "artist": "Description Artist",
    }
    title, artist = ingestion._select_artist_title_from_metadata(data)
    assert title == "Only Song Name"
    assert artist == "Description Artist"
