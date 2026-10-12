import base64
import re
from xml.etree import ElementTree

from PIL import Image

from labmentor.config import Settings
from labmentor.ui import (
    ICONS,
    LOADING_CSS,
    STATIC,
    icon_css,
    interface_css,
    launch_options,
    loading_text,
)


def test_every_main_block_has_an_icon_from_the_set():
    css = icon_css()
    images = re.findall(r'url\("data:image/svg\+xml;base64,([^"]+)"\)', css)
    assert len(images) == 2 * len(ICONS)
    for selector, name in ICONS.items():
        assert f"{selector}::before{{-webkit-mask-image:" in css
        root = ElementTree.fromstring((STATIC / "icons" / f"{name}.svg").read_text())
        assert root.get("viewBox") == "0 0 24 24" and len(root) > 0
    assert all(base64.b64decode(data).startswith(b"<svg") for data in images)
    assert "Lucide" in (STATIC / "icons" / "LICENSE").read_text()


def test_launch_options_add_icons_and_favicon(tmp_path):
    options = launch_options(Settings(storage_dir=tmp_path))
    assert options["css"] == interface_css() == icon_css() + "\n" + LOADING_CSS
    assert "@keyframes lm-spin" in LOADING_CSS and "#btn-explain:disabled::before" in LOADING_CSS
    with Image.open(options["favicon_path"]) as favicon:
        assert favicon.size == (64, 64) and favicon.mode == "RGBA"


def test_loading_message_names_the_lab():
    text = loading_text("Лабораторная работа 1.4. Изучение КМОП-схем")
    assert text.startswith('<div class="lm-loading">Готовлю выводы по лабораторной работе 1.4:')
    assert loading_text("Мои заметки").startswith('<div class="lm-loading">Готовлю выводы:')
