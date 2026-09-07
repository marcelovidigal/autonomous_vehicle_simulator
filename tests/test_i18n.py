import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def test_every_key_has_both_languages():
    from app import i18n

    for key, row in i18n._STR.items():
        assert set(row) == {"en-US", "pt-BR"}, key


def test_default_and_switch():
    from app import i18n

    i18n.set_lang("en-US")
    assert i18n.t("menu.about") == "About"
    i18n.set_lang("pt-BR")
    assert i18n.get_lang() == "pt-BR"
    assert i18n.t("menu.about") == "Sobre"
    i18n.set_lang("en-US")


def test_format_args_and_fallback():
    from app import i18n

    i18n.set_lang("en-US")
    assert i18n.t("ga.hud.gen", g=3, n=45, s="running") == "Generation 3/45   [running]"
    assert i18n.t("no.such.key") == "no.such.key"


def test_fps_setting():
    from app import i18n

    assert i18n.get_fps() in i18n.FPS_CHOICES
    i18n.set_fps(30)
    assert i18n.get_fps() == 30
    i18n.set_fps(999)  # not an allowed choice -> ignored
    assert i18n.get_fps() == 30
    i18n.set_lang("en-US")
    assert i18n.fps_label(0) == "Uncapped"
    assert i18n.fps_label(60) == "60 FPS"
    i18n.set_fps(60)
