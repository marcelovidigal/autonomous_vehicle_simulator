"""The Apply button on the GA and DQN screens: applies slider changes + restarts."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

SIZE = (1300, 864)


@pytest.fixture(autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode(SIZE)
    from app import i18n

    i18n.set_lang("en-US")
    yield
    pygame.quit()


def _press(button):
    button._clicked = True


def test_ga_apply_button_exists_and_applies():
    from app.gascreen import GAScreen

    scr = GAScreen(SIZE)
    assert hasattr(scr, "b_apply") and not hasattr(scr, "b_retrain")
    assert scr._dirty() is False

    scr.sliders["hidden"].value = 3          # was the default 6
    assert scr._dirty() is True
    assert scr.trainer.cfg.network.hidden != 3

    _press(scr.b_apply)
    scr.update(1 / 60)

    assert scr.trainer.cfg.network.hidden == 3
    assert scr.trainer.state == "running"
    assert scr._dirty() is False


def test_ga_apply_key_and_reset_clear_dirty():
    from app.gascreen import GAScreen
    from core.config import Config

    scr = GAScreen(SIZE)
    scr.sliders["population"].value = 50
    assert scr._dirty()

    # keyboard 'a' also applies
    scr.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
    assert scr.trainer.cfg.ga.population == 50
    assert not scr._dirty()

    scr.sliders["population"].value = 12
    _press(scr.b_reset)
    scr.update(1 / 60)
    assert not scr._dirty()
    assert scr.trainer.cfg.ga.population == Config().ga.population


def test_dqn_apply_button_exists_and_applies():
    from app.dqnscreen import DQNScreen

    scr = DQNScreen(SIZE)
    assert hasattr(scr, "b_apply")
    assert scr._dirty() is False

    scr.sliders["hidden"].value = 48         # was the default 32
    assert scr._dirty() is True

    _press(scr.b_apply)
    scr.update(1 / 60)

    assert scr.trainer.dqn.hidden == 48
    assert scr.trainer.state == "running"
    assert scr._dirty() is False


def test_dqn_apply_survives_track_change():
    from app.dqnscreen import DQNScreen

    scr = DQNScreen(SIZE)
    scr.sliders["gamma"].value = 0.95
    _press(scr.b_track)                      # changes track + applies
    scr.update(1 / 60)
    # track change applied the pending gamma too (rebuilt from sliders)
    assert scr.trainer.dqn.gamma == 0.95
    assert not scr._dirty()
