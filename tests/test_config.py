from core import Config
from core.config import GACfg


def test_defaults_sane():
    c = Config()
    assert c.network.hidden >= 0
    assert c.sensors.count >= 1
    assert c.ga.population >= 2
    assert 0.0 <= c.ga.mutation_rate <= 1.0


def test_roundtrip_dict():
    c = Config(ga=GACfg(population=17, seed=99, crossover=False))
    again = Config.from_dict(c.to_dict())
    assert again == c
    assert again.ga.population == 17
    assert again.ga.crossover is False


def test_from_dict_partial():
    c = Config.from_dict({"sim": {"track": "circuito_2"}})
    assert c.sim.track == "circuito_2"
    assert c.network == Config().network
