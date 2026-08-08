import numpy as np

from sims.environment.stretch_controller import StretchController


def _controller_without_simulator(rotation_noise_std_degrees):
    controller = object.__new__(StretchController)
    controller.rotation_noise_std_degrees = rotation_noise_std_degrees
    controller.should_render_image_synthesis = False
    controller.controller = type(
        "FakeController", (), {"step": staticmethod(lambda **kwargs: kwargs)}
    )()
    return controller


def test_rotation_noise_leaves_translation_deterministic(monkeypatch):
    controller = _controller_without_simulator(rotation_noise_std_degrees=0.5)

    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("MoveAgent must not consume motion-noise RNG")

    monkeypatch.setattr(np.random, "normal", fail_if_sampled)
    result = controller.step(action="MoveAgent", ahead=0.2)

    assert result == {
        "action": "MoveAgent",
        "ahead": 0.2,
        "renderImageSynthesis": False,
    }


def test_controller_adds_configured_rotation_noise(monkeypatch):
    controller = _controller_without_simulator(rotation_noise_std_degrees=0.5)
    calls = []

    def fixed_normal(mean, std):
        calls.append((mean, std))
        return np.float64(1.0)

    monkeypatch.setattr(np.random, "normal", fixed_normal)

    result = controller.step(action="RotateAgent", degrees=30.0)

    assert calls == [(0.0, 0.5)]
    assert result == {
        "action": "RotateAgent",
        "degrees": 31.0,
        "renderImageSynthesis": False,
    }


def test_rotation_noise_is_disabled_by_default(monkeypatch):
    controller = _controller_without_simulator(rotation_noise_std_degrees=0.0)

    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("disabled rotation noise must not consume RNG")

    monkeypatch.setattr(np.random, "normal", fail_if_sampled)

    result = controller.step(action="RotateAgent", degrees=30.0)

    assert result["degrees"] == 30.0


def test_instance_rendering_keeps_image_synthesis_enabled():
    controller = StretchController(
        initialize_controller=False,
        renderDepthImage=False,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=False,
    )

    assert controller.should_render_image_synthesis is True
