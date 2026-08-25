import numpy as np

from openpi.serving import websocket_policy_server


class _FakePolicy:
    def __init__(self):
        self.calls = []

    def infer(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        return {"actions": np.zeros((1, 1))}


def test_infer_policy_request_without_noise():
    policy = _FakePolicy()

    websocket_policy_server.infer_policy_request(policy, {"state": np.ones(2)})

    observation, kwargs = policy.calls[0]
    np.testing.assert_array_equal(observation["state"], np.ones(2))
    assert kwargs == {}


def test_infer_policy_request_forwards_reserved_noise():
    policy = _FakePolicy()
    noise = np.arange(6).reshape(2, 3)

    websocket_policy_server.infer_policy_request(
        policy,
        {"state": np.ones(2), websocket_policy_server.POLICY_NOISE_KEY: noise},
    )

    observation, kwargs = policy.calls[0]
    assert websocket_policy_server.POLICY_NOISE_KEY not in observation
    np.testing.assert_array_equal(kwargs["noise"], noise)
