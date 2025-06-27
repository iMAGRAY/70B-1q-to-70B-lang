from sigla import CapsuleStore
from sigla.dsl import INJECT


def test_inject_returns_id(tmp_path):
    store = CapsuleStore(model_name="dummy")
    msg = INJECT("hello", store)
    assert store.meta[0]["text"] == "hello"
    assert msg.startswith("[INJECTED AS CAPSULE 0]")

