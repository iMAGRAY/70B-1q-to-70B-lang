import os

from sigla import CapsuleStore

def test_add_and_query(tmp_path):
    store = CapsuleStore(auto_link_k=2)
    texts = ["hello world", "world of ai", "machine learning is fun"]
    store.add_capsules([{"text": t} for t in texts])

    # Basic query
    results = store.query("ai world", top_k=2)
    assert results, "No results returned"
    assert results[0]["id"] in {0,1}

    # Auto links present
    assert any(store.meta[0]["links"]), "Auto links not created"

    # Save & load roundtrip
    index_path = tmp_path / "test_caps"
    store.save(str(index_path))

    new_store = CapsuleStore()
    new_store.load(str(index_path))
    assert new_store.meta[1]["text"] == "world of ai" 