import numpy as np
import pytest
from embodied.core import selectors


STEPIDS = [np.zeros(20, np.uint8)]


def invariants(sel):
  """The two containers of Uniform must stay consistent.

  Uniform keeps keys in a list for O(1) sampling and their positions in a
  dict for O(1) removal, so indices[keys[i]] == i must hold for every i.
  """
  assert len(sel.keys) == len(sel.indices), (sel.keys, sel.indices)
  assert set(sel.keys) == set(sel.indices), (sel.keys, sel.indices)
  for index, key in enumerate(sel.keys):
    assert sel.indices[key] == index, (index, key, sel.indices)
  assert len(sel) == len(sel.keys)


class TestUniform:

  @pytest.mark.parametrize('amount', [1, 2, 5, 20])
  def test_insert_invariants(self, amount):
    sel = selectors.Uniform(seed=0)
    for key in range(amount):
      sel[key] = STEPIDS
      invariants(sel)
    assert len(sel) == amount

  def test_delete_single_item(self):
    # Replay._insert evicts before inserting, so a capacity=1 buffer asks the
    # selector to delete its only item. This used to assert.
    sel = selectors.Uniform(seed=0)
    sel['only'] = STEPIDS
    assert len(sel) == 1
    del sel['only']
    assert len(sel) == 0
    assert sel.keys == []
    assert sel.indices == {}
    invariants(sel)

  def test_delete_then_reinsert(self):
    # The eviction window of a capacity=1 buffer: empty, then refilled.
    sel = selectors.Uniform(seed=0)
    for step in range(5):
      sel[step] = STEPIDS
      invariants(sel)
      del sel[step]
      invariants(sel)
      assert len(sel) == 0
    sel['after'] = STEPIDS
    invariants(sel)
    assert sel() == 'after'

  def test_delete_from_empty_raises(self):
    # The guard that is kept: deleting from an empty selector is a caller bug.
    sel = selectors.Uniform(seed=0)
    with pytest.raises(AssertionError):
      del sel['missing']

  @pytest.mark.parametrize('amount', [1, 2, 5, 20])
  def test_delete_in_insertion_order(self, amount):
    sel = selectors.Uniform(seed=0)
    for key in range(amount):
      sel[key] = STEPIDS
    for index, key in enumerate(range(amount)):
      del sel[key]
      invariants(sel)
      assert len(sel) == amount - index - 1
    assert len(sel) == 0

  @pytest.mark.parametrize('seed', [0, 1, 2])
  def test_delete_in_random_order(self, seed):
    amount = 20
    sel = selectors.Uniform(seed=0)
    for key in range(amount):
      sel[key] = STEPIDS
    order = np.random.default_rng(seed).permutation(amount)
    for index, key in enumerate(order):
      del sel[int(key)]
      invariants(sel)
      assert len(sel) == amount - index - 1

  def test_sampling_only_returns_live_keys(self):
    sel = selectors.Uniform(seed=0)
    for key in range(10):
      sel[key] = STEPIDS
    for key in (0, 3, 9):
      del sel[key]
    live = set(sel.keys)
    assert live == {1, 2, 4, 5, 6, 7, 8}
    seen = {sel() for _ in range(500)}
    assert seen <= live
    assert seen == live, (seen, live)

  def test_sampling_after_capacity_one_cycle(self):
    # What a capacity=1 replay does: evict, insert, sample, repeatedly.
    sel = selectors.Uniform(seed=0)
    sel[0] = STEPIDS
    for step in range(1, 10):
      del sel[step - 1]
      sel[step] = STEPIDS
      assert len(sel) == 1
      assert sel() == step
      invariants(sel)

  def test_call_on_empty_selector(self):
    # Documents a known, narrow limitation rather than asserting it is fine.
    #
    # Uniform.__call__ draws rng.integers(0, len(self.keys)), which numpy
    # rejects when the selector is empty. Replay._insert evicts before it
    # inserts, so a capacity=1 buffer is briefly empty, and Replay.add holds
    # only a read lock while Replay.sample takes no lock at all. A sampler
    # thread that calls into the selector inside that window therefore sees
    # ValueError, which Replay._sample does not catch (it retries KeyError).
    #
    # Making __call__ raise KeyError instead would route it into that existing
    # retry path, but __call__ is outside this change's authorization. If this
    # test starts failing, that fix has landed and the comment above should be
    # updated.
    sel = selectors.Uniform(seed=0)
    with pytest.raises(ValueError):
      sel()
