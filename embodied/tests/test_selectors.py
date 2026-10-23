import collections

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

  def test_call_on_empty_selector_raises_keyerror(self):
    # KeyError, not numpy's ValueError: it is the signal Replay._sample's
    # retry loop understands, which turns the capacity=1 eviction window into
    # a retry instead of a crash.
    sel = selectors.Uniform(seed=0)
    with pytest.raises(KeyError):
      sel()

  def test_empty_call_does_not_advance_the_generator(self):
    # The fix must not change which keys get sampled. The empty path returns
    # before the draw, so a selector that was called while empty produces the
    # same sequence as one that never was.
    def sequence(empty_calls):
      sel = selectors.Uniform(seed=7)
      for _ in range(empty_calls):
        with pytest.raises(KeyError):
          sel()
      for key in range(10):
        sel[key] = STEPIDS
      return [sel() for _ in range(50)]

    assert sequence(0) == sequence(5) == sequence(50)

  def test_distribution_is_unchanged(self):
    # Same seed, same keys, same draws as before the fix: uniform over the
    # live keys, with every key reachable.
    sel = selectors.Uniform(seed=0)
    for key in range(4):
      sel[key] = STEPIDS
    counts = collections.Counter(sel() for _ in range(4000))
    assert set(counts) == {0, 1, 2, 3}
    for key, count in counts.items():
      assert 800 < count < 1200, counts
