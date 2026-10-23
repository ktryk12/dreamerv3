"""Concurrency regression tests for the replay eviction window.

`Replay._insert` evicts before it inserts:

    while self.capacity and len(self.items) >= self.capacity:
        self._remove()                      # selector loses an item
    ...
    stepids = self._getseq(chunkid, index, ['stepid'])['stepid']
    self.sampler[itemid] = stepids          # selector gains it back

With `capacity=1` the selector is therefore empty from the eviction until the
new item is registered. `Replay.add` holds only `rwlock.reading` and
`Replay.sample` takes no rwlock at all, so a sampler thread can observe that
window: eviction and sampling are not mutually exclusive, and `Uniform.lock`
does not span the pair.

The first test reproduces the window deterministically by hooking `_getseq`,
which is called inside it — no timing involved. The rest drive it from real
threads at capacity 1, 2 and a normal capacity.
"""

import threading
import time

import embodied
import numpy as np
import pytest


class TestEvictionWindow:

  def test_window_exists_and_is_a_retryable_keyerror(self):
    """Deterministic: land exactly in the window and call the selector."""
    replay = embodied.replay.Replay(length=1, capacity=1)
    original = replay._getseq
    observed = []

    def probe(*args, **kwargs):
      # Inside _insert, after the evict and before the selector insert.
      observed.append(len(replay.sampler))
      try:
        replay.sampler()
        observed.append('returned')
      except Exception as error:  # noqa: BLE001 - the type is the assertion
        observed.append(type(error))
      return original(*args, **kwargs)

    replay.add({'step': 0})       # fills the buffer; no eviction yet
    replay._getseq = probe
    replay.add({'step': 1})       # evicts, so the probe sees the window

    assert observed[0] == 0, 'expected the selector to be empty in the window'
    assert observed[1] is KeyError, observed
    # numpy's ValueError would not be caught by Replay._sample's retry.
    assert observed[1] is not ValueError

  def test_replay_still_samples_after_the_window(self):
    replay = embodied.replay.Replay(length=1, capacity=1)
    for step in range(5):
      replay.add({'step': step})
      batch = replay.sample(1)
      assert batch['step'].item() == step


class TestSamplerEnteringTheWindow:

  def test_a_sampler_waits_through_a_widened_window(self):
    """What protects a sampler that *enters* during the window: the limiter.

    Measured, including against the unfixed code: this passes either way, and
    the selector fix is not what makes it pass. Replay.sample begins with

        limiters.wait(lambda: len(self.sampler), ...)

    so a thread that calls sample() while the selector is empty blocks there,
    politely, until the insert completes. It never reaches the selector.

    That bounds how severe the window is. The interleaving the selector fix
    actually covers is narrower: a thread that passed the length check before
    the window opened, was then descheduled, and calls self.sampler() inside
    it. TestEvictionWindow is the regression guard for that one, because it is
    the only way to reach it deterministically.

    This test is kept because the waiting behaviour is worth pinning: if a
    future change moved the length check or dropped the limiter, a sampler
    would start meeting the empty selector head-on and this test would be the
    first to say so.
    """
    replay = embodied.replay.Replay(length=1, capacity=1)
    replay.add({'step': 0})

    original = replay._getseq
    hold = threading.Event()
    entered = threading.Event()

    def slow_getseq(*args, **kwargs):
      if not entered.is_set():
        entered.set()
        hold.wait(5.0)          # keep the selector empty while we sample
      return original(*args, **kwargs)

    errors = []
    sampled = []

    def sampler():
      try:
        while len(sampled) < 5:
          sampled.append(replay.sample(1)['step'].item())
      except BaseException as error:  # noqa: BLE001 - recorded, then asserted
        errors.append(error)

    thread = threading.Thread(target=sampler)
    thread.start()
    try:
      replay._getseq = slow_getseq
      inserter = threading.Thread(target=lambda: replay.add({'step': 1}))
      inserter.start()
      assert entered.wait(5.0), 'the window was never entered'
      # The sampler is now retrying against an empty selector.
      time.sleep(0.2)
      assert not errors, f'{type(errors[0]).__name__}: {errors[0]}'
      hold.set()
      inserter.join(timeout=10)
    finally:
      hold.set()
      thread.join(timeout=10)

    assert not thread.is_alive()
    assert not errors, f'{type(errors[0]).__name__}: {errors[0]}'
    assert len(sampled) == 5
    assert set(sampled) <= {0, 1}, sampled


class TestRetryIsBounded:
  """The retry must terminate, and it must actually be a retry.

  Replay._sample's loop has no sleep, so if the selector could stay empty the
  retry would spin without end. It cannot: the window closes when _insert
  registers the new item, and Replay.sample waits on len(self.sampler) before
  it ever gets there. These tests pin both halves - that the retry happens,
  and that it ends.
  """

  def test_sample_returns_and_the_retry_is_counted(self):
    replay = embodied.replay.Replay(length=1, capacity=1)
    replay.add({'step': 0})

    # Count the KeyErrors the selector raises, so "it retried" is measured
    # rather than assumed.
    selector = replay.sampler
    original_call = selector.__call__
    retries = [0]

    class Counting:
      def __init__(self, inner):
        self.inner = inner
      def __call__(self):
        try:
          return self.inner()
        except KeyError:
          retries[0] += 1
          raise
      def __getattr__(self, name):
        return getattr(self.inner, name)
      def __setitem__(self, key, value):
        self.inner[key] = value
      def __delitem__(self, key):
        del self.inner[key]
      def __len__(self):
        return len(self.inner)

    replay.sampler = Counting(selector)

    original_getseq = replay._getseq
    opened = threading.Event()
    sampled = []
    errors = []

    def slow_getseq(*args, **kwargs):
      # Hold the window open, but only once, and only for a moment.
      if not opened.is_set():
        opened.set()
        time.sleep(0.3)
      return original_getseq(*args, **kwargs)

    def sampler():
      # _sample bypasses the limiter, which is the interleaving the fix
      # covers: a caller that already passed the length check.
      try:
        opened.wait(5.0)
        started = time.time()
        seq, _ = replay._sample('train')
        sampled.append(time.time() - started)
      except BaseException as error:  # noqa: BLE001 - recorded, then asserted
        errors.append(error)

    thread = threading.Thread(target=sampler)
    thread.start()
    replay._getseq = slow_getseq
    inserter = threading.Thread(target=lambda: replay.add({'step': 1}))
    inserter.start()

    inserter.join(timeout=30)
    thread.join(timeout=30)

    assert not thread.is_alive(), 'the retry did not terminate'
    assert not errors, f'{type(errors[0]).__name__}: {errors[0]}'
    assert len(sampled) == 1
    # It finished, and well inside the join timeout.
    assert sampled[0] < 10.0, sampled
    # And it really did go round the retry at least once.
    assert retries[0] >= 1, retries

  def test_many_evictions_do_not_wedge_a_sampler(self):
    """A long churn must not leave a sampler stuck or starved."""
    replay = embodied.replay.Replay(length=1, capacity=1)
    replay.add({'step': -1})
    done = threading.Event()
    counts = [0]
    errors = []

    def sampler():
      try:
        while not done.is_set():
          replay.sample(1)
          counts[0] += 1
      except BaseException as error:  # noqa: BLE001
        errors.append(error)

    thread = threading.Thread(target=sampler)
    thread.start()
    started = time.time()
    try:
      for step in range(3000):
        replay.add({'step': step})
    finally:
      done.set()
      thread.join(timeout=30)
    elapsed = time.time() - started

    assert not thread.is_alive(), 'sampler thread did not finish'
    assert not errors, f'{type(errors[0]).__name__}: {errors[0]}'
    assert counts[0] > 0, 'sampler never completed a sample'
    # 3000 inserts at capacity=1 with a sampler alongside: seconds, not
    # minutes. A wedged or unboundedly spinning retry would blow this.
    assert elapsed < 60.0, elapsed


@pytest.mark.parametrize('capacity', [1, 2, 10])
class TestConcurrentSampling:

  def test_sampling_during_continuous_eviction(self, capacity):
    """A sampler thread must not see an exception while the buffer churns."""
    replay = embodied.replay.Replay(length=1, capacity=capacity)
    replay.add({'step': -1})

    errors = []
    samples = [0]
    stop = threading.Event()

    def sampler():
      try:
        while not stop.is_set():
          data = replay.sample(1)
          assert data['step'].shape == (1, 1), data['step'].shape
          samples[0] += 1
      except BaseException as error:  # noqa: BLE001 - recorded, then asserted
        errors.append(error)

    threads = [threading.Thread(target=sampler) for _ in range(3)]
    [thread.start() for thread in threads]
    try:
      for step in range(2000):
        replay.add({'step': step})
      # Give the samplers a moment to keep going past the last insert.
      time.sleep(0.2)
    finally:
      stop.set()
      [thread.join(timeout=30) for thread in threads]

    assert not any(thread.is_alive() for thread in threads)
    assert not errors, f'{type(errors[0]).__name__}: {errors[0]}'
    assert samples[0] > 0
    assert len(replay) == min(2001, capacity)

  def test_eviction_semantics_hold_under_concurrency(self, capacity):
    """Sampling concurrently must not change what the buffer retains."""
    replay = embodied.replay.Replay(length=1, capacity=capacity)
    errors = []
    stop = threading.Event()

    def sampler():
      try:
        while not stop.is_set():
          replay.sample(1)
      except BaseException as error:  # noqa: BLE001
        errors.append(error)

    thread = threading.Thread(target=sampler)
    replay.add({'step': 0})
    thread.start()
    try:
      for step in range(1, 500):
        replay.add({'step': step})
        assert len(replay) == min(step + 1, capacity)
    finally:
      stop.set()
      thread.join(timeout=30)

    assert not errors, f'{type(errors[0]).__name__}: {errors[0]}'
    # Only the most recent `capacity` steps survive, concurrency or not.
    live = {replay.sample(1)['step'].item() for _ in range(200)}
    assert live <= set(range(500 - capacity, 500)), (live, capacity)


class TestSamplingDistributionPreserved:

  def test_sequence_matches_a_run_without_retries(self):
    """The retry path must not perturb which items come out.

    Uniform returns before the draw when empty, so the generator is not
    advanced. A capacity=1 buffer that hits the window therefore yields the
    same sequence as a buffer that never does.
    """
    def sequence(capacity):
      replay = embodied.replay.Replay(
          length=1, capacity=capacity,
          selector=embodied.replay.selectors.Uniform(seed=3))
      for step in range(capacity):
        replay.add({'step': step})
      return [replay.sample(1)['step'].item() for _ in range(100)]

    # capacity=1 drives the eviction window on every add after the first;
    # capacity=100 never evicts. Both must sample from their own live set with
    # the same generator stream, so the first is constant and the second is
    # the untouched uniform draw.
    assert sequence(1) == [0] * 100
    draws = sequence(100)
    assert set(draws) <= set(range(100))
    assert len(set(draws)) > 50, 'expected a spread of draws, not a constant'

  def test_uniformity_at_normal_capacity(self):
    replay = embodied.replay.Replay(
        length=1, capacity=4,
        selector=embodied.replay.selectors.Uniform(seed=0))
    for step in range(4):
      replay.add({'step': step})
    counts = {}
    for _ in range(4000):
      step = replay.sample(1)['step'].item()
      counts[step] = counts.get(step, 0) + 1
    assert set(counts) == {0, 1, 2, 3}, counts
    for step, count in counts.items():
      assert 800 < count < 1200, counts
    assert np.isclose(sum(counts.values()), 4000)
