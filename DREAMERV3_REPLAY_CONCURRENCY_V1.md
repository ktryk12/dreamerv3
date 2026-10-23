# DreamerV3 — replay concurrency V1

```text
BRANCH:   fix/replay-concurrency-v1, from canonical main @ b8df67a
DATE:     2026-10-23
SCOPE:    one bounded synchronisation fix in embodied/core/selectors.py,
          plus concurrency regression tests
NOT DONE: no replay architecture change, no new locks, no API change
```

---

## 1. The window

`Replay._insert` evicts before it inserts, and reads the new item's stepids
between the two:

```python
while self.capacity and len(self.items) >= self.capacity:
    self._remove()                               # selector loses an item
itemid = self.itemid
self.itemid += 1
self.items[itemid] = (chunkid, index)
stepids = self._getseq(chunkid, index, ['stepid'])['stepid']   # <-- window
self.sampler[itemid] = stepids                   # selector gains it back
```

With `capacity == 1` the selector is empty across that read. Eviction and
sampling are not mutually exclusive: `Replay.add` holds only
`rwlock.reading`, `Replay.sample` takes no rwlock at all, and `Uniform.lock`
serialises the individual selector calls without spanning the evict/insert
pair.

## 2. Reproduced deterministically

No timing involved. `_getseq` is called inside the window, so hooking it lands
there exactly:

```text
len(replay.sampler)  -> 0
replay.sampler()     -> ValueError: high <= 0
```

`numpy` raises that from `rng.integers(0, 0)`. `Replay._sample` does not catch
it — its retry loop handles `KeyError`:

```python
try:
    itemid = self.sampler()
    chunkid, index = self.items[itemid]
    ...
except KeyError:
    continue
```

## 3. How severe it actually is

Measured, not assumed, and narrower than the closeout implied.

A thread that **enters** `Replay.sample` while the selector is empty does not
reach the selector at all: `sample()` begins with

```python
limiters.wait(lambda: len(self.sampler), message)
```

so it blocks there, politely, until the insert completes. Holding the window
open with a slow `_getseq` and sampling from a real thread passes **with or
without** this fix. That is recorded as
`TestSamplerEnteringTheWindow::test_a_sampler_waits_through_a_widened_window`
rather than presented as a catch.

The interleaving the fix covers is the narrower one: a thread that passed the
length check **before** the window opened, was descheduled, and calls
`self.sampler()` inside it. That cannot be reached reliably from threads, which
is why the deterministic hook is the regression guard.

So: a real race, with a real fix, and a smaller blast radius than first
described.

## 4. The fix

One behavioural line in `Uniform.__call__`:

```python
def __call__(self):
    with self.lock:
        if not self.keys:
            raise KeyError('no items to sample from')
        index = self.rng.integers(0, len(self.keys)).item()
        return self.keys[index]
```

`KeyError` is the signal the caller already implements, so the window becomes
the retry `Replay._sample` was written for.

### Why not a lock around evict+insert

That would be the other way to close it: hold a write lock across `_insert`
and take it in `sample()`. It is strictly larger — it serialises inserts
against every sample on the hot path, changes `Replay`'s locking contract, and
would need throughput measurement to justify. The selector-level fix is local,
costs one branch on an already-locked path, and changes no semantics. A lock
would be the right answer only if the window had to be made invisible rather
than survivable, and nothing here requires that.

### What is preserved

- **Sampling distribution.** The draw is untouched, and the generator is
  **not advanced** on the empty path, so the exact sequence of sampled keys is
  unchanged. Asserted by `test_empty_call_does_not_advance_the_generator`
  (sequences after 0, 5 and 50 empty calls must be identical) and
  `test_distribution_is_unchanged` (uniformity over 4000 draws).
- **Eviction semantics.** Untouched; `test_eviction_semantics_hold_under_concurrency`
  checks the retained set at capacity 1, 2 and 10 while a sampler runs.
- **The public API.** No signature changed. `Uniform.__call__` on an empty
  selector previously raised `ValueError` and now raises `KeyError`; it is the
  only observable difference, and it is the point.

## 5. Tests

`embodied/tests/test_replay_concurrency.py`, new:

| Test | Covers |
|---|---|
| `test_window_exists_and_is_a_retryable_keyerror` | the deterministic hook; asserts `KeyError` and explicitly not `ValueError` |
| `test_replay_still_samples_after_the_window` | capacity=1 add/sample round trip |
| `test_a_sampler_waits_through_a_widened_window` | the limiter in front of the selector (section 3) |
| `test_sampling_during_continuous_eviction[1,2,10]` | 3 sampler threads against 2000 inserts |
| `test_eviction_semantics_hold_under_concurrency[1,2,10]` | retained set unchanged under concurrent sampling |
| `test_sequence_matches_a_run_without_retries` | distribution preservation at replay level |
| `test_uniformity_at_normal_capacity` | uniform over 4 items, 4000 draws |

`embodied/tests/test_selectors.py` updated: the test that documented the
`ValueError` limitation now pins `KeyError`, plus the two distribution tests.

**Verified to fail without the fix.** Against a tree exported from
`b8df67a` with only the new test files copied in: 3 failed, 26 passed, with
`ValueError: high <= 0`. The threaded tests pass either way, for the reason in
section 3.

## 6. Verification

```text
collection:  239 tests, 0 errors
run 1:       235 passed, 4 skipped, 0 failed in 74.30s
run 2:       235 passed, 4 skipped, 0 failed in 74.87s
smoke:       exit 0 (--network none, 600 steps Crafter)
```

## 7. Not fixed, recorded

Other selectors fail the same way on an empty call:

```text
Fifo         -> IndexError: deque index out of range
Prioritized  -> ValueError: a cannot be empty unless no samples are taken
Recency      -> requires uprobs; not constructed by default
```

They are only reachable with `replay.fracs.uniform < 1`
(`dreamerv3/main.py:197`), which is not the default
(`configs.yaml: fracs: {uniform: 1.0, priority: 0.0, recency: 0.0}`), so the
default path uses `Replay`'s own `selectors.Uniform`. Left alone deliberately:
fixing selectors nothing currently constructs would widen this change without
closing a reachable defect. If a prioritised or mixture replay is ever
configured, they need the same treatment.
