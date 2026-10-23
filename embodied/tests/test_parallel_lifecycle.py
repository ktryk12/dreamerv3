"""Time-bounded lifecycle test for the parallel run loop.

`embodied.run.parallel.combined()` ends in `portal.run(workers)` and every
worker loop is `while True`, so it never returns; `args.duration` is no longer
read anywhere in `embodied/run/parallel.py`. `test_parallel.py` asserts on
stats collected after `combined()` returns and therefore cannot pass without a
production change.

This module tests the same thing from the outside instead: run `combined()` in
a `portal.Process`, watch the stats it reports to a server in this process,
stop watching as soon as the run has demonstrably made progress, then kill the
worker and verify that it and all its nested children are gone.

Every wait here is bounded by `BUDGET`, so the module cannot hang.
"""

import time
from collections import deque
from functools import partial as bind

import elements
import embodied
import numpy as np
import portal
import psutil
import pytest
from embodied.envs import dummy

import utils


# Wall-clock budget for one run to reach the progress thresholds below. The
# measured time to get there is a few seconds; this is the ceiling at which the
# test fails rather than waits.
BUDGET = 120.0
POLL = 0.2

# Progress thresholds, taken from the assertions in test_parallel.py.
MIN_ENV_STEPS = 400
MIN_SAVES = 2
MIN_REPORTS = 1


def make_env(index):
  return dummy.Dummy('disc', size=(64, 64), length=100)


def make_agent(addr):
  env = make_env(0)
  agent = utils.TestAgent(env.obs_space, env.act_space, addr)
  env.close()
  return agent


def make_replay(args):
  return embodied.replay.Replay(length=args.batch_length, capacity=1e4)


def make_logger():
  return elements.Logger(elements.Counter(), [
      elements.logger.TerminalOutput(),
  ])


def make_args(logdir, train_ratio, eval_envs):
  return elements.Config(
      train_ratio=float(train_ratio),
      log_every=0.1,
      report_every=0.2,
      save_every=0.2,
      envs=4,
      eval_envs=int(eval_envs),
      report_batches=1,
      from_checkpoint='',
      episode_timeout=10,
      actor_addr='localhost:{auto}',
      replay_addr='localhost:{auto}',
      logger_addr='localhost:{auto}',
      ipv6=False,
      actor_batch=-1,
      actor_threads=2,
      agent_process=False,
      remote_replay=False,
      remote_envs=False,
      usage=dict(psutil=True, nvsmi=False),
      debug=False,
      logdir=str(logdir),
      batch_size=8,
      batch_length=16,
      replay_context=0,
      report_length=8,
      consec_report=1,
  )


def run_combined(addr, args):
  """Entry point of the worker process. Never returns on its own."""
  embodied.run.parallel.combined(
      bind(make_agent, addr),
      bind(make_replay, args),
      bind(make_replay, args),
      make_env,
      make_env,
      bind(utils.make_stream, args),
      make_logger,
      args)


class Harness:
  """Runs combined() in a killable process and collects its stats."""

  def __init__(self, logdir, train_ratio=32, eval_envs=2):
    self.addr = f'localhost:{portal.free_port()}'
    self.received = deque(maxlen=1)
    self.server = portal.Server(self.addr, name='TestStats')
    self.server.bind('report', lambda stats: self.received.append(stats))
    self.server.start(block=False)
    self.args = make_args(logdir, train_ratio, eval_envs)
    self.worker = None

  def descendants(self):
    """Child pids of this process, excluding Python's own helpers.

    multiprocessing spawns one `resource_tracker` per parent interpreter to
    clean up shared semaphores. portal.Process uses mp barriers, so the
    tracker appears on first use and then lives as long as pytest does. It is
    not a leaked worker, so it is not counted; anything else is.
    """
    pids = set()
    try:
      children = psutil.Process().children(recursive=True)
    except psutil.Error:  # pragma: no cover - host-dependent
      return pids
    for proc in children:
      try:
        if 'multiprocessing.resource_tracker' in ' '.join(proc.cmdline()):
          continue
      except psutil.Error:  # pragma: no cover - process exited while reading
        continue
      pids.add(proc.pid)
    return pids

  def start(self):
    self.before = self.descendants()
    self.worker = portal.Process(
        run_combined, self.addr, self.args, name='combined', start=True)
    return self.worker

  def wait_for_progress(
      self, min_env_steps=MIN_ENV_STEPS, min_saves=MIN_SAVES,
      min_reports=MIN_REPORTS):
    """Block until the run reports real progress, or fail inside BUDGET.

    Every quantity that a caller asserts on has to be waited for here. The
    stats arrive on one channel but are produced by different workers, so
    reports can still be zero when saves have already happened.
    """
    deadline = time.time() + BUDGET
    while time.time() < deadline:
      assert self.worker.running, 'combined() exited before reporting progress'
      if self.received:
        stats = self.received[0]
        if (stats['env_steps'] >= min_env_steps
            and stats['saves'] >= min_saves
            and stats['reports'] >= min_reports):
          return stats
      time.sleep(POLL)
    last = dict(self.received[0]) if self.received else None
    pytest.fail(
        f'No progress within {BUDGET}s. Last stats: {last}')

  def stop(self):
    if self.worker is not None:
      self.worker.kill()
    self.server.close()

  def assert_cleaned_up(self):
    assert not self.worker.running, 'worker still running after kill()'
    deadline = time.time() + 30.0
    while time.time() < deadline:
      leaked = self.descendants() - self.before
      if not leaked:
        return
      time.sleep(POLL)
    leaked = self.descendants() - self.before
    details = []
    for pid in leaked:
      try:
        details.append((pid, psutil.Process(pid).cmdline()))
      except psutil.Error:  # pragma: no cover
        pass
    pytest.fail(f'Processes left behind after kill(): {details}')


@pytest.mark.timeout(BUDGET * 3)
class TestParallelLifecycle:

  def test_runs_terminates_and_cleans_up(self, tmpdir):
    harness = Harness(tmpdir)
    try:
      harness.start()
      stats = harness.wait_for_progress()
      # The assertions of test_parallel.py that do not need combined() to
      # return. train_ratio is 32, so replay steps should track env steps.
      assert stats['env_steps'] >= MIN_ENV_STEPS
      assert stats['replay_steps'] > 100
      assert stats['reports'] >= 1
      assert stats['saves'] >= MIN_SAVES
      assert stats['loads'] == 0
      expected = stats['env_steps'] * harness.args.train_ratio
      assert np.allclose(stats['replay_steps'], expected, 100, 0.1)
    finally:
      harness.stop()
    harness.assert_cleaned_up()

  def test_restart_loads_the_checkpoint(self, tmpdir):
    # The second half of test_parallel.py's assertion: a fresh run against the
    # same logdir restores from the checkpoint the first run wrote.
    first = Harness(tmpdir)
    try:
      first.start()
      stats = first.wait_for_progress()
      assert stats['loads'] == 0
    finally:
      first.stop()
    first.assert_cleaned_up()

    second = Harness(tmpdir)
    try:
      second.start()
      stats = second.wait_for_progress(min_saves=1)
      assert stats['loads'] == 1, stats
    finally:
      second.stop()
    second.assert_cleaned_up()

  def test_without_eval_envs(self, tmpdir):
    harness = Harness(tmpdir, eval_envs=0)
    try:
      harness.start()
      stats = harness.wait_for_progress()
      assert stats['env_steps'] >= MIN_ENV_STEPS
    finally:
      harness.stop()
    harness.assert_cleaned_up()
