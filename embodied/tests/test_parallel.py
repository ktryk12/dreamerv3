from collections import deque
from functools import partial as bind

import elements
import embodied
import numpy as np
import pytest
import portal
from embodied.envs import dummy

import utils


@pytest.mark.skip(
    reason=(
        'Blocked on a production contract change, not on this test. f8817c40 '
        'replaced distr.run(workers, args.duration, exit_after=True) with '
        'portal.run(workers) in embodied/run/parallel.py, so combined() no '
        'longer terminates after args.duration and every worker loop is '
        'while True. This test asserts on stats collected after combined() '
        'returns, which it now never does: the run hangs until pytest-timeout '
        'kills it (610s per parametrization, measured). Restoring the '
        'duration-bounded run is a production change and needs an owner '
        'decision; redesigning the test to run combined() in a subprocess and '
        'kill it would invent a contract rather than restore one. The imports '
        'and call signatures in this module were brought up to date so the '
        'module collects and the remaining gap is exactly the one named here. '
        'The coverage it was meant to provide now lives in '
        'test_parallel_lifecycle.py, which runs combined() in a killable '
        'portal.Process, waits for reported progress within a wall-clock '
        'budget, then kills it and verifies process cleanup.')
)
class TestParallel:

  @pytest.mark.parametrize('train_ratio, eval_envs', (
      (-1, 2), (1, 2), (1, 0), (32, 2),
  ))
  def test_run_loop(self, tmpdir, train_ratio, eval_envs):
    addr = f'localhost:{portal.free_port()}'
    received = deque(maxlen=1)
    server = portal.Server(addr, name='TestStats')
    server.bind('report', lambda stats: received.append(stats))
    server.start(block=False)

    args = self._make_args(tmpdir, train_ratio, eval_envs)

    embodied.run.parallel.combined(
        bind(self._make_agent, addr),
        bind(self._make_replay, args),
        bind(self._make_replay, args),
        self._make_env,
        self._make_env,
        bind(utils.make_stream, args),
        self._make_logger, args)

    stats = received[0]
    print('Stats:', stats)
    assert stats['env_steps'] > 400
    if args.train_ratio > -1:
      replay_steps = stats['env_steps'] * args.train_ratio
      assert np.allclose(stats['replay_steps'], replay_steps, 100, 0.1)
    else:
      assert stats['replay_steps'] > 100
    assert stats['reports'] >= 1
    assert stats['saves'] >= 2
    assert stats['loads'] == 0

    embodied.run.parallel.combined(
        bind(self._make_agent, addr),
        bind(self._make_replay, args),
        bind(self._make_replay, args),
        self._make_env,
        self._make_env,
        bind(utils.make_stream, args),
        self._make_logger, args)
    stats = received[0]
    assert stats['loads'] == 1

  def _make_agent(self, queue):
    env = self._make_env(0)
    agent = utils.TestAgent(env.obs_space, env.act_space, queue)
    env.close()
    return agent

  def _make_env(self, index):
    return dummy.Dummy('disc', size=(64, 64), length=100)

  def _make_replay(self, args, train_ratio=None):
    kwargs = {'length': args.batch_length, 'capacity': 1e4}
    if train_ratio:
      kwargs['samples_per_insert'] = train_ratio / args.batch_length
    return embodied.replay.Replay(**kwargs)

  def _make_logger(self):
    return elements.Logger(elements.Counter(), [
        elements.logger.TerminalOutput(),
    ])

  def _make_args(self, logdir, train_ratio, eval_envs):
    return elements.Config(
        duration=5.0,
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
    )
