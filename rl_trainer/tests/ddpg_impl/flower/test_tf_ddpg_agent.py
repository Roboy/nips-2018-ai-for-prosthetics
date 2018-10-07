import os
import shutil

import pytest

from rl_trainer.commons import MockSpace, Episode, ExperienceTuple
from rl_trainer.ddpg_impl.flower.actor_critic import TensorFlowDDPGAgent
from rl_trainer.ddpg_impl.flower.actor_critic.tf_model_saver import TFModelSaver

STATE_DIM = 2
STATE_SPACE = MockSpace(STATE_DIM)
ACTION_DIM = 3
ACTION_SPACE = MockSpace(ACTION_DIM)


@pytest.fixture(scope="module")
def flower():
    return TensorFlowDDPGAgent(state_dim=STATE_DIM, action_space=ACTION_SPACE)


def test_tf_ddpg_agent_act(flower: TensorFlowDDPGAgent):
    action = flower.act(current_state=STATE_SPACE.sample())
    assert ACTION_SPACE.contains(action)


def test_tf_ddpg_agent_observe_episode(flower: TensorFlowDDPGAgent):
    """
    Observing an episode may trigger model saving, so we need to remove
    the created folder.
    """
    new_episode = Episode([ExperienceTuple.mock(STATE_DIM, ACTION_DIM)])
    try:
        flower.observe_episode(new_episode)
    finally:
        if os.path.exists(TFModelSaver.DEFAULT_MODEL_DIR):
            shutil.rmtree(TFModelSaver.DEFAULT_MODEL_DIR)


def test_tf_ddpg_agent_rejects_invalid_episodes(flower: TensorFlowDDPGAgent):
    for invalid_episode in [None, "one"]:
        with pytest.raises(TypeError):
            flower.observe_episode(invalid_episode)


def test_tf_ddpg_agent_set_seed(flower: TensorFlowDDPGAgent):
    flower.set_seed(1)


def test_tf_ddpg_agent_reject_invalid_seed(flower: TensorFlowDDPGAgent):
    for invalid_seed in [None, "one"]:
        with pytest.raises(TypeError):
            flower.set_seed(invalid_seed)
