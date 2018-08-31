import os

from rl_trainer.commons.experience_tuple import ExperienceTupleFactory
from rl_trainer.agent.gym_agent import MockSpace
from rl_trainer.commons import Episode
from rl_trainer.serializer import CSVEpisodeDeserializer
from rl_trainer.serializer import CSVEpisodeSerializer

STATE_SPACE = MockSpace(2)
ACTION_SPACE = MockSpace(3)
EXP_TUPLE_FACTORY = ExperienceTupleFactory(state_space=STATE_SPACE,
                                           action_space=ACTION_SPACE)
EXPERIENCE_TUPLE = EXP_TUPLE_FACTORY.random_tuple()


def test_construction():
    CSVEpisodeDeserializer()


def test_serializer_serialize():
    fname = os.path.join(os.path.dirname(__file__), "example")
    assert not os.path.isfile(fname)

    episode = Episode([EXPERIENCE_TUPLE])
    CSVEpisodeSerializer().serialize(
        episode=episode, out_fname=fname)
    read_episode = CSVEpisodeDeserializer().deserialize_episode(fname + ".csv")

    assert read_episode == episode
