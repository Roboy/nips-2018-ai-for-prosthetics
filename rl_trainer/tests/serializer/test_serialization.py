import os

from rl_trainer.commons.experience_tuple import mock_experience_tuple
from rl_trainer.commons import Episode
from rl_trainer.serializer import CSVEpisodeDeserializer
from rl_trainer.serializer import CSVEpisodeSerializer

EXPERIENCE_TUPLE = mock_experience_tuple(action_dim=3, state_dim=2)


def test_construction():
    CSVEpisodeDeserializer()


def test_serializer_serialize():
    fname = os.path.join(os.path.dirname(__file__), "example")
    assert not os.path.isfile(fname)

    episode = Episode([EXPERIENCE_TUPLE])
    CSVEpisodeSerializer().serialize(
        episode=episode, out_fname=fname)
    parsed_episode = CSVEpisodeDeserializer().deserialize_episode(fname + ".csv")

    assert parsed_episode == episode
