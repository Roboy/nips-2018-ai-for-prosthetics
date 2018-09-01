import os

from rl_trainer.commons import Episode
from rl_trainer.commons.experience_tuple import mock_experience_tuple
from rl_trainer.episode_serializer import ProtobufEpisodeSerializer

FNAME = os.path.join(os.path.dirname(__file__), "delme.pb")


def test_serialization_roundtrip():
    assert not os.path.isfile(FNAME)

    exp_tuples = [mock_experience_tuple(action_dim=2, state_dim=3)]
    episode = Episode(experience_tuples=exp_tuples)
    serializer = ProtobufEpisodeSerializer()

    serializer.serialize(episode=episode, output_fname=FNAME)
    parsed_episode = serializer.parse(episode_fname=FNAME)

    assert episode == parsed_episode

    os.remove(FNAME)
