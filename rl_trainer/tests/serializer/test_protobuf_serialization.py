import os

from rl_trainer.commons import Episode, ExperienceTuple
from rl_trainer.episode_serializer import ProtobufEpisodeSerializer

FNAME = os.path.join(os.path.dirname(__file__), "delme.pb")


def test_serialization_roundtrip():
    assert not os.path.isfile(FNAME)

    exp_tuples = [ExperienceTuple.mock(action_dim=2, state_dim=3)]
    episode = Episode(experience_tuples=exp_tuples)
    serializer = ProtobufEpisodeSerializer()

    try:
        serializer.serialize(episode=episode, output_fname=FNAME)
        parsed_episode = serializer.parse(episode_fname=FNAME)
        assert episode == parsed_episode
    finally:
        if os.path.exists(FNAME):
            os.remove(FNAME)
