import os

from rl_trainer.commons.experience_tuple import mock_experience_tuple, \
    ExperienceTuple
from rl_trainer.commons import Episode
from rl_trainer.serializer import CSVEpisodeParser
from rl_trainer.serializer import CSVEpisodeSerializer

EXPERIENCE_TUPLE = mock_experience_tuple(action_dim=3, state_dim=2)


def test_parser_construction():
    CSVEpisodeParser()


def test_serializer_construction():
    CSVEpisodeSerializer()


def test_serializer_roundtrip():
    fname = os.path.join(os.path.dirname(__file__), "delme")
    fname_with_postfix = fname + ".csv"
    assert not os.path.exists(fname_with_postfix)

    episode = Episode([EXPERIENCE_TUPLE])
    CSVEpisodeSerializer().serialize(episode=episode, out_fname=fname)
    parsed_episode = CSVEpisodeParser().parse_episode(fname_with_postfix)

    assert parsed_episode == episode
    os.remove(fname_with_postfix)


def test_parse():
    fname = os.path.join(os.path.dirname(__file__), "example.csv")
    assert os.path.isfile(fname)
    parsed_episode = CSVEpisodeParser().parse_episode(fname)

    expected_episode = Episode([ExperienceTuple(
        initial_state=[1.0, 1.0],
        action=[0.6, 0.6, 0.6],
        reward=0.7,
        final_state=[0.3, 0.3],
        final_state_is_terminal=True,
    )])

    assert parsed_episode == expected_episode
