import os

from rl_trainer.commons import ExperienceTuple, Episode
from rl_trainer.episode_serializer import CSVEpisodeParser, CSVEpisodeSerializer

EXPERIENCE_TUPLE = ExperienceTuple.mock(action_dim=3, state_dim=2)


def test_parser_construction():
    CSVEpisodeParser()


def test_serializer_construction():
    CSVEpisodeSerializer()


def test_serializer_roundtrip():
    fname = os.path.join(os.path.dirname(__file__), "delme")
    fname_with_postfix = fname + ".csv"
    assert not os.path.exists(fname_with_postfix)

    episode = Episode([EXPERIENCE_TUPLE])
    try:
        CSVEpisodeSerializer().serialize(episode=episode, out_fname=fname)
        parsed_episode = CSVEpisodeParser().parse_episode(fname_with_postfix)

        assert parsed_episode == episode
    finally:
        if os.path.exists(fname_with_postfix):
            os.remove(fname_with_postfix)


def test_parse():
    fname = os.path.join(os.path.dirname(__file__), "example.csv")
    assert os.path.isfile(fname)
    parsed_episode = CSVEpisodeParser().parse_episode(fname)

    expected_episode = Episode([ExperienceTuple(
        state_1=[1.0, 1.0],
        action=[0.6, 0.6, 0.6],
        reward=0.7,
        state_2=[0.3, 0.3],
        state_2_is_terminal=True,
    )])

    assert parsed_episode == expected_episode
