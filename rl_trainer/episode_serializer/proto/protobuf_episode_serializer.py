import os

from rl_trainer.episode_serializer.proto import episode_pb2
from typeguard import typechecked

from rl_trainer.commons import Episode, ExperienceTuple
from rl_trainer.episode_serializer.episode_serializer import EpisodeSerializer, EpisodeParser


class ProtobufEpisodeSerializer(EpisodeSerializer, EpisodeParser):

    @typechecked
    def parse(self, episode_fname: str) -> Episode:
        assert os.path.isfile(episode_fname), f"{episode_fname} does not exist."
        pb_episode = episode_pb2.Episode()
        with open(episode_fname, "rb") as file:
            pb_episode.ParseFromString(file.read())
        exp_tuples = [self._to_exp_tuple(pb_e) for pb_e in pb_episode.experience_tuples]
        return Episode(experience_tuples=exp_tuples)

    @typechecked
    def serialize(self, episode: Episode, output_fname: str) -> None:
        assert not os.path.isfile(output_fname), f"{output_fname} already exists."
        pb_episode = episode_pb2.Episode()
        pb_exp_tuples = [self._to_pb_exp_tuple(e) for e in episode.experience_tuples]
        pb_episode.experience_tuples.extend(pb_exp_tuples)
        with open(output_fname, "wb") as file:
            file.write(pb_episode.SerializeToString())

    @staticmethod
    def _to_exp_tuple(pb_exp_tuple: episode_pb2.ExperienceTuple) -> ExperienceTuple:
        return ExperienceTuple(
            state_1=pb_exp_tuple.state_1,
            action=pb_exp_tuple.action,
            reward=pb_exp_tuple.reward,
            state_2=pb_exp_tuple.state_2,
            state_2_is_terminal=pb_exp_tuple.state_2_is_terminal,
        )

    @staticmethod
    def _to_pb_exp_tuple(exp_tuple: ExperienceTuple) -> episode_pb2.ExperienceTuple:
        pb_exp_tuple = episode_pb2.ExperienceTuple()
        pb_exp_tuple.state_1.extend(exp_tuple.state_1)
        pb_exp_tuple.action.extend(exp_tuple.action)
        pb_exp_tuple.reward = exp_tuple.reward
        pb_exp_tuple.state_2.extend(exp_tuple.state_2)
        pb_exp_tuple.state_2_is_terminal = exp_tuple.state_2_is_terminal
        return pb_exp_tuple
