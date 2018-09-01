from typeguard import typechecked

from rl_trainer.commons import Episode


class EpisodeSerializer:
    def serialize(self, episode: Episode, output_fname: str) -> None:
        raise NotImplementedError


class EpisodeParser:
    def parse(self, episode_fname: str) -> Episode:
        raise NotImplementedError
