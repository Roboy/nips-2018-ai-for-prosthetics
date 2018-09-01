from typing import NamedTuple, Collection

from rl_trainer.commons import ExperienceTuple


class Episode(NamedTuple):
    experience_tuples: Collection[ExperienceTuple]
