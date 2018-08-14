from typing import NamedTuple, Sequence

from rl_trainer.commons import ExperienceTuple


class Episode(NamedTuple):
    experience_tuples: Sequence[ExperienceTuple]
