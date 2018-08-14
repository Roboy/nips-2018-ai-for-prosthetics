from typing import NamedTuple, Sequence

from commons import ExperienceTuple


class Episode(NamedTuple):
    experience_tuples: Sequence[ExperienceTuple]
