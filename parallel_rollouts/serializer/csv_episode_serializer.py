import csv
from typing import Sequence

from commons.experience_tuple import ExperienceTuple


class CSVEpisodeSerializer:

    def serialize(self, episode: Sequence[ExperienceTuple], fname: str):
        with open(fname + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            for tuple in episode:
                line = [
                    self._format_line(tuple.initial_state),
                    self._format_line(tuple.action),
                    self._format_line(tuple.final_state),
                ]
                writer.writerow(line)

    @staticmethod
    def _format_line(line: Sequence[float]) -> Sequence[str]:
        return list((f"{float(e):.4}" for e in line))
