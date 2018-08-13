import csv
from typing import List

from commons.experience_tuple import ExperienceTuple


class CSVEpisodeDeserializer:

    def deserialize_episode(self, fname: str) -> List[ExperienceTuple]:
        episode: List[ExperienceTuple] = []
        with open(fname, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                experience_tuple = ExperienceTuple(
                    initial_state=self._deserialize_line(line[0]),
                    action=self._deserialize_line(line[1]),
                    final_state=self._deserialize_line(line[2]),
                )
                episode.append(experience_tuple)
        return episode

    @staticmethod
    def _deserialize_line(line: str):
        nums = line.strip()[1:-1].replace("'", "").split(",")
        return [float(num) for num in nums]
