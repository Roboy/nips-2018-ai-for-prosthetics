import csv
from typing import List

from rl_trainer.commons import Episode
from rl_trainer.commons.experience_tuple import ExperienceTuple


class CSVEpisodeDeserializer:

    def deserialize_episode(self, episode_fname: str) -> Episode:
        experience_tuples: List[ExperienceTuple] = []
        with open(episode_fname, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                experience_tuple = ExperienceTuple(
                    initial_state=self._deserialize_line(line[0]),
                    action=self._deserialize_line(line[1]),
                    final_state=self._deserialize_line(line[2]),
                    reward=self._deserialize_line(line[3])[0],
                )
                experience_tuples.append(experience_tuple)
        return Episode(experience_tuples=experience_tuples)

    @staticmethod
    def _deserialize_line(line: str):
        nums = line.strip()[1:-1].replace("'", "").split(",")
        return [float(num) for num in nums]
