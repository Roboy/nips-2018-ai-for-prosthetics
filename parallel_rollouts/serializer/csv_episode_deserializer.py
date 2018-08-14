import csv
from typing import List

from commons import Episode
from commons.experience_tuple import ExperienceTuple


class CSVEpisodeDeserializer:

    def deserialize_episode(self, episode_fname: str) -> Episode:
        experience_tuples: List[ExperienceTuple] = []
        with open(episode_fname, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                initial_state=self._deserialize_line(line[0])
                if not len(initial_state) == 158:
                    print("found bad length of state")
                    continue
                action=self._deserialize_line(line[1])
                if not len(action) == 19:
                    print("found bad length of action")
                    continue
                final_state=self._deserialize_line(line[2])
                if not len(final_state) == 158:
                    print("found bad length of resuting state")
                    continue
                experience_tuple = ExperienceTuple(
                    initial_state=initial_state,
                    action=action,
                    final_state=final_state,
                )
                experience_tuples.append(experience_tuple)
        return Episode(experience_tuples=experience_tuples)

    @staticmethod
    def _deserialize_line(line: str):
        nums = line.strip()[1:-1].replace("'", "").split(",")
        return [float(num) for num in nums]
