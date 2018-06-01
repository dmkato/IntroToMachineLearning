import numpy as np
import datetime as dt

class Data:
    """
    Stores a data frame and allows data to be transformed into instances.
    """
    def __init__(self, filepath):
        self.frame = self._get_data(filepath)

    def __getitem__(self, index):
        """
        Overrides the [] syntax to allow for direct array indexing.
            Eg. d = Data("file.txt")
                x0 = d[0]
        """
        return self.frame[index]

    def _time_to_hr(self, timestamp):
        """
        Returns the hour of the day from a timestamp in the format
          '2016-04-05T01:49:47Z'
        """
        date, time = timestamp.split('T')
        return time[:2]

    def _get_data(self, filepath):
        """
        Returns data from filepath as an np.array
        """
        with open(filepath, 'r') as f:
            lines = [l.strip().split(',') for l in f.readlines()]
        hrs = [self._time_to_hr(l[0]) for l in lines]
        data_frame = [[h] + l[1:] for l,h in zip(lines, hrs)]
        return np.matrix(data_frame, dtype=float)
