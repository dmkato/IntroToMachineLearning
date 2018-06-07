import numpy as np
import datetime as dt

class Data:
    """
    Stores a data frame and allows data to be transformed into instances.
    """
    def __init__(self, type=None, subsample_rate=1, individual_num=None):
        self.type = type
        self.subsample_rate = subsample_rate  
        self.individual_num = individual_num
        self.frame, self.idxs = self.get_data(type)


    def time_to_hr(self, timestamp):
        """
        Returns the hour of the day from a timestamp in the format
          '2016-04-05T01:49:47Z'
        """
        date, time = timestamp.split('T')
        return [time[:2]]


    def get_file_names(self, type):
        """
        Returns 2 arrays: one for the data files, and one for the idx files
        """
        if type =="general":
            subj_nums = [1, 4, 6, 9]
            data = ['data/general/Subject_{}.csv.xls'.format(n) for n in subj_nums]
            idxs = ['data/general/list_{}.csv.xls'.format(n) for n in subj_nums]
        elif type == "individual":
            num = self.individual_num
            data = ['data/individual/Subject_{}_part1.csv.xls'.format(num)]
            idxs = ['data/individual/list{}_part1.csv.xls'.format(num)]
        else:
            print("Error type not 'general' or 'individual'")
            exit()

        return data, idxs


    def test_data(self):
        """
        Gets test data
        """
        test_arr = []
        filenames = ['data/test/sampleinstance_{}.csv'.format(n) for n in range(1, 6)]
        with open('data/test/groundtruth.csv.xls', 'r') as f:
            ys = [int(l.strip()) for l in f.readlines()]
        for file, y in zip(filenames, ys):
            with open(file, 'r') as f:
                data = np.array([l.strip().split(',') for l in f.readlines()], dtype=float)
                test_arr += [list(data.flatten()) + [y]]

        return test_arr, None


    def get_data(self, type):
        """
        Returns data with index prepended and time converted
        """
        if type == "test":
            return self.test_data()
        all_data = []
        all_idxs = []
        data_files, idx_files = self.get_file_names(type)
        for data_file, idx_file in zip(data_files, idx_files):
            with open(data_file, 'r') as f:
                data = [l.strip().split(',') for l in f.readlines()]
            with open(idx_file, 'r') as f:
                idxs = [int(l.strip()) for l in f.readlines()]
            hr_data = [self.time_to_hr(l[0]) + l[1:] for l in data]
            all_data += hr_data
            all_idxs += idxs
        return np.array(all_data, dtype=float), all_idxs


    def sufficient_data(self, idx):
        """
        Returns true if 30 minutes of continuous data is present
        idx: current element in self.idxs
        """
        e_range = list(range(self.idxs[idx] - 6, self.idxs[idx]+1))
        a_range = self.idxs[idx - 6:idx+1]
        return e_range == a_range


    def instance_at_idx(self, idx):
        """
        Returns instance for the previous 30 mins
        """
        rows = self.frame[idx-6:idx+1]
        rows_without_y = [r[:-1] for r in rows]
        labeled_rows = np.array(rows_without_y)
        f = list(labeled_rows.flatten()) + [rows[-1][-1]]
        return f


    def should_keep(self, counter):
        """
        Returns true if we should keep the example given the subsampling rate
        """
        if i[-1] == 0:
            if removed_count == r_rate:
                removed_count = 0
            else:
                instances.remove(i)
                removed_count += 1


    def subsample(self, instances):
        """
        Removes every subsample_rate negative instance.
        Without subsampling, general data creates 21723 instances.
        """
        removed_count = 0
        r_rate = int(1 / self.subsample_rate)
        ss_instances = []

        for i in instances:
            if i[-1] == 0 and removed_count != r_rate:
                removed_count += 1
                continue
            else:
                ss_instances += [i]
                removed_count = 0

        return ss_instances


    def to_instances(self):
        """
        Converts data to a set of instances where each instance is a set of data that spans 30 consecutive minutes. Each instance overlaps with the last.
        """
        if self.type == "test":
            instances = self.frame
        else:
            raw_instances = [self.instance_at_idx(idx) for idx, _
                 in enumerate(self.idxs) if self.sufficient_data(idx)]
            instances = self.subsample(raw_instances)
        return instances
