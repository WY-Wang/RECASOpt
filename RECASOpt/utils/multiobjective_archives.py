import numpy as np
from .multiobjective_utilities import ND_add

INF = float('inf')


class Record:
    def __init__(self, x, fx, rank = None):
        self.x = x
        self.fx = fx

        self.bar_fx = None
        self.rank = rank

        self.maxradius = 0.2
        self.minradius = 0.2 * 0.5 ** 6.0
        self.radius = self.maxradius

        self.associate = None
        self.associate_theta = None

    def reset(self):
        self.radius = self.maxradius


class FrontArchive:
    def __init__(self, size = None):
        self.fronts = []
        self.size = size
        if self.size is None:
            self.size = 300
        self.num_records = 0

    def reset(self):
        self.fronts = []
        self.num_records = 0

    def add(self, record, cur_rank = None):
        if cur_rank is None:
            cur_rank = 1
        if self.fronts:
            ranked = False
            while cur_rank <= len(self.fronts):
                front = self.fronts[cur_rank - 1]

                fvals = [rec.fx for rec in front]
                nrecords = len(fvals)
                nondominated = list(range(nrecords))
                dominated = []
                fvals.append(record.fx)
                fvals = np.asarray(fvals)
                (nondominated, dominated) = ND_add(np.transpose(fvals), nondominated, dominated)
                if not dominated:
                    ranked = True
                    record.rank = cur_rank
                    front.append(record)
                    self.num_records += 1
                    break
                elif dominated[0] == nrecords:
                    fvals = None
                else:
                    ranked = True
                    record.rank = cur_rank
                    front.append(record)
                    self.num_records += 1
                    dominated = sorted(dominated, reverse = True)
                    for i in dominated:
                        dominated_record = front[i]
                        front.remove(front[i])
                        self.num_records -= 1
                        self.add(dominated_record, cur_rank)
                    break
                cur_rank += 1

            if not ranked:
                record.rank = len(self.fronts) + 1
                self.fronts.append([record])
                self.num_records += 1

        else:
            self.fronts.append([record])
            self.num_records += 1
            record.rank = 1

        if self.num_records > self.size:
            if len(self.fronts[-1]) == 1:
                index = 0
            else:
                index = int(np.random.choice(len(self.fronts[-1]), 1, replace=False))
            self.fronts[-1].remove(self.fronts[-1][index])

            if not self.fronts[-1]:
                self.fronts.remove(self.fronts[-1])
            self.num_records -= 1


class SimpleFrontArchive:
    def __init__(self, size = None):
        self.front = []
        self.size = size
        if self.size is None:
            self.size = 10000
        self.num_records = 0

    def reset(self):
        self.front = []
        self.num_records = 0

    def add(self, record):
        improvement = False
        if self.front:
            front = self.front
            fvals = [rec.fx for rec in front]
            nrecords = len(fvals)
            nondominated = list(range(nrecords))
            dominated = []
            fvals.append(record.fx)
            fvals = np.asarray(fvals)
            (nondominated, dominated) = ND_add(np.transpose(fvals), nondominated, dominated)
            if not dominated:
                improvement = True
                front.append(record)
                self.num_records += 1
            elif dominated[0] == nrecords:
                fvals = None
            else:
                improvement = True
                front.append(record)
                self.num_records += 1
                dominated = sorted(dominated, reverse = True)
                for i in dominated:
                    dominated_record = front[i]
                    front.remove(front[i])
                    self.num_records -= 1

        else:
            self.front.append(record)
            self.num_records += 1
            improvement = True

        if self.num_records > self.size:
            index = int(np.random.choice(self.num_records, 1, replace=False))
            self.front.remove(self.front[index])
            self.num_records -= 1

        return improvement