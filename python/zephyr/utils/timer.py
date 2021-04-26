import os, time
import torch
import numpy as np

class TorchTimer:
    def __init__(self, heading = None, agg_list = None, timing=False, verbose = True):
        self.timing = timing
        if not self.timing:
            return
        self.verbose = verbose
        if(agg_list is None and heading is None):
            heading = ""
        self.agg_list = agg_list
        self.heading = heading
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not self.timing:
            return self
        self.start.record()
        self.start_cpu = time.time()
        return self

    def __exit__(self, *args):
        if not self.timing:
            return
        self.end.record()
        torch.cuda.synchronize()
        self.interval_cpu = time.time() - self.start_cpu
        self.interval = self.start.elapsed_time(self.end)/1000.0
        if(self.agg_list is not None):
            if(self.heading is not None):
                self.agg_list.append((self.heading, self.interval, self.interval_cpu))
            else:
                self.agg_list.append((self.interval, self.interval_cpu))
        if (self.heading is not None and self.verbose):
            print('{} GPU:{}, CPU:{}'.format(self.heading, self.interval, self.interval_cpu))
