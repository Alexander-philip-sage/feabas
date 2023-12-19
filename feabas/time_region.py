import time
import logging
from collections import defaultdict
import os, csv
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NUMRANKS = COMM.Get_size()
#RANK=None
class TimeRegion():
    def __init__(self):
        self.region_time = defaultdict(lambda: 0)
        self.region_call_count= defaultdict(lambda: 0)
        self.log_summary_called = False
        logging.getLogger().setLevel(logging.INFO)
    def log_summary(self, save_csv=False):
        self.log_summary_called = True
        if RANK is None or RANK==0:
            if logging.root.level>logging.INFO:
                print(f"Warning. root log level {logging.root.level}. likely will not print time region logging")
            #print("timing information")
            logging.info("rank {}".format( RANK))
            #logging.info("regions {}".format(str(self.region_time.keys())))
            regions = list(self.region_time.keys())
            all_times = [['region_name', 'ct_calls', 'total_time','time_per_call'] ]
            if len(regions)==0:
                logging.info("timed_region: no timings to log")
            for reg in regions:
                if self.region_time[reg]>0:
                    total_time = self.region_time[reg]
                    call_count = self.region_call_count[reg]
                    all_times.append([reg, call_count,total_time,total_time/call_count])
                    logging.info(reg+":tmg: calls {} total time {} time per call {}".format(call_count, 
                        round(total_time,3), 
                        round(total_time/call_count,6)))
            if save_csv:
                try:
                    csv_path = os.path.split(logging.getLoggerClass().root.handlers[0].baseFilename)[0]
                except:
                    csv_path = os.path.join(os.path.expanduser('~'),"feabas_timings" )
                    if not os.path.exists(csv_path):
                        os.mkdir(csv_path)
                print("going to write to csv_path", csv_path)
                with open(os.path.join(csv_path, "feabas_timings.csv"), 'w') as fileobj:
                    csvwriter  = csv.writer(fileobj)
                    csvwriter.writerow(all_times)
        #print("rank", RANK, "exited logger")
        #else:
        #    logstr = "not reporting log summary rank {}".format( RANK)
        #    logging.info(logstr)
        #    print(logstr)
    def track_time(self,name: str, time_passed_s: int):
        '''times are expected in s '''
        if RANK is None or RANK==0:
            self.region_time[name] += time_passed_s
            self.region_call_count[name] +=1

time_region = TimeRegion()
