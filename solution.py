import multiprocessing as mlp
import itertools as iters
import functools as fnct
from dataclasses import dataclass
from multiprocessing.pool import MapResult
from typing import Optional
import pprint
import numpy as np
from numpy.typing import NDArray
import time
from pathlib import Path

@dataclass
class Status:
    _min : float
    _max : float
    _sum : float
    _count :int
    @property
    def min(self):
        return self._min
    @property
    def max(self):
        return self._max
    @property
    def sum(self):
        return self._sum
    @property
    def count(self):
        return self._count


INPUT_PATH = ""
OUTPUT_PATH = ""
BATCH_SIZE = 100000 # number of lines to read simultaneously
CPUS = 4 # specify the number of the cpu cores to use
MAX_JOBS = 100 # the max number of jobs which can be queued
TIMEOUT = 0.5  # the time for the main thread to wait until the number of jobs in the queue < MAX_JOBS

INPUT_PATH = INPUT_PATH.strip()
OUTPUT_PATH = OUTPUT_PATH.strip()
BATCH_SIZE = int(BATCH_SIZE)
CPUS = int(CPUS)
MAX_JOBS = int(MAX_JOBS)
TIMEOUT = float(TIMEOUT)

assert INPUT_PATH != OUTPUT_PATH
assert INPUT_PATH != "" and OUTPUT_PATH != ""
assert Path(INPUT_PATH).exists()
assert BATCH_SIZE > 1
assert CPUS > 0
assert MAX_JOBS > 0
assert TIMEOUT > 0

def parse(line:str) -> Optional[tuple[str,float]]:
    """
    :param line: a single line of the form <city name>;<temperature as a float>\n
    :return: if the line can't be parsed it return None, if the line is parseable it returns the city name, temperature
    """
    line=line.strip()
    try:
        city,temp,*_ = line.split(";")
        city,temp = city.strip(),float(temp.strip())
        return city,temp
    except ValueError:...
    
def consume(batch:tuple[str,...]) -> dict[str,Status]:
    """
    :param batch: a tuple of <BATCH_SIZE> string, each one represent a line of the form <city name>;<temperature as a float>\n
    :return: a dictionary that accumulate the temperatures of the same city
    """
    parsed_lines = (x for x in (parse(x) for x in batch) if x is not None)
    grouped_lines = iters.groupby(sorted(parsed_lines,key=lambda x:x[0]),key=lambda x:x[0])
    acc:dict[str,Status]= {}
    for city,temps in grouped_lines:
        temps: NDArray[float] = np.array(list(temps))
        _min,_max,_sum,_count = min(temps),max(temps),sum(temps),len(temps)
        if city not in acc:
            acc[city] = Status(_min,_max,_sum,_count)
        else:
            old = acc[city]
            acc[city] = Status(min(old.min, _min), max(old.max, _max), old.sum + _sum, old.count + _count)
    return acc

def accumulate(x:dict[str,Status],y:dict[str,Status]) -> dict[str,Status]:
    """
    :param x: A number of measurements, where the key is the city name
    :param y: A number of measurements, where the key is the city name
    :return:  a dictionary that combine x and y
    """
    for city,temps in y.items():
        if city not in x:
            x[city] = temps
            continue
        x[city] = Status(
            min(x[city].min, temps.min),max(x[city].max,temps.max),
            x[city].sum + temps.sum, x[city].count + temps.count)
    return x 

def main():
    jobs:list[MapResult[dict[str,Status]]] = []
    acc = {}
    with open(INPUT_PATH) as f, mlp.Pool(CPUS) as pool:
        for batch in iters.batched(f,BATCH_SIZE):
            if len(jobs) > MAX_JOBS:time.sleep(TIMEOUT)
            jobs.append(pool.map_async(consume,iters.batched(batch,1+BATCH_SIZE//CPUS)))
            ready_jobs = (j for j in jobs if j.ready())    
            jobs = [j for j in jobs if not j.ready()]
            ready_jobs = (jj for j in ready_jobs for jj in j.get())
            acc = fnct.reduce(accumulate,ready_jobs,acc)
            
    with open(OUTPUT_PATH,"wt") as f:
        f.write(pprint.pformat(acc))

if __name__ == "__main__":
    main()