import multiprocessing as mlp
from multiprocessing.pool import MapResult
import itertools as iters
import functools as fnct
from dataclasses import dataclass
from typing import Optional
import pprint
import numpy as np
import time


@dataclass
class Status:
    _min : float
    _max : float
    _sum : float
    _count :int


INPUT_PATH = "/home/mar/Desktop/1brc-python/measurements.txt"
OUTPUT_PATH = "output.txt"
BATCH_SIZE = 100000
CPUS = 4
MAX_JOBS = 100
TIMEOUT = 0.5
MAX_CACHE_SIZE = 1000

@fnct.lru_cache(maxsize=MAX_CACHE_SIZE)
def parse(line:str) -> Optional[tuple[str,float]]:
    line=line.strip()
    try:
        city,temp,*_ = line.split(";")
        city,temp = city.strip(),float(temp.strip())
        return city,temp
    except ValueError:...
    return None
    
def consume(batch:tuple[str,...]) -> dict[str,Status]:
    parsed_lines = (parse(x) for x in batch)
    parsed_lines = (x for x in parsed_lines if x is not None)
    grouped_lines = iters.groupby(sorted(parsed_lines,key=lambda x:x[0]),key=lambda x:x[0])
    acc:dict[str,Status]= {}
    for city,temps in grouped_lines:
        temps = np.array([temp for _,temp in temps])
        _min,_max,_sum,_count = min(temps),max(temps),sum(temps),len(temps)
        if city not in acc:
            acc[city] = Status(_min,_max,_sum,_count)
        else:
            old = acc[city]
            old._min = min(old._min,_min)
            old._max = max(old._max,_max)
            old._sum += _sum
            old._count += _count
    return acc

def accumulate(x:dict[str,Status],y:dict[str,Status]) -> dict[str,Status]:
    for city,temps in y.items():
        if city not in x:
            x[city] = temps
        else:
            x[city]._min = min(x[city]._min,temps._min)
            x[city]._max = max(x[city]._max,temps._max)
            x[city]._sum += temps._sum
            x[city]._count += temps._count
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

main()