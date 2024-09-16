from smart_open import open
import multiprocessing as mlp
import itertools as itrs
import re 
import pprint
import numpy as np
import timeit

INPUT_FILE,OUTPUT_FILE = "/home/mar/Desktop/1brc-main/measurements_0.txt","output.txt"
N_CPUS=mlp.cpu_count()
PATTERN = re.compile(r"^(?P<city>[^\d;]+);(?P<temp>-?\d+\.\d+)$")
BATCH_SIZE = 200_000

type Token = tuple[str,float]

def parse_tokens(line:str) -> Token:
    global PATTERN
    match_obj = PATTERN.match(line)
    assert match_obj is not None
    city = str(match_obj.group('city'))
    temp = float(match_obj.group('temp'))
    return city,temp

def accumulate_group(city_temps:tuple[str,tuple[float,...]]) -> tuple[str,float,float,float,float]:
    city,temps = city_temps
    temps = np.fromiter(temps,dtype=float)
    _min,_max,_sum,_count = min(temps),max(temps),sum(temps),len(temps)
    return city,_min,_max,_sum,_count

def main() -> None:
    acc = {}
    with open(INPUT_FILE) as f, mlp.Pool(N_CPUS) as pool, open(OUTPUT_FILE,"w") as o:
        for batch in itrs.batched(f,BATCH_SIZE):
            tokens = pool.map(parse_tokens,batch)
            sorted(tokens,key=lambda x:x[0])
            grouped_tokens: list[tuple[str,tuple[float,...]]] = \
                [(city,tuple(temps for _,temps in tokens)) for city,tokens in itrs.groupby(tokens,key=lambda x:x[0])]
            
            accumulated_tokens = pool.map(accumulate_group,grouped_tokens)
            
            for city,_min,_max,_sum,_count in accumulated_tokens:
                if city in acc:
                    acc[city] = (min(_min,acc[city][0]),max(_max,acc[city][1]),_sum+acc[city][2] ,_count+acc[city][3])
                else:
                    acc[city] = (_min,_max,_sum,_count)
            
        o.write(str(pprint.pformat(acc)))



t = timeit.timeit(lambda:main(),number=10)
print(t)