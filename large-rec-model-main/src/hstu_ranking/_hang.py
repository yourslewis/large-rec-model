import torch.multiprocessing as mp
import torch
import os
import sys
import time
from datetime import datetime

World_Size=torch.cuda.device_count()

def fn(local_rank):
    torch.cuda.set_device(local_rank)
    i = 0
    while True:
        a=torch.rand([1],device="cuda")
        T1 = time.time()
        while True:
            a=torch.rand([1],device="cuda")
            T2 = time.time()
            if (T2 - T1) > 60:
                break
        gpu_util = gpu_info()
        if min(gpu_util) > 40:
            break
        if i % 30 == 0:
            gpu_util_str = ','.join([str(i)+'%' for i in gpu_util])
            print('\r' + str(datetime.now()) + ' GPU is idle and fn runnig. gpu_util: ' + gpu_util_str)
            #sys.stdout.flush()
        i += 1
    torch.cuda.empty_cache()
    print('GPU is busy now. Stop fn!')
    return

def narrow_setup():
    gpu_util = gpu_info()
    while max(gpu_util) > 5:  # set waiting condition
        gpu_util = gpu_info()
        gpu_util_str = ','.join([str(i)+'%' for i in gpu_util])
        print('\r' + str(datetime.now()) + ' GPU is not idle and fn not runnig. gpu_util: ' + gpu_util_str)
        time.sleep(60 * 30) #30 min
    gpu_util = gpu_info()
    gpu_util_str = ','.join([str(i)+'%' for i in gpu_util])
    print('GPU is Maybe idle now. Waiting 3 mins to check again !' + gpu_util_str)
    time.sleep(60 * 3)
    gpu_util = gpu_info()
    if max(gpu_util) > 5:
        print(f'GPU is busy now. ')
        return
    gpu_util_str = ','.join([str(i)+'%' for i in gpu_util])
    print('GPU is idle now. Waiting 5 mins again !' + gpu_util_str)
    time.sleep(60 * 5)
    gpu_util = gpu_info()
    if max(gpu_util) > 5:
        print('GPU is busy now. ')
        return
    gpu_util_str = ','.join([str(i)+'%' for i in gpu_util])
    print('GPU is idle now. Waiting final 2 mins !' + gpu_util_str)
    time.sleep(60 * 2)
    gpu_util = gpu_info()
    if max(gpu_util) > 5:
        print('GPU is busy now. ')
        return
    gpu_util_str = ','.join([str(i)+'%' for i in gpu_util])
    print('GPU is idle now. Run fn !' + gpu_util_str)
    mp.spawn(fn,nprocs=World_Size,join=True)
    return

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_util = [int(i.split('%')[0].strip()) for i in gpu_status if '%' in i and 'W ' not in i]
    if not gpu_util:
        gpu_status = os.popen('rocm-smi | grep %').read().split('\n')
        gpu_util = [int((i.split('%'))[-2].strip()) for i in gpu_status if 'Mhz' in i]
    return gpu_util

if __name__=="__main__":
    while True:
        narrow_setup()
        