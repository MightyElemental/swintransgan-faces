import matplotlib.pyplot as plt
import os
import re
import torch

def count_parameters(model)->int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gen_graph(ylabel:str, xlabel:str, plots:list, legend:list):
    for p in plots:
        plt.plot(range(0,len(p)), p, marker=".")
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def save_graph(file:str, ylabel:str, xlabel:str, plots:list, legend:list):
    gen_graph(ylabel, xlabel, plots, legend)
    plt.savefig(file)
    plt.close()

def make_path_if_not_exist(path:str):
    if not os.path.isdir(path):
        os.makedirs(path)

def file_num(f:str):
    n = re.findall("e(\d{4,})",f)
    return int(n[0]) if n else -1, f

def get_latest_checkpoint(path:str)->str:
    if not os.path.isdir(path): return None
    files = os.listdir(path)
    # only accept checkpoint*.pt files
    files = [f for f in files if f.endswith(".pt") and f.startswith("checkpoint")]
    return max(files, key=file_num) if files else None

def get_gpu_alloc_mem()->list[int]:
    usage = []
    for device in range(torch.cuda.device_count()):
        usage.append(torch.cuda.memory_allocated(device))
    return usage

def get_gpu_cached_mem()->list[int]:
    usage = []
    for device in range(torch.cuda.device_count()):
        usage.append(torch.cuda.memory_reserved(device))
    return usage

def get_gpu_mem()->list[int]:
    return [x+y for (x,y) in zip(get_gpu_alloc_mem(), get_gpu_cached_mem())]

def print_mem_usage():
    mem_usage = get_gpu_mem()
    for i,mem in enumerate(mem_usage):
        print(f"Device {i} currently using {mem/1024**2:.1f} MiB of memory")

# get the learning rate from an optimizer
def get_learning_rate(optim:torch.optim.Optimizer) -> float:
    for param_group in optim.param_groups:
        return param_group['lr']

class CollapseError(Exception): pass

def text_to_file(text:str, filepath:str):
    with open(filepath, "w") as file:
        file.write(text)