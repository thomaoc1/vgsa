import psutil
import torch


def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info().rss / 1024**3
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    gpu_reserved = torch.cuda.memory_reserved() / 1024**3

    return mem_info, gpu_mem, gpu_reserved


def print_memory_usage(mem_info, gpu_mem, gpu_reserved):
    print(f"CPU Memory: {mem_info:.2f} GB | GPU Memory: {gpu_mem:.2f} GB (Reserved: {gpu_reserved:.2f} GB)", flush=True)
