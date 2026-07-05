import subprocess
import time
import torch

def get_gpu_temperature():
    """Queries the NVIDIA GPU temperature using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except Exception:
        return None
    
def check_and_cooldown_gpu(max_temp=75, cooldown_temp=60):
    """Checks the GPU temperature and pauses training if it exceeds the limit."""
    temp = get_gpu_temperature()
    if temp is not None and temp >= max_temp:
        print(f"\n[Thermal Guard] GPU Temperature reached {temp}°C (limit: {max_temp}°C). Pausing training to cooldown...")
        while temp is not None and temp > cooldown_temp:
            time.sleep(10)
            temp = get_gpu_temperature()
            print(f"  [Thermal Guard] Cooling down -- Current temp: {temp}°C / Target: {cooldown_temp}°C")
        print(f"\n[Thermal Guard] GPU cooled down to {temp}°C. Resuming training.")

def get_allocated_memory(device:str):
    if device == 'cuda':
        return torch.cuda.memory_allocated(device=torch.cuda.current_device()) / (1024 * 1024)
        
def check_vram_limit(vram_limit, device):
    """Warns or clears cache if allocated memory exceeds the limit."""
    if device.type == "cuda":
        allocated = get_allocated_memory(device.type)
        
        if allocated > vram_limit:
            print(f"\n[Memory Guard] Allocated [{allocated:.1f}]MB memory exceeds limit ({vram_limit} MB). Emptying cache...")
            torch.cuda.empty_cache()
            
if __name__ == '__main__':
    print(get_allocated_memory('cuda'))