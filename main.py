import os
import traceback
from config import get_configs
from train import run_train
from train_flow import run_train_flow

from logger import set_wandb

def main():
    P = get_configs()
    print(P, '\n')
    set_wandb(P)
    os.environ['CUDA_VISIBLE_DEVICES'] = P['gpu_num']
    print('###### Train start ######')
    if P["flow"]:
        run_train_flow(P)
    else:
        run_train(P)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
