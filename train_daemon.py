import os
import time


log_root = './log'
checkpoint_root = './checkpoint'

def submmit_training(full_command, trial):
    print("\n"+"="*40)
    print("[Command]")
    print(f"{full_command}\n")

    print("[Infor]")
    print(f"Submitting a new trainning program at {trial}th trail...\n")
    flag = os.system(full_command)
    trial += 1
    return flag, trial

base_command = 'python -u ./train.py'
command_ls = [base_command, ]


trial = 0
flag = 1
try_interval = 0.5 #  should be non-zero, OR the daemon itself couldn't be shut down by CTRL+C

while flag:
    # command compile
    command_ls = [base_command, ]
    
    resume_command = f'--resume {checkpoint_root}/latest.pth'
    command_ls.append(resume_command)

    log_command = f'> {log_root}/log_{trial}.txt'
    command_ls.append(log_command)

    full_command = " \\\n".join(command_ls)

    # training submit
    time.sleep(try_interval)
    flag, trial = submmit_training(full_command, trial)

print("training over, shut daemon")