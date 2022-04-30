import time
import random
import argparse


def save_checkpoint(epoch):
    with open(f'./checkpoint/latest.pth', 'w') as f:
        f.write(str(epoch))
    return

def read_checkpoint(checkpoint):
    with open(checkpoint, 'r') as f:
        epoch = f.read()
    return int(epoch)


parser = argparse.ArgumentParser(description='Nothing')
parser.add_argument(
    '--resume',
    type=str,
    default=None
)
args = parser.parse_args()


N = 400
epoch = 0
t = 0.1
if args.resume is not None:
    epoch = read_checkpoint(args.resume)

while epoch < N:
    num = random.randint(1, 10)
    if num != 4:
        print(f"Epoch [{epoch}]: got {num}, keep training.")
        save_checkpoint(epoch)
    else:
        raise OSError("An error raised, fuc...")

    # wait
    time.sleep(t)
    epoch += 1

print(f"Success training for {epoch} epoch!!")
