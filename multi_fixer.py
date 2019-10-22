import argparse
import datetime
import subprocess
import time

def spawn_fixers(num_gpus, input_dir, output_dir, offset, last_game):
    start = time.time()
    subprocs = []
    for i in range(num_gpus):
        for j in [0, 1]:
            process_command = [
                'python3',
                'fixQ.py',
                f'--offset={offset}',
                f'--last-game={last_game}',
                f'--input-folder={input_dir}',
                f'--output-folder={output_dir}',
                f'--proc-id={(10 * j) + i}',
                f'--num-processes={num_gpus * 2}',
                f'--gpu-id={i}',
            ]
            print(process_command)
            subproc = subprocess.Popen(process_command)
            subprocs.append(subproc)
    while any([s.poll() is None for s in subprocs]):
        print(datetime.datetime.now(), [s.poll() for s in subprocs])
        time.sleep(5)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset', dest='offset', type=int, default=0)
    parser.add_argument('--input-folder', dest='input_folder', default='/root/404/test_games')
    parser.add_argument('--output-folder', dest='output_folder', default='/root/404/rescored_test/')
    parser.add_argument('--last-game', dest='last_game', type=int, default=1000000)
    args = parser.parse_args()

    output = subprocess.run(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE)
    num_gpus = len([line for line in output.stdout.decode().split('\n') if line])
    print(num_gpus)
    spawn_fixers(num_gpus, args.input_folder, args.output_folder, args.offset)
