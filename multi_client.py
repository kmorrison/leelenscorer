import argparse
import datetime
import socket
import subprocess
import time

def spawn_clients(num_gpus, clients_per_gpu, chunk_size, engine, weights, host, port, dry_run, backend, client_name, num_nodes, minibatchsize):
    subprocs = []
    for i in range(num_gpus):
        for _ in range(clients_per_gpu):
            process_command = [
                'python3',
                'rescore_client.py',
                f'--gpu-id={i}',
                f'--chunk-size={chunk_size}',
                f'--engine-path={engine}',
                f'--weights-path={weights}',
                f'--host={host}',
                f'--port={port}',
                f'--backend={backend}',
                f'--client-name={client_name}',
                f'--num-nodes={num_nodes}',
                f'--minibatchsize={minibatchsize}',
            ]
            if dry_run:
                process_command.append(f'--dry-run=True')
            print(process_command)
            subproc = subprocess.Popen(process_command)
            subprocs.append(subproc)
    while any([s.poll() is None for s in subprocs]):
        print(datetime.datetime.now(), [s.poll() for s in subprocs])
        time.sleep(5)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients-per-gpu', dest='clients_per_gpu', type=int, default=2)
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=5)
    parser.add_argument(
        '--backend',
        dest='backend',
        type=str,
        default='cudnn',
        help='backend type to pass to the scoring engine',
    )
    parser.add_argument(
        '--engine-path',
        dest='path_to_rescore_engine_binary',
        type=str,
        default='/root/binaries/lc0',
        help='path to the UCI binary client will run to rescore the games'
    )
    parser.add_argument(
        '--weights-path',
        dest='path_to_weights',
        type=str,
        default='/root/binaries/ls-n11-1.pb.gz',
        help='path to NN weights that will be provided to engine-binary'
    )
    parser.add_argument(
        '--host',
        dest='host',
        type=str,
        default='localhost',
        help='host of game server'
    )
    parser.add_argument(
        '--port',
        dest='port',
        type=int,
        default='8888',
        help='port of game server'
    )
    parser.add_argument(
        '--dry-run',
        dest='dry_run',
        type=bool,
        default=False,
        help='Just parrot back the data the server sends. Useful for testing the client, not actually scoring anything'
    )
    parser.add_argument(
        '--client-name',
        dest='client_name',
        type=str,
        default=f'{socket.gethostname()}',
        help='string with which to identify your client to the server'
    )
    parser.add_argument(
        '--num-nodes',
        dest='num_nodes',
        type=int,
        default=1,
        help='number of game nodes to evaluate per move'
    )
    parser.add_argument(
        '--minibatchsize',
        dest='minibatchsize',
        type=int,
        default=1,
        help='minibatch arg to engine'
    )
    args = parser.parse_args()

    output = subprocess.run(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE)
    num_gpus = len([line for line in output.stdout.decode().split('\n') if line])
    print(num_gpus)
    spawn_clients(
        num_gpus,
        args.clients_per_gpu,
        args.chunk_size,
        args.path_to_rescore_engine_binary,
        args.path_to_weights,
        args.host,
        args.port,
        args.dry_run,
        args.backend,
        args.client_name,
        args.num_nodes,
        args.minibatchsize,
    )
