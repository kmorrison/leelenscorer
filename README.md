# Leelenscorer

## Purpose

Collection of tools to assist in the development of the Leelenstein nets

## Setup

### Server
All you should need to run the server is a collection of games in the `.gz` format that you wish to score, as well as python3.7.
Then, install `requirements.txt` globally or in a virtualenv using `pip install -r requirements.txt`

### Client

If you're pro python, just set up a virtualenv and pip install the requirements in there. Client uses asyncio features that are new as of `python3.7`, so `python3.7` is a requirement

If you like docker instead, the client code should be all ready to go in the `kmorrison64/leelenscorer` dockerhub repo. If you're using vast.ai, just switch the container to the latest version at that repo and you're half-way there.


## Usage

### Server
The server can be run like 
`python game_server.py --input-folder=<example folder> --output-folder=<different folder>`

### Client
The client can be run in two forms, either the single client which will pull games from the server and give to one engine instance for scoring, or the `multi-client.py` which uses `nvidia-smi` to detect the number of available GPUs and attempts to use them all.

#### Examples

```
python multi_client.py --clients-per-gpu=1 --chunk-size=10 --engine-path=<where to engine> --weights-path=<where to weights> --host=<route to server> --port=<server port>

python rescore_client.py --chunk-size=10 --engine-path=<where to engine> --weights-path=<where to weights> --host=<route to server> --port=<server port>

#To just parrot back the input files and not score anything, use --dry-run option
python rescore_client.py --dry-run=True --chunk-size=10 --engine-path=<where to engine> --weights-path=<where to weights> --host=<route to server> --port=<server port>

python3 multi_client.py --host=localhost --port=8888 --backend=cudnn-fp16 --clients-per-gpu=5 --engine-path=/root/binaries/lc0 --weights-path=/root/binaries/ls-n11-1.pb.gz --chunk-size=10
```

## Gotchas

- the client-server protocol is custom (seemed like a good idea at the time :P) and uses four newlines as a separator between messages. If any files you're transmitting have `b'\n\n\n\n'` in them, we're gonna have a bad time
- if a client fails to score a set of games it is handed for some reason, there is no mechanism for requeueing them and handing them to another client.
- when running locally via docker you will have to set `--network="host"` as an arg to docker run, and pass `--host="host.docker.internal"` to your client script
