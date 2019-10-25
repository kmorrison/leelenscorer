import argparse
import asyncio
from asyncio import IncompleteReadError
import time

import chess
import chess.engine

import encoding
import rescore_logic



async def main(args):
    options = {
        "WeightsFile": args.path_to_weights,
        "Threads": 1,
        "MinibatchSize": 1,
        "ScoreType": "Q",
        "Backend": args.backend,
        "BackendOptions": f'gpu={args.gpu_id}',
    }
    command = args.path_to_rescore_engine_binary
    if args.dry_run:
        engine = None
    else:
        engine = chess.engine.SimpleEngine.popen_uci(command, timeout=20)
        engine.configure(options)

    reader, writer = await asyncio.open_connection(
        args.host,
        args.port,
    )

    encoding.write_payload(writer, [b'ready'])
    await writer.drain()

    encoding.write_payload(writer, [str(args.chunk_size).encode()])
    await writer.drain()

    while True:
        files_to_score = []
        for _ in range(args.chunk_size):
            try:
                new_file = await reader.readuntil(encoding.SEP)
            except IncompleteReadError:
                print('server sent eof, probably done')
                break

            new_file = encoding.remove_sep(new_file)
            if not new_file:
                break
            files_to_score.append(new_file)

        if not files_to_score:
            print('no files to score, exiting')
            break

        scored_files = []
        start = time.time()
        for file in files_to_score:
            if args.dry_run:
                compressed_unscored_game = rescore_logic.score_file(
                    file,
                    None,
                )
                scored_files.append(compressed_unscored_game)
            else:
                compressed_scored_game = rescore_logic.score_file(
                    file,
                    engine,
                )
                scored_files.append(compressed_scored_game)
        time_elapsed = time.time() - start

        print(f'finished scoring {len(scored_files)} files in {time_elapsed} seconds, {len(scored_files) / time_elapsed} files-per-second')
        encoding.write_payload(writer, scored_files)
        await writer.drain()

    writer.close()
    await writer.wait_closed()
    try:
        engine.quit()
    except AttributeError:
        assert args.dry_run
    finally:
        print("Done")

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', dest='gpu_id', default='0')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=10)
    parser.add_argument(
        '--engine-path',
        dest='path_to_rescore_engine_binary',
        type=str,
        help='path to the UCI binary client will run to rescore the games'
    )
    parser.add_argument(
        '--weights-path',
        dest='path_to_weights',
        type=str,
        help='path to NN weights that will be provided to engine-binary'
    )
    parser.add_argument(
        '--backend',
        dest='backend',
        type=str,
        default='cudnn',
        help='backend type to pass to the scoring engine',
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
    args = parser.parse_args()
    asyncio.run(main(args))
