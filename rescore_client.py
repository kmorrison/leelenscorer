import argparse
import asyncio
import struct
import gzip
import hashlib
from asyncio import IncompleteReadError

import chess
import chess.engine
import np

import constants
import encoding


def npmax(l):
    max_idx = np.nanargmax(l)
    return max_idx

def parseV4(data):
    (version, probs_, planes_, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d) = struct.unpack(
        constants.V4_STRUCT_STRING,
        data
    )
    probs = np.frombuffer(probs_, dtype=np.float32)
    return (constants.MOVES[npmax(probs)] , data)


def read_chunks(f, length):
    while True:
        data = f.read(length)
        if not data: break
        yield data


def score_file(data, engine):
    decompressed_data = gzip.decompress(data)
    game = [parseV4(chunk) for chunk in read_chunks(decompressed_data, constants.V4_BYTES)]
    board = chess.Board()
    rescored_game = struct.pack("")
    for record in game:
        if len(board.piece_map()) == 5:
            break
        move = record[0]
        i = 0
        if move[1] == "7" and move[3] == "8" and board.piece_type_at(
                chess.SQUARE_NAMES.index(move[0:2])) == chess.PAWN and len(move) == 4:
            move += "n"
        if move is "e1h1" and board.piece_type_at(chess.E1) == chess.KING:
            move = "e1g1"
        if move is "e1a1" and board.piece_type_at(chess.E1) == chess.KING:
            move = "e1c1"
        m = chess.Move.from_uci(move)
        info = engine.analyse(board, chess.engine.Limit(nodes=1))
        q = info["score"].relative.score(mate_score=100) / 10000
        (version, probs_, planes_, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q,
         root_d, best_d) = struct.unpack(constants.V4_STRUCT_STRING, record[1])
        rescored_game += struct.pack(
            constants.V4_STRUCT_STRING,
            version,
            probs_,
            planes_,
            us_ooo,
            us_oo,
            them_ooo,
            them_oo,
            stm,
            rule50_count,
            move_count,
            winner,
            q,
            q,
            root_d,
            best_d,
        )
        board.push(m)
        board = board.mirror()
        i += 1
    return gzip.compress(rescored_game)


async def main(args):
    options = {
        "WeightsFile": args.path_to_weights,
        "Threads": 2,
        "ScoreType": "Q",
        "Backend": "cudnn",
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
        for file in files_to_score:
            if args.dry_run:
                scored_files.append(file)
            else:
                compressed_scored_game = score_file(engine, file)
                scored_files.append(compressed_scored_game)

        print(f'finished scoring {len(scored_files)} files')
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
