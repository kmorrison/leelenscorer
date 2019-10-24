import argparse
import asyncio
import struct
import gzip
import itertools
from collections import namedtuple
from asyncio import IncompleteReadError

import chess
import chess.engine
import np

import constants
import encoding
from util import pairwise

V4Encoding = namedtuple(
    'V4Encoding',
    [
        'version',
        'probs',
        'planes',
        'us_ooo',
        'us_oo',
        'them_ooo',
        'them_oo',
        'stm',
        'rule50_count',
        'move_count',
        'winner',
        'root_q',
        'best_q',
        'root_d',
        'best_d',
    ]
)

def convert_planes(planes):
    retval = []
    for idx in range(0, len(planes), 8):
        pl = planes[idx:idx + 8]
        a = np.fromstring(pl, dtype=np.uint8).reshape(8, 1)
        retval.append(np.unpackbits(a, 1))
    return retval


def new_board_from_planes(planes):
    new_board = chess.Board.empty()
    piece_maps = convert_planes(planes)[:len(constants.PIECES)]

    for i, piece in enumerate(constants.PIECES):
        for (j, k) in itertools.product(range(8), range(8)):
            if piece_maps[i][k][j]:
                new_board.set_piece_at(chess.square(j, k), chess.Piece.from_symbol(piece))
    return new_board


def parse_game(data):
    for chunk in read_chunks(data, constants.V4_BYTES):
        move_encoding = V4Encoding(*struct.unpack(constants.V4_STRUCT_STRING, chunk))
        yield move_encoding


def read_chunks(data, length):
    for i in range(0, len(data), length):
        yield data[i:i + length]

def set_castling(board, move_encoding):
    castle = []
    if move_encoding.us_oo:
        castle.append('K')
    if move_encoding.us_ooo:
        castle.append('Q')
    if move_encoding.them_oo:
        castle.append('k')
    if move_encoding.them_ooo:
        castle.append('q')
    castling_fen = "".join(castle)
    board.set_castling_fen(castling_fen)


def _infer_move_from_planes_and_current_board(planes, current_board):
    new_board = new_board_from_planes(planes)
    # Because `board` gets mirrored after move-push, it is always white to move. When the game is parsed from
    # planes, white is always to move. Therefore to anticipate the next board, we must change who is to move
    # on next board.
    new_board = new_board.mirror()
    for legal_move in current_board.legal_moves:
        current_board.push(legal_move)
        if current_board.piece_map() == new_board.piece_map():
            move = legal_move.uci()
            current_board.pop()
            return move
        current_board.pop()

    else:
        print(f"Couldn't infer next move from planes, board {current_board.fen()} planes {new_board.fen()}")
        return None


def score_file(data, engine):
    decompressed_data = gzip.decompress(data)
    board = chess.Board()
    rescored_game = struct.pack("")
    for current_encoding, next_encoding in pairwise(parse_game(decompressed_data)):
        if len(board.piece_map()) == 5:
            break
        if engine is not None:
            info = engine.analyse(board, chess.engine.Limit(nodes=1))
            q = info["score"].relative.score(mate_score=100) / 10000

            rescored_game += struct.pack(
                constants.V4_STRUCT_STRING,
                current_encoding.version,
                current_encoding.probs,
                current_encoding.planes,
                current_encoding.us_ooo,
                current_encoding.us_oo,
                current_encoding.them_ooo,
                current_encoding.them_oo,
                current_encoding.stm,
                current_encoding.rule50_count,
                current_encoding.move_count,
                current_encoding.winner,
                q,
                q,
                current_encoding.root_d,
                current_encoding.best_d,
            )
        else:
            rescored_game += struct.pack(constants.V4_STRUCT_STRING, *current_encoding)

        # Find next move that was played in game
        probs = np.frombuffer(current_encoding.probs, dtype=np.float32)
        if sum(element > 0 for element in probs) == 1:
            move = constants.MOVES[np.nanargmax(probs)]
        else:
            move = _infer_move_from_planes_and_current_board(next_encoding.planes, board)
            assert move is not None, "Couldn't infer move, failing"

        # Clean move to fit python-chess data expectations
        if move[1] == "7" and move[3] == "8" and board.piece_type_at(
                chess.SQUARE_NAMES.index(move[0:2])) == chess.PAWN and len(move) == 4:
            move += "n"
        if move is "e1h1" and board.piece_type_at(chess.E1) == chess.KING:
            move = "e1g1"
        if move is "e1a1" and board.piece_type_at(chess.E1) == chess.KING:
            move = "e1c1"
        m = chess.Move.from_uci(move)

        board.push(m)
        board = board.mirror()

    # This is a super ugly hack to solve the off-by-one problem iterating through pairwise gives me, just to get this thing working.
    if len(board.piece_map()) != 5:
        if engine is not None:
            info = engine.analyse(board, chess.engine.Limit(nodes=1))
            q = info["score"].relative.score(mate_score=100) / 10000

            rescored_game += struct.pack(
                constants.V4_STRUCT_STRING,
                next_encoding.version,
                next_encoding.probs,
                next_encoding.planes,
                next_encoding.us_ooo,
                next_encoding.us_oo,
                next_encoding.them_ooo,
                next_encoding.them_oo,
                next_encoding.stm,
                next_encoding.rule50_count,
                next_encoding.move_count,
                next_encoding.winner,
                q,
                q,
                next_encoding.root_d,
                next_encoding.best_d,
            )
        else:
            rescored_game += struct.pack(constants.V4_STRUCT_STRING, *next_encoding)
    return gzip.compress(rescored_game)


async def main(args):
    options = {
        "WeightsFile": args.path_to_weights,
        "Threads": 2,
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
        for file in files_to_score:
            if args.dry_run:
                compressed_unscored_game = score_file(
                    file,
                    None,
                )
                scored_files.append(compressed_unscored_game)
            else:
                compressed_scored_game = score_file(
                    file,
                    engine,
                )
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
