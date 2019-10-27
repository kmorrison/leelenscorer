import struct
import gzip
import itertools
from collections import namedtuple

import chess
import chess.engine
import np

import constants
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
    new_board_piece_map = new_board.piece_map()
    for legal_move in current_board.legal_moves:
        current_board.push(legal_move)
        if new_board_piece_map == current_board.piece_map():
            move = legal_move.uci()
            current_board.pop()
            return move
        current_board.pop()

    else:
        print(f"Couldn't infer next move from planes, board {current_board.fen()} planes {new_board.fen()}")
        return None


async def score_move(engine, board, move_encoding):
    if engine is None:
        return struct.pack(constants.V4_STRUCT_STRING, *move_encoding)
    info = await engine.analyse(board, chess.engine.Limit(nodes=1))
    q = info["score"].relative.score(mate_score=100) / 10000

    return struct.pack(
        constants.V4_STRUCT_STRING,
        move_encoding.version,
        move_encoding.probs,
        move_encoding.planes,
        move_encoding.us_ooo,
        move_encoding.us_oo,
        move_encoding.them_ooo,
        move_encoding.them_oo,
        move_encoding.stm,
        move_encoding.rule50_count,
        move_encoding.move_count,
        move_encoding.winner,
        q,
        q,
        move_encoding.root_d,
        move_encoding.best_d,
    )


def _is_single_probability_encoding(probs):
    """Probability array has a probability associated with each move. For some games, this array is simple. It's p=1
    for the move that was played, p=0 for legal moves not played, and nan for nonlegal moves. We're trying to find if
    the probability array given is that type of array, or if many legal moves has nonzero p values. We'd like to do
    np.count_nonzero(probs), but np.nan is truthy in python and counts as nonzero, so we need to subtract the number
    of nans present.
    """
    num_nan = np.count_nonzero(~np.isnan(probs))
    num_nonzero = np.count_nonzero(probs)
    return bool(num_nan - num_nonzero == 1)


async def score_file(data, engine):
    decompressed_data = gzip.decompress(data)
    board = chess.Board()
    rescored_game = struct.pack("")
    for current_encoding, next_encoding in pairwise(parse_game(decompressed_data)):
        if len(board.piece_map()) == 5:
            break
        rescored_game += await score_move(engine, board, current_encoding)

        # Find next move that was played in game
        probs = np.frombuffer(current_encoding.probs, dtype=np.float32)
        if _is_single_probability_encoding(probs):
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
    if rescored_game and len(board.piece_map()) > 5:
        rescored_game += await score_move(engine, board, next_encoding)

    return gzip.compress(rescored_game)
