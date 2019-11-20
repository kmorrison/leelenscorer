import math
import gzip
import struct
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


def make_bitboards(planes):
    bitboards = dict(
        white=0,
        black=0,
    )
    for piece_symbol, idx in zip(constants.PIECES, range(0, 8 * len(constants.PIECES), 8)):
        current_bitmask = 0
        pl = planes[idx:idx + 8]
        for i, byte in enumerate(pl):
            new_mask = constants.BIT_REVERSE[byte] << ((8 - i - 1) * 8)
            current_bitmask += new_mask
            if not piece_symbol.isupper():
                bitboards['white'] += new_mask
            else:
                bitboards['black'] += new_mask
        board_attr = constants.SYMBOL_TO_BOARD_ATTR[piece_symbol.upper()]
        bitboards.setdefault(
            board_attr,
            0,
        )
        bitboards[board_attr] += current_bitmask
    return bitboards


def parse_game(data):
    for chunk in read_chunks(data, constants.V4_BYTES):
        move_encoding = V4Encoding(*struct.unpack(constants.V4_STRUCT_STRING, chunk))
        yield move_encoding


def read_chunks(data, length):
    for i in range(0, len(data), length):
        yield data[i:i + length]


def board_equals_planes(board, bitboards):
    for attr in constants.BOARD_ATTRS:
        if getattr(board, attr) != bitboards[attr]:
            return False
    return bitboards['white'] == board.occupied_co[chess.WHITE] and bitboards['black'] == board.occupied_co[chess.BLACK]


def _infer_move_from_planes_and_current_board(planes, current_board):
    bitboards = make_bitboards(planes)
    # Only need the first 8 bits times 12 distinct pieces
    for legal_move in current_board.legal_moves:
        current_board.push(legal_move)
        if board_equals_planes(current_board, bitboards):
            move = legal_move.uci()
            current_board.pop()
            return move
        current_board.pop()

    else:
        print(f"Couldn't infer next move from planes, board {current_board.fen()}")
        return None


def q_and_probs_from_engine_score(infos, probs, num_nodes, board, next_move_in_game):
    boosting_nodes = math.ceil((num_nodes / 0.7) - num_nodes)
    q = None
    total_visited_nodes = 0
    move_nodes = {}
    for i, info in enumerate(infos):
        if i == 0:
            q = info["score"].relative.score(mate_score=100) / 10000
        pythonchess_move = info['pv'][0]
        move = unclean_uci_move_to_lc0(pythonchess_move.uci(), board)
        node_count = info['nodes']

        total_visited_nodes += node_count
        move_nodes[move] = node_count

        if pythonchess_move == next_move_in_game:
            move_nodes[move] += boosting_nodes
            total_visited_nodes += boosting_nodes

    if total_visited_nodes == 0:
        raise Exception("somehow no moves visited in position, crashing")

    # create writeable probability array
    probs = np.array(probs)
    for move, node_count in move_nodes.items():
        probs[constants.MOVES_LOOKUP[move]] = node_count / total_visited_nodes
    return q, probs


async def score_move(engine, board, move_encoding, probs, num_nodes, next_move_in_game):
    if engine is None:
        return struct.pack(constants.V4_STRUCT_STRING, *move_encoding)

    engine_kwargs = {}
    if num_nodes > 1:
        engine_kwargs['multipv'] = num_nodes

    infos = await engine.analyse(
        board,
        chess.engine.Limit(nodes=num_nodes),
        multipv=math.ceil(num_nodes / 2),
    )

    q, probs = q_and_probs_from_engine_score(
        infos,
        probs,
        num_nodes,
        board,
        next_move_in_game,
    )

    return struct.pack(
        constants.V4_STRUCT_STRING,
        move_encoding.version,
        probs.tobytes(),
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
        constants.MOVES_LOOKUP[unclean_uci_move_to_lc0(next_move_in_game.uci(), board)],
        move_encoding.best_d,
    )


def _is_single_probability_encoding(probs):
    """Probability array has a probability associated with each move. For some games, this array is simple. It's p=1
    for the move that was played, p=0 for legal moves not played, and nan for nonlegal moves. We're trying to find if
    the probability array given is that type of array, or if many legal moves has nonzero p values. We'd like to do
    np.count_nonzero(probs), but np.nan is truthy in python and counts as nonzero, so we need to subtract the number
    of nans present.
    """
    num_nan = np.count_nonzero(np.isnan(probs))
    num_nonzero = np.count_nonzero(probs)
    return bool(num_nonzero - num_nan == 1)


def clean_lc0_to_uci_move(move, board):
    # Clean move to fit python-chess data expectations
    if move[1] == "7" and move[3] == "8" and board.piece_type_at(
            chess.SQUARE_NAMES.index(move[0:2])) == chess.PAWN and len(move) == 4:
        return move + 'n'
    if move == "e1h1" and board.piece_type_at(chess.E1) == chess.KING:
        return "e1g1"
    if move == "e1a1" and board.piece_type_at(chess.E1) == chess.KING:
        return "e1c1"
    return move


def unclean_uci_move_to_lc0(move, board):
    if len(move) == 5 and move.endswith('n'):
        return move[:4]
    if move == "e1g1" and board.piece_type_at(chess.E1) == chess.KING:
        return "e1h1"
    if move == "e1c1" and board.piece_type_at(chess.E1) == chess.KING:
        return "e1a1"
    return move


async def score_file(data, engine, num_nodes=1):
    decompressed_data = gzip.decompress(data)
    board = chess.Board()
    rescored_game = struct.pack("")
    for current_encoding, next_encoding in pairwise(parse_game(decompressed_data)):
        if len(board.piece_map()) == 5:
            break
        probs = np.frombuffer(current_encoding.probs, dtype=np.float32)

        # Find next move that was played in game
        if _is_single_probability_encoding(probs):
            move = constants.MOVES[np.nanargmax(probs)]
        else:
            move = _infer_move_from_planes_and_current_board(next_encoding.planes, board)
            assert move is not None, "Couldn't infer move, failing"

        move = clean_lc0_to_uci_move(move, board)
        m = chess.Move.from_uci(move)

        rescored_game += await score_move(
            engine,
            board,
            current_encoding,
            probs,
            num_nodes,
            m,
        )

        board.push(m)
        board = board.mirror()

    # This is a super ugly hack to solve the off-by-one problem iterating through pairwise gives me, just to get this thing working.
    if rescored_game and len(board.piece_map()) > 5:
        rescored_game += await score_move(
            engine,
            board,
            next_encoding,
            np.frombuffer(next_encoding.probs, dtype=np.float32),
            num_nodes,
            m,
        )

    return gzip.compress(rescored_game)
