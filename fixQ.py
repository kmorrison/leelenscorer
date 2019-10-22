import argparse
import struct
import gzip
import os

import chess
import chess.engine
import np

import constants


def npmax(l):
    max_idx = np.nanargmax(l)
    return max_idx

def parseV4(data):
    (version, probs_, planes_, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d) = struct.unpack(
        constants.V4_STRUCT_STRING,
        data
    )
    probs = np.frombuffer(probs_, dtype=np.float32)
    #return MOVES[np.unravel_index(np.nanargmax(probs, axis=None), probs.shape)[0]]
    #return MOVES[npmax(probs)]
    #s = struct.pack(V4_STRUCT_STRING, version, probs_, planes_, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d)
    return (constants.MOVES[npmax(probs)] , data)


def read_chunks(f, length):
    while True:
        data = f.read(length)
        if not data: break
        yield data


def main(args):
    #folder = r"D:\lczero-training\tf\data\training-run2-tcec-rescored"
    #folder = r"D:\lczero-training\tf\data\training-run2-4040-rescored"
    options={
        "WeightsFile": '/root/binaries/ls-n11-1.pb.gz',
        "Threads": 2, 
        "ScoreType": "Q", 
        "Backend": "cudnn",
        "BackendOptions": f'gpu={args.gpu_id}',
    }
    command = '/root/binaries/lc0'
    engineLocation = r"I:\driveI\lc0-v0.21.1-windows-cuda\lc0TPEarly.exe"
    #engine = chess.engine.SimpleEngine.popen_uci(engineLocation, timeout=20)
    engine = chess.engine.SimpleEngine.popen_uci(command, timeout=20)
    engine.configure(options)

    for filenumber in range(args.offset + args.proc_id, args.last_game, args.num_processes):
        filename = f'game_{filenumber:06}.gz'
        # use path.join because it will behave correctly on windows and linux
        full_path = os.path.join(args.input_folder, filename)
        try:
            os.stat(full_path)
        except Exception:
            continue
        if filenumber % args.num_processes != args.proc_id:
            continue

        print(filename)
        with gzip.open(full_path, 'rb') as file:
            game = [parseV4(chunk) for chunk in read_chunks(file, constants.V4_BYTES)]
            board = chess.Board()
            rescoredGame = struct.pack("")
            for record in game:
                if len(board.piece_map()) == 5:
                    break
                move = record[0]
                i=0
                if move[1] == "7" and move[3] == "8" and board.piece_type_at(chess.SQUARE_NAMES.index(move[0:2])) == chess.PAWN and len(move)==4:
                    move += "n"
                if move is "e1h1" and board.piece_type_at(chess.E1) == chess.KING:
                    move = "e1g1"
                if move is "e1a1" and board.piece_type_at(chess.E1) == chess.KING:
                    move = "e1c1"
                m = chess.Move.from_uci(move)
                #print(move,m)
                #if i%2 == 1:
                #    board = board.mirror()
                info = engine.analyse(board, chess.engine.Limit(nodes=1))
                #if i%2 == 1:
                #    board = board.mirror()
                q=info["score"].relative.score(mate_score=100) / 10000
                (version, probs_, planes_, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d) = struct.unpack(constants.V4_STRUCT_STRING, record[1])
                rescoredGame += struct.pack(
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
                #print(struct.pack(V4_STRUCT_STRING, record[1]))
                board.push(m)
                board = board.mirror()    
                i += 1
            with gzip.open(args.output_folder + filename, 'wb') as file:
                file.write(rescoredGame)
    try:
        engine.quit()
    finally:
        print("Done")

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', dest='gpu_id', default='0')
    parser.add_argument('--proc-id', dest='proc_id', type=int, default=0, help="zero indexed, which process of num_processes are you")
    parser.add_argument('--num-processes', dest='num_processes', type=int, default=1, help="how many total processes are doing this indexing")
    parser.add_argument('--first-game', dest='first_game', type=int)
    parser.add_argument('--chunk-size', dest='chunk_size', type=int)
    parser.add_argument('--offset', dest='offset', type=int, default=0)
    parser.add_argument('--last-game', dest='last_game', type=int, default=1000000)
    parser.add_argument('--input-folder', dest='input_folder', default='/root/404/test_games')
    parser.add_argument('--output-folder', dest='output_folder', default='/root/404/rescored_test/')
    args = parser.parse_args()
    main(args)
