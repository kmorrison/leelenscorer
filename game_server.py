import aiofiles
import argparse
import asyncio
import os
import hashlib

from encoding import write_payload
import encoding

def read_chunks(f, length):
    while True:
        data = f.read(length)
        if not data:
            break
        yield data


def all_gzipped_files(scan_iter):
    for (dir, _, filenames) in scan_iter:
        for filename in filenames:
            if not filename.endswith('.gz'):
                continue
            yield os.path.join(dir, filename)


async def load_files(filenames):
    files = []
    for filename in filenames:
        async with aiofiles.open(filename, 'rb') as f:
            bytestream = await f.read()
            files.append(bytestream)
    return files


async def write_files_to_disk(output_dir, filepaths, files):
    for path, file in zip(filepaths, files):
        if not file:
            #TODO: some warning
            continue
        path = os.path.normpath(path)
        path_parts = path.split(os.sep)
        dir_parts = [output_dir] + path_parts[:len(path_parts) - 1]
        filename = path_parts[-1]
        os.makedirs(os.sep.join(dir_parts), exist_ok=True)

        async with aiofiles.open(os.sep.join(dir_parts + [filename]), 'wb') as f:
            await f.write(file)
            await f.close()


class DirectoryQueue:
    def __init__(self, input_dir, output_dir, default_chunk_size):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.default_chunk_size = default_chunk_size
        self.scan_iter = all_gzipped_files(os.walk(input_dir))

    async def handle_new_client(self, reader, writer):
        # Check it's a valid connection and client is ready
        start_message = await reader.readuntil(encoding.SEP)
        assert encoding.remove_sep(start_message) == b'ready', start_message

        client_set_chunk_size_message = encoding.remove_sep(await reader.readuntil(encoding.SEP))
        client_set_chunk_size = int(client_set_chunk_size_message.decode())

        # Find some files to give the client
        # TODO: If the client fails to process files, put them back in a queue?
        print('new client')
        effective_chunk_size = client_set_chunk_size

        while True:
            filenames = []
            for i, filepath in enumerate(self.scan_iter):
                filenames.append(filepath)
                if i == effective_chunk_size - 1:
                    break

            # Current files have been exhausted, good job
            if not filenames:
                print('closing conn because all done')
                writer.write_eof()
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            # Read out the files to pass to client
            files = await load_files(filenames)
            write_payload(writer, files)
            if len(filenames) < effective_chunk_size:
                # Loaded less than CHUNK_SIZE files, means we're out of files to load, aka we're done!
                writer.write_eof()
            await writer.drain()

            file_outputs = []
            for _ in filenames:
                file_output = encoding.remove_sep(await reader.readuntil(encoding.SEP))
                if not file_output:
                    break
                file_outputs.append(file_output)
            # TODO: Do some sanity checking on these files to make sure they're roughly the right size.

            print(f'persisting {len(file_outputs)} files')
            await write_files_to_disk(self.output_dir, filenames, file_outputs)


async def main(args):
    directory_queue = DirectoryQueue(
        args.input_folder,
        args.output_folder,
        args.chunk_size
    )
    server = await asyncio.start_server(
        directory_queue.handle_new_client,
        '127.0.0.1',
        8888,
    )

    async with server:
        await server.serve_forever()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-folder',
        dest='input_folder',
        default='/root/404/test_games',
        help='Folder from which to serve the games. Games are expected to be in .gz format'
    )
    parser.add_argument(
        '--output-folder',
        dest='output_folder',
        default='/root/404/rescored_test/',
        help='Folder to which to write the games. Games will be rescored and written to output folder on the same '
             'name. DO NOT MAKE THE INPUT FOLDER THE SAME AS THE OUTPUT '
    )
    parser.add_argument(
        '--chunk-size',
        dest='chunk_size',
        default=10,
        type=int,
        help='How many games to hand at once to a client (can be overriden by client)'
    )
    args = parser.parse_args()
    asyncio.run(main(args))
