import aiofiles
import argparse
import asyncio
import os

from encoding import write_payload
import encoding


def all_gzipped_files(scan_iter, filter_text, output_dir, input_dir, resume_mode):
    for (dir, _, filenames) in scan_iter:
        for filename in filenames:
            if filter_text and filter_text not in filename and filter_text not in dir:
                continue
            if not filename.endswith('.gz'):
                continue
            filename_to_load = os.path.join(dir, filename)
            if resume_mode:
                outpath, _ = _get_full_output_filename(output_dir, input_dir, filename_to_load)
                try:
                    os.stat(outpath + os.sep + filename)
                    continue
                except:
                    pass
            yield filename_to_load


def _get_full_output_filename(output_dir, input_dir, filepath):
    if not input_dir.endswith(os.sep):
        input_dir += os.sep
    relative_filepath = filepath[len(input_dir):]
    path = os.path.normpath(relative_filepath)
    path_parts = path.split(os.sep)
    dir_parts = [output_dir] + path_parts[:len(path_parts) - 1]
    filename = path_parts[-1]
    return os.sep.join(dir_parts), filename


async def load_files(filenames):
    files = []
    for filename in filenames:
        async with aiofiles.open(filename, 'rb') as f:
            bytestream = await f.read()
            files.append(bytestream)
    return files


async def write_files_to_disk(output_dir, input_dir, filepaths, files):
    for path, file in zip(filepaths, files):
        if not file:
            #TODO: some warning
            continue

        full_out_directory, filename = _get_full_output_filename(output_dir, input_dir, path)
        os.makedirs(full_out_directory, exist_ok=True)

        async with aiofiles.open(full_out_directory + os.sep + filename, 'wb') as f:
            await f.write(file)
            await f.close()


class DirectoryQueue:
    def __init__(self, input_dir, output_dir, filter_text, resume_mode):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filter_text = filter_text
        self.scan_iter = all_gzipped_files(os.walk(input_dir), filter_text, output_dir, input_dir, resume_mode)
        self.resume_mode = resume_mode

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
            await write_files_to_disk(self.output_dir, self.input_dir, filenames, file_outputs)


async def main(args):
    directory_queue = DirectoryQueue(
        args.input_folder,
        args.output_folder,
        args.filter_text,
        args.resume_mode
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
        '--filter-text',
        dest='filter_text',
        default='',
        help='If passed, only games/folders with filter-text will be considered'
    )
    parser.add_argument(
        '--resume-mode',
        dest='resume_mode',
        type=bool,
        default=False,
        help='Pass this if in you expect a lot of work to already be done in output_dir, will turn on checking output_dir first before yielding files to clients'
    )
    args = parser.parse_args()
    asyncio.run(main(args))
