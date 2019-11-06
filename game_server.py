import aiofiles
import argparse
import asyncio
from asyncio import IncompleteReadError
from collections import deque
import datetime
import time
import os

from encoding import write_payload
import encoding


def all_gzipped_files(scan_iter, filter_text, output_dir, input_dir, resume_mode):
    for (dir, dirnames, filenames) in scan_iter:
        if filter_text and filter_text not in dir:
            continue
        for filename in filenames:
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


class ClientStats:
    def __init__(self):
        self.num_attached_clients = 0
        # Format = [(timestamp, num_processed, time_taken), (timestamp, num_processed, time_taken)...]
        self.processed_queue = deque(maxlen=100)
        self.total_processed = 0

    def compute_stats_for_client(self, last_n_seconds):
        now = datetime.datetime.now()
        cutoff_time = now - datetime.timedelta(seconds=last_n_seconds)

        num_processed_in_timeframe = 0
        for (timestamp, num_processed, time_taken) in self.processed_queue:
            if timestamp < cutoff_time:
                break
            if timestamp - datetime.timedelta(seconds=time_taken) < cutoff_time:
                # Have to apply discounting because not all files processed by this client were processed in last_n_seconds, so calculate roughly how many were using *math*
                seconds_in_stats_window = timestamp - cutoff_time
                # What percentage of time_taken is in period we care about
                overlap_ratio = seconds_in_stats_window.total_seconds() / time_taken
                effective_num_processed = num_processed * overlap_ratio
                num_processed_in_timeframe += effective_num_processed
            else:
                num_processed_in_timeframe += num_processed
        return dict(
            files_per_second=(num_processed_in_timeframe / last_n_seconds),
            total_files=self.total_processed,
        )



class DirectoryQueue:
    def __init__(self, input_dir, output_dir, filter_text, resume_mode):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filter_text = filter_text
        self.scan_iter = all_gzipped_files(os.walk(input_dir), filter_text, output_dir, input_dir, resume_mode)
        self.resume_mode = resume_mode
        self.total_processed = 0
        self.client_tracker = {}

    def track_stats(self, num_processed, time_taken, timestamp, client_name):
        tracker = self.client_tracker[client_name]
        tracker.total_processed += num_processed
        tracker.processed_queue.appendleft((timestamp, num_processed, time_taken))

    def register_client(self, client_name):
        if client_name not in self.client_tracker:
            self.client_tracker[client_name] = ClientStats()
        self.client_tracker[client_name].num_attached_clients += 1

    async def print_stats(self, stats_period):
        while True:
            computed_rate = 0
            for client_name, client_stats in self.client_tracker.items():
                stats = client_stats.compute_stats_for_client(last_n_seconds=stats_period)
                if not stats:
                    continue
                print(f'client {client_name}: procs {client_stats.num_attached_clients} files {stats["total_files"]}  rate {stats["files_per_second"]:.2f}')
                computed_rate += stats["files_per_second"]
            print(f'total {self.total_processed}  rate {computed_rate:.2f}')
            await asyncio.sleep(stats_period)

    async def handle_new_client(self, reader, writer):
        # Check it's a valid connection and client is ready
        start_message = await reader.readuntil(encoding.SEP)
        assert encoding.remove_sep(start_message) == b'ready', start_message

        client_identification_message = encoding.remove_sep(await reader.readuntil(encoding.SEP))
        client_name, client_set_chunk_size_str = client_identification_message.decode().split(' ')
        client_set_chunk_size = int(client_set_chunk_size_str)

        # Find some files to give the client
        print(f'new client: {client_name} chunksize {client_set_chunk_size}')
        self.register_client(client_name)
        effective_chunk_size = client_set_chunk_size

        while True:
            filenames = []
            start = time.time()
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
                try:
                    file_output = encoding.remove_sep(await reader.readuntil(encoding.SEP))
                    if not file_output:
                        break
                    file_outputs.append(file_output)
                except IncompleteReadError:
                    tracker = self.client_tracker.get(client_name)
                    # TODO: probably put requeueing broken jobs right here
                    if tracker is not None:
                        tracker.num_attached_clients -= 1
                    return
            # TODO: Do some sanity checking on these files to make sure they're roughly the right size.

            self.track_stats(
                num_processed=len(filenames),
                time_taken=time.time() - start,
                timestamp=datetime.datetime.now(),
                client_name=client_name,
            )
            self.total_processed += len(filenames)
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

    loop = asyncio.get_event_loop()
    loop.create_task(directory_queue.print_stats(args.stats_period))

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
        '--stats-period',
        dest='stats_period',
        type=int,
        default=30,
        help='Compute client stats and print them every N seconds'
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
