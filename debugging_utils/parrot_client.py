import asyncio
import json

from encoding import write_payload


async def tcp_echo_client():
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8888)

    write_payload(writer, [b'ready'])
    await writer.drain()

    write_payload(writer, [str(1).encode()])
    await writer.drain()

    data = await reader.read()

    write_payload(writer, [data])
    await writer.drain()
    print('calling back')
    writer.close()
    await writer.wait_closed()

asyncio.run(tcp_echo_client())
