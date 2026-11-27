"""Smoke/timing client for live_server.py (v1 and v2 protocols).

  python ws_smoke.py ws://127.0.0.1:8000/ws?token=TOKEN [windows=6] [W=39] [camera WxH]
With a camera size (e.g. 1280x720) the client reports it (&cw=&ch=) and sends frames at the size the server answers.
Streams a synthetic camera at 24 fps for `windows` windows while receiving, and prints per window the time from the
window's last captured frame to the first output frame received (client clock) plus the server-side breakdown.
"""
import asyncio
import io
import json
import struct
import sys
import time

import websockets
from PIL import Image

URL = sys.argv[1]
N_WINDOWS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
W = int(sys.argv[3]) if len(sys.argv) > 3 else 39
if len(sys.argv) > 4:
    URL += "&cw={}&ch={}".format(*sys.argv[4].lower().split("x"))


async def main():
    async with websockets.connect(URL, max_size=None) as ws:
        t_conn = time.time()
        hello = json.loads(await ws.recv())
        t_hello = time.time()
        offset = hello["server_time"] - (t_conn + t_hello) / 2  # server - client
        print("hello:", hello, f"clock offset {offset:+.3f}s", flush=True)
        buf = io.BytesIO()
        Image.effect_noise((hello.get("w", 288), hello.get("h", 512)), 60).convert("RGB").save(buf, format="JPEG", quality=70)
        jpg = buf.getvalue()
        total = W * N_WINDOWS

        async def sender():
            t0 = time.time()
            for i in range(total):
                await ws.send(struct.pack("<d", time.time()) + jpg)
                await asyncio.sleep(max(0.0, t0 + (i + 1) / 24 - time.time()))
            print(f"sent {total} frames in {time.time() - t0:.2f}s", flush=True)

        send_task = asyncio.create_task(sender())
        seen_first = {}
        done = 0
        meta = None
        bytes_in = 0
        deadline = time.time() + total / 24 + 30
        while done < N_WINDOWS and time.time() < deadline:
            try:
                m = await asyncio.wait_for(ws.recv(), 30)
            except asyncio.TimeoutError:
                print("timeout waiting for output", flush=True)
                break
            t = time.time()
            if isinstance(m, str):
                m = json.loads(m)
                if m["type"] in ("clip", "window"):
                    meta = m
                    if m.get("offset", 0) == 0 and m["window"] not in seen_first:
                        seen_first[m["window"]] = True
                        srv = lambda x: x - offset  # noqa: E731
                        print(f"window {m['window']}: last-capture->first-output {t - m['t_last_capture']:.2f}s  "
                              f"[uplink {srv(m['t_recv_last']) - m['t_last_capture']:.2f} wait {m['t_gen_start'] - m['t_recv_last']:.2f} "
                              f"gen1 {m['t_send'] - m['t_gen_start']:.2f} down {t - srv(m['t_send']):.2f}]  "
                              f"first-capture->first-output {t - m['t_first_capture']:.2f}s", flush=True)
                elif m["type"] == "window_done":
                    done += 1
                    print(f"window {m['window']} done: first clip {m['first_clip_s']}s gen {m['gen_s']}s queue {m['queue_windows']} "
                          f"ready+{m.get('ready_after_recv_s', 0):.2f} start+{m.get('start_after_ready_s', 0):.2f}", flush=True)
                elif m["type"] == "pong":
                    pass
            else:
                bytes_in += len(m)
        await send_task
        print(f"received {bytes_in / 1024:.0f} KB of output", flush=True)


asyncio.run(main())
