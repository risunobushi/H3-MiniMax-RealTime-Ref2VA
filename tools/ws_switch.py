"""Stream N windows and switch garment / framing mid-run; prints per-window gen + first clip so recompiles would show."""
import asyncio
import io
import json
import struct
import sys
import time

import websockets
from PIL import Image

URL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
W = 39


async def main():
    async with websockets.connect(URL, max_size=None) as ws:
        t_conn = time.time()
        hello = json.loads(await ws.recv())
        offset = hello["server_time"] - (t_conn + time.time()) / 2
        garments = hello["garments"]
        print("garments:", garments, flush=True)
        buf = io.BytesIO()
        Image.effect_noise((288, 512), 60).convert("RGB").save(buf, format="JPEG", quality=70)
        jpg = buf.getvalue()
        switches = {2: {"garment": garments[1]}, 4: {"framing": "selfie"}, 6: {"garment": garments[3], "framing": "half"},
                    8: {"garment": "default", "framing": "auto"}}

        async def sender():
            t0 = time.time()
            for i in range(W * N):
                if i % W == 0 and (i // W) in switches:
                    await ws.send(json.dumps({"type": "config", **switches[i // W]}))
                await ws.send(struct.pack("<d", time.time()) + jpg)
                await asyncio.sleep(max(0.0, t0 + (i + 1) / 24 - time.time()))

        task = asyncio.create_task(sender())
        done = 0
        while done < N:
            m = await asyncio.wait_for(ws.recv(), 60)
            if isinstance(m, str):
                m = json.loads(m)
                if m["type"] == "clip" and m.get("offset", 0) == 0:
                    t = time.time()
                    print(f"window {m['window']} {m.get('garment')}/{m.get('framing')}: last-capture->first-output "
                          f"{t - m['t_last_capture']:.2f}s  gen1 {m['t_send'] - m['t_gen_start']:.2f}  "
                          f"total {t - m['t_first_capture']:.2f}s", flush=True)
                elif m["type"] == "window_done":
                    done += 1
                    print(f"  done {m['window']}: gen {m['gen_s']}s queue {m['queue_windows']} start+{m.get('start_after_ready_s', 0):.2f}",
                          flush=True)
                elif m["type"] == "config":
                    print("  config ack:", m, flush=True)
        await task


asyncio.run(main())
