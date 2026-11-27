"""Poll Runpod every 20 s and create the first available Blackwell pod (B300 secure/community, then B200 secure/community).
Prints one line per state change; prints 'CREATED <json>' and exits on success. Key: env RUNPOD_API_KEY,
else ~/.runpod/config.toml (written by `runpodctl config --apiKey <key>`)."""
import json
import os
import re
import sys
import time
import urllib.request


def runpod_key():
    """RUNPOD_API_KEY, else the key runpodctl stores in ~/.runpod/config.toml."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    path = os.path.expanduser("~/.runpod/config.toml")
    if os.path.exists(path):
        m = re.search(r'(?i)api_?key\s*=\s*"?([A-Za-z0-9_\-]+)', open(path).read())
        if m:
            return m.group(1)
    sys.exit("No Runpod API key found. Set RUNPOD_API_KEY, or run: runpodctl config --apiKey <key>  "
             "(keys: https://console.runpod.io/user/settings)")


KEY = runpod_key()
# SSH public keys injected into the pod: env RUNPOD_PUBLIC_KEYS (newline-separated), else every ~/.ssh/*.pub
_SSH = os.path.expanduser("~/.ssh")
PUBKEYS = os.environ.get("RUNPOD_PUBLIC_KEYS") or "\n".join(
    open(os.path.join(_SSH, f)).read().strip() for f in (sorted(os.listdir(_SSH)) if os.path.isdir(_SSH) else [])
    if f.endswith(".pub"))
if not PUBKEYS.strip():
    sys.exit("No SSH public key found. Create one (ssh-keygen -t ed25519) or set RUNPOD_PUBLIC_KEYS.")
NAME = sys.argv[1] if len(sys.argv) > 1 else "h3-mirror"  # python snipe_pod.py [pod-name]
ATTEMPTS = [("NVIDIA B300 SXM6 AC", "SECURE"), ("NVIDIA B300 SXM6 AC", "COMMUNITY"),
            ("NVIDIA B200", "SECURE"), ("NVIDIA B200", "COMMUNITY")]


def api(method, path, body=None):
    req = urllib.request.Request("https://api.runpod.io/v2" + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return -1, {"detail": str(e)[:80]}


def body(gpu, cloud):
    return {"name": NAME, "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "gpu": {"id": gpu, "count": 1}, "cloud": cloud, "ports": ["22/tcp", "8000/http"], "disk": 250,
            "startSsh": True, "env": {"HF_HOME": "/workspace/hf", "PIP_BREAK_SYSTEM_PACKAGES": "1", "PUBLIC_KEY": PUBKEYS}}


n = 0
last = ""
while True:
    n += 1
    for gpu, cloud in ATTEMPTS:
        st, d = api("POST", "/pods", body(gpu, cloud))
        if st in (200, 201) and d.get("id"):
            print("CREATED " + json.dumps({"id": d["id"], "gpu": gpu, "cloud": cloud, "dc": d.get("dataCenterId"),
                                          "cost": d.get("cost")}), flush=True)
            sys.exit(0)
        detail = str(d.get("detail", d))[:100]
        if "no longer any instances" not in detail and st != 400:
            msg = f"{time.strftime('%H:%M:%S')} {gpu.split()[1]}/{cloud}: HTTP {st} {detail}"
            if msg[9:] != last[9:]:
                print(msg, flush=True)
                last = msg
    if n % 45 == 0:
        print(f"{time.strftime('%H:%M:%S')} heartbeat: still no B300/B200 ({n} rounds)", flush=True)
    time.sleep(20)
