"""Live mirror server v3: browser camera -> WebSocket JPEG frames -> H3 Ref2VA garment swap -> frames back, progressively,
on one or more GPUs (round-robin windows).

  cd /workspace/Minimax-H3-Turbo && python live_server.py --port 8000 --w 288 --h 512 --ref 224 --window 39 --steps 3 --compile --gpus 0,1
Open https://<podid>-8000.proxy.runpod.net/?token=<TOKEN> on the phone/laptop.

Lag levers implemented:
  1. progressive output: every VAE clip (17 frames) is JPEG-encoded on the GPU and sent as soon as the decoder has
     finalised it (P_first instead of P_total), packed as ONE binary message per clip;
  2. smaller payload (JPEG q70 by default) and per-window timing fields so the client can attribute the lag;
  3. intake VAE encode runs on its own CUDA stream (does not wait behind the running denoise);
  4. N GPUs: window k is generated on GPU k mod N (throughput x N -> shorter windows keep up); output is sequenced.
Protocol (server -> client): text {"type":"hello"...}; per clip text {"type":"clip",...} then one binary
[uint32 n][uint32 size_i * n][jpeg bytes ...]; per window text {"type":"window_done",...}.
Fallback (progressive disabled): text {"type":"window",...} then N binary JPEG frames (v1 protocol).

v4 (2026-09-23), single-GPU fixed-cost work, each lever switchable for A/B:
  5. per-stage timing on every window (CPU wall vs GPU span per pipeline block, CUDA events, no extra syncs) +
     --profile-window K (torch.profiler chrome trace of one window) + host CPU info in the log;
  6. CPU thread caps (H3_THREADS, default 8: OMP/MKL pools otherwise size themselves to every core the host shows,
     e.g. 344 on US-WA-2) and --compile-mode (reduce-overhead = CUDA graphs); every generate() of a GPU runs on one
     dedicated thread, warm-up included, so CUDA graphs recorded in warm-up are the ones replayed;
  7. TF32 for fp32 matmuls (--no-tf32 to disable), --enc-autocast fp16|bf16 for the intake VAE encode (the decode
     already runs under fp16 autocast in diffusers), output_type="pt" (no 39-frame float32 numpy copy per window);
  8. early conditioner (--no-early-cond to disable): at 1 fps the conditioner reads only frames 0 and 24 of a
     39-frame window, so it runs on a side stream as soon as frame 24 arrives, 0.6 s before the window completes,
     and the text-encoder step consumes the result. Verified at warm-up against the in-pipeline computation.

v5 (2026-10-09), temporal continuity: latent prefix (--prefix-clips K, single GPU, off by default).
  The last K generated 17-frame clips of window j are kept as latents (normalized, taken before the decode, so never
  re-encoded) and become clean conditioning rows at the START of window j+1's target timeline: the fl2va keyframe
  mechanism (rows held at keyframe_noise_aug, never written by the scheduler) extended from one frame to whole clips,
  inside the ref2va layout. Windows then overlap: window j reads frames [j*S, j*S + T) with S = W - 5 (34 for W39)
  new frames, T = W + 17K frames of timeline (K context clips + S new + a 5-frame tail that is generated, dropped and
  regenerated as the start of the next window, because the 17n+5 layout only allows a short clip at the end). The
  webcam reference covers the whole timeline, so reference and output stay frame-aligned. Window 0 of a session, the
  first window after a garment switch and the first after a catch-up skip have no usable prefix: they generate the
  context clips too (same sequence length, so no recompile) and emit only the S new frames. Costs ~T/W x the
  denoise of a plain W window per S/24 s; the first 17K frames of a session are never shown.
"""
import os

# must precede numpy/torch imports: thread pools size themselves from the visible core count, not the cgroup quota
_THREADS = os.environ.get("H3_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import collections  # noqa: E402
import faulthandler  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import secrets  # noqa: E402
import signal  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

sys.path.insert(0, "/workspace/Minimax-H3-Turbo")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import before_encoder as be  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import encoders as enc  # noqa: E402

from shades import load_shades, write_swatches  # noqa: E402
from stream_sim import CLIP, FPS, LatentRing, build_pipeline  # noqa: E402

try:
    from torchvision.io import encode_jpeg as tv_encode_jpeg
except Exception:  # noqa: BLE001
    tv_encode_jpeg = None

app = FastAPI()
STATE = {}
INTAKE = ThreadPoolExecutor(max_workers=1)
JPEG_POOL = ThreadPoolExecutor(max_workers=8)
HTML = Path(__file__).with_name("live.html").read_text()
VAE_OWNER = {}  # id(vae) -> GpuWorker (for the module-level encode_vae_condition patch)


def host_info():
    """Visible cores vs affinity vs cgroup quota vs load: the fixed per-window cost has been host-dependent."""
    info = {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "omp": os.environ.get("OMP_NUM_THREADS")}
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except Exception:  # noqa: BLE001
        pass
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            info["cgroup_cpu"] = Path(path).read_text().strip()
            break
        except Exception:  # noqa: BLE001
            pass
    try:
        info["load"] = [round(x, 1) for x in os.getloadavg()]
    except Exception:  # noqa: BLE001
        pass
    return info


def leaf_blocks(block):
    subs = getattr(block, "sub_blocks", None)
    if subs:
        for sub in (subs.values() if hasattr(subs, "values") else list(subs)):
            yield from leaf_blocks(sub)
    else:
        yield block


class StageTimer:
    """Per-window timing of every leaf pipeline block, recorded only on the thread that set `rec` (the generation
    thread). Per call: CPU wall (perf_counter around the call) and GPU span (CUDA events on the current stream,
    read after the window's final synchronize, so no extra syncs are added). CPU wall ~= GPU span on a block means
    it is launch/CPU-bound; GPU span >> CPU wall means the GPU is the limit."""

    local = threading.local()
    patched = set()

    @classmethod
    def install(cls, pipe):
        for leaf in leaf_blocks(pipe.blocks):
            klass = type(leaf)
            if klass in cls.patched:
                continue
            cls.patched.add(klass)
            orig = klass.__call__

            def make(orig, name):
                def timed_call(self, *a, **k):
                    rec = getattr(StageTimer.local, "rec", None)
                    if rec is None:
                        return orig(self, *a, **k)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    e0.record()
                    out = orig(self, *a, **k)
                    e1.record()
                    rec.append((name, time.perf_counter() - t0, e0, e1))
                    return out

                return timed_call

            klass.__call__ = make(orig, klass.__name__.replace("MiniMaxH3", "").replace("Step", ""))

    @classmethod
    def start(cls):
        cls.local.rec = []

    @classmethod
    def collect(cls):
        """Call after torch.cuda.synchronize(). Returns {name: [cpu_s, gpu_s, calls]} in first-call order."""
        rec, cls.local.rec = getattr(cls.local, "rec", None) or [], None
        out = {}
        for name, cpu, e0, e1 in rec:
            s = out.setdefault(name, [0.0, 0.0, 0])
            s[0] += cpu
            s[1] += e0.elapsed_time(e1) / 1000.0
            s[2] += 1
        return {k: [round(v[0], 4), round(v[1], 4), v[2]] for k, v in out.items()}


def fmt_stages(stages, min_s=0.005):
    parts = []
    for name, (cpu, gpu, n) in stages.items():
        if max(cpu, gpu) < min_s:
            continue
        parts.append(f"{name}{'x' + str(n) if n > 1 else ''} {cpu:.3f}/{gpu:.3f}")
    return " | ".join(parts)


def pack_jpegs(jpegs):
    """[uint32 n][uint32 size_i ...][payload]"""
    head = struct.pack("<I", len(jpegs)) + struct.pack(f"<{len(jpegs)}I", *[len(j) for j in jpegs])
    return head + b"".join(jpegs)


FRAMINGS = {
    "auto": "The framing, crop and distance from the camera are exactly those of <Video 1>: if <Video 1> shows only "
            "the head and shoulders, the output shows only the head and shoulders; if it shows the full body, the output "
            "shows the full body. Never zoom out and never reveal more of the body or of the room than <Video 1> does.",
    "selfie": "<Video 1> is a hand-held selfie close-up: head, shoulders and upper chest fill the frame, the camera is at "
              "arm's length and the background is only what is visible behind the shoulders. Keep exactly this close "
              "framing; the garment is seen from the collar to the chest only.",
    "half": "<Video 1> is a half-body shot from the head to about the waist, camera at a short distance. Keep exactly "
            "this framing; do not show the legs or feet.",
    "full": "<Video 1> is a full-body shot with the whole person in frame. Keep exactly this framing and distance.",
}


# lipstick task: the lips must be large in frame, so only the camera's own framing and a face close-up are offered
LIP_FRAMINGS = {
    "auto": FRAMINGS["auto"],
    "selfie": "<Video 1> is a hand-held selfie close-up: the face fills most of the frame and the camera is at arm's "
              "length. Keep exactly this close framing.",
}


def canvas_sizes(w, h, aspects):
    """Output sizes a client can get: the default (w, h) plus, for every extra aspect "a:b", the size on the model's
    32-px grid whose aspect is closest to a:b within the same pixel budget (288x512 -> 16:9 512x288, 4:3 448x320,
    3:4 320x448, 1:1 384x384). Each size is its own torch.compile shape per framing, compiled in the warm-up."""
    sizes, budget = [(w, h)], w * h
    for a in [x for x in (aspects or "").split(",") if x.strip()]:
        num, den = (float(v) for v in a.split(":"))
        cands = [(cw, ch) for cw in range(128, 1025, 32) for ch in range(128, 1025, 32)
                 if 0.8 * budget <= cw * ch <= budget]
        best = min(cands, key=lambda c: (round(abs(math.log(c[0] / c[1] * den / num)), 3), -c[0] * c[1]))
        if best not in sizes:
            sizes.append(best)
    return sizes


def closest_size(sizes, cam_w, cam_h):
    """The offered size whose aspect ratio is closest to the camera's (ties: the first, i.e. the default)."""
    return min(sizes, key=lambda c: abs(math.log(c[0] / c[1] * cam_h / cam_w)))


def framings_for(task):
    return LIP_FRAMINGS if task == "lipstick" else FRAMINGS


def build_prompt(framing="auto", prev=False, task="garment", swatch="chip"):
    if task == "lipstick":
        return build_lip_prompt(framing, swatch)
    fr = FRAMINGS.get(framing, FRAMINGS["auto"])
    p2_def = ("<Picture 2> is the last frame of the previous second of this same output video: <Subject 1> already "
              "wearing <Subject 2>. The new video continues seamlessly from <Picture 2>, with identical face, hair, "
              "skin tone, body, garment rendering (color, fit, folds, print placement), lighting, framing and "
              "background.\n") if prev else ""
    p2_sum = " Continue seamlessly from <Picture 2>; the garment must look exactly as it does there." if prev else ""
    p2_ret = ("<Picture 2> (start of the video): fully_preserved - the first frame matches <Picture 2> and the look "
              "of the garment stays identical throughout.\n") if prev else ""
    return (
        "subject_definitions:\n"
        "<Subject 1> is the person in <Video 1>, preserving their face, hair, skin, body proportions, every piece of "
        "clothing that is not replaced by <Subject 2>, and every movement exactly as in <Video 1>.\n"
        "<Subject 2> is the garment in <Picture 1> (it may be a top, a bottom, a dress or outerwear), preserving its "
        "exact color, fabric, cut, texture and print.\n"
        "<Subject 3> is the scene of <Video 1>, preserving its exact background, camera position, distance, crop, "
        "framing and lighting.\n" + p2_def + "\n"
        "summary:\n"
        "[reference generation] This is <Video 1> itself, unchanged, except that <Subject 1> now wears <Subject 2> in "
        "place of the garment that covered the same part of the body in <Video 1>. Everything else stays exactly as "
        "in <Video 1>: same face, hair, skin, hands, every other garment, accessories, background, camera, framing, "
        "crop, lighting, motion and timing. Do not add, remove, move, recolor or restyle anything else. "
        + fr + p2_sum + "\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears throughout): fully_preserved - identical face, hair, body, pose, motion timing and "
        "position in frame as <Video 1>.\n"
        "<Subject 2> (appears throughout): fully_preserved - worn by <Subject 1> in place of the original garment on "
        "that part of the body, keeps its color, cut, texture and print, fitted naturally to the body and following "
        "the motion; only the part of it that the framing of <Video 1> would show is visible.\n"
        "<Subject 3> (appears throughout): fully_preserved - identical background, crop, framing, camera distance and "
        "lighting as <Video 1>; no zoom-out, no new room, no mirror, no new objects, no camera movement beyond what "
        "<Video 1> has.\n" + p2_ret + "\n"
        "detailed_description:\n"
        "The same video as <Video 1> with one garment swapped for <Subject 2>. Nothing else changes.\n\n"
        "overall_soundscape:\nquiet room tone.\n"
    )


def build_lip_prompt(framing="auto", swatch="chip"):
    """Same structure as the garment prompt, but <Picture 1> is a lipstick swatch (a color chip, or lips filled with
    the shade, on white). The shade is never named in text, so all shades share one token count (no recompile)."""
    fr = LIP_FRAMINGS.get(framing, LIP_FRAMINGS["auto"])
    what = ("a color chip; its shape never appears in the video" if swatch == "chip"
            else "a color swatch; the drawn lips themselves never appear in the video")
    return (
        "subject_definitions:\n"
        "<Subject 1> is the person in <Video 1>, preserving their face, facial features, lip shape and size, teeth, "
        "hair, skin, every piece of clothing, every expression and every movement exactly as in <Video 1>.\n"
        "<Subject 2> is the lipstick shade shown in <Picture 1>, preserving its exact color (hue, depth and "
        "saturation) and its finish: matte if the swatch has no highlight, satin if it has a soft sheen, glossy if it "
        "has a sharp shine. <Picture 1> is only " + what + ".\n"
        "<Subject 3> is the scene of <Video 1>, preserving its exact background, camera position, distance, crop, "
        "framing and lighting.\n\n"
        "summary:\n"
        "[reference generation] This is <Video 1> itself, unchanged, except that <Subject 1> now wears lipstick in the "
        "shade <Subject 2>, applied evenly and precisely to the lips only, exactly following the natural lip line and "
        "replacing any lip color in <Video 1>, like a real, natural lipstick application. The lips keep exactly "
        "their natural shape, size, outline and fullness, and the mouth opens and closes exactly as in <Video 1>: "
        "if it is closed there, it stays closed. When the mouth opens, the teeth, tongue and inside of the mouth are not colored. Everything else stays exactly as in "
        "<Video 1>: same face, skin around the mouth, eyes, any other makeup, hair, clothing, hands, background, "
        "camera, framing, crop, lighting, motion and timing. Do not add, remove, move, recolor or restyle anything "
        "else. " + fr + "\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears throughout): fully_preserved - identical face, lips, expressions, hair, body, pose, "
        "motion timing and position in frame as <Video 1>.\n"
        "<Subject 2> (appears throughout): fully_preserved - on the lips only, same color and finish as <Picture 1> "
        "under the lighting of <Video 1>, following every lip movement, no color on the skin, teeth or chin.\n"
        "<Subject 3> (appears throughout): fully_preserved - identical background, crop, framing, camera distance and "
        "lighting as <Video 1>; no zoom, no new objects, no camera movement beyond what <Video 1> has.\n\n"
        "detailed_description:\n"
        "The same video as <Video 1> with the lips of the person wearing lipstick in shade <Subject 2>. Nothing else "
        "changes.\n\n"
        "overall_soundscape:\nquiet room tone.\n"
    )


def install_latent_injection():
    """Module-level patch: garment latents cached per VAE, window latents come from the owning worker's ring."""
    orig = enc.encode_vae_condition
    cache = {}

    def encode_cond(vae_, pixels, pixel_mean, pixel_std, encode_seed=42):
        if pixels.shape[2] == 1:
            key = (id(vae_), pixels.shape, int(pixels.float().sum().item()))
            if key not in cache:
                if len(cache) > 32:
                    cache.pop(next(iter(cache)))
                cache[key] = orig(vae_, pixels, pixel_mean, pixel_std, encode_seed)
            return cache[key]
        w = VAE_OWNER[id(vae_)]
        post = DiagonalGaussianDistribution(w.current["moments"])
        lat = post.sample(generator=torch.Generator().manual_seed(encode_seed)).to(torch.float16).float().cpu()
        return (lat - w.lm) / w.ls

    enc.encode_vae_condition = encode_cond


_PFX = threading.local()  # per generation thread: .latents (prefix to clamp), .keep (latent slice to save), .saved


def install_latent_prefix():
    """Class-level patches for --prefix-clips (diffusers 0.40.0). The layout is built for the full timeline (prefix +
    generated); its target rows are frame-major, so the first rows of the target are exactly the prefix latent frames,
    and declaring them conditioning rows is all it takes: PrepareConditionLatents packs (and noises to
    keyframe_noise_aug) the prefix after the references, so condition row i still maps to video_indices[i]; the
    timestep plan pins them; the scheduler and the unpack/decode only see the rows after them. AfterDenoise saves the
    requested slice of the generated latents (normalized) for the next window."""
    from diffusers.modular_pipelines.minimax_h3 import before_denoise as bd
    from diffusers.modular_pipelines.minimax_h3 import decoders as dec

    layout_cls, cond_cls, after_cls = (bd.MiniMaxH3Ref2VAPrepareLayoutStep, bd.MiniMaxH3PrepareConditionLatentsStep,
                                       dec.MiniMaxH3AfterDenoiseStep)
    if getattr(layout_cls, "_prefix_patched", False):
        return
    orig_layout, orig_cond, orig_after = layout_cls.__call__, cond_cls.__call__, after_cls.__call__

    def layout_call(self, components, state):
        components, state = orig_layout(self, components, state)
        pre = getattr(_PFX, "latents", None)
        if pre is not None:
            pt, ph, pw = components.patch_size
            lh, lw = state.get("latent_height"), state.get("latent_width")
            if tuple(pre.shape[3:]) != (lh, lw) or pre.shape[2] % pt:
                raise ValueError(f"latent prefix {tuple(pre.shape)} does not fit the target latents {lh}x{lw}")
            rows = (pre.shape[2] // pt) * (lh // ph) * (lw // pw)
            state.set("num_condition_video_rows", state.get("num_condition_video_rows") + rows)
            state.set("num_latent_frames", state.get("num_latent_frames") - pre.shape[2])
        return components, state

    def cond_call(self, components, state):
        pre = getattr(_PFX, "latents", None)
        if pre is None:
            return orig_cond(self, components, state)
        lat = state.get("condition_latents")
        state.set("condition_latents", list(lat) + [pre])  # packed last = adjacent to the generated rows
        try:
            return orig_cond(self, components, state)
        finally:
            state.set("condition_latents", lat)

    def after_call(self, components, state):
        components, state = orig_after(self, components, state)
        keep = getattr(_PFX, "keep", None)
        if keep is not None:
            _PFX.saved = state.get("latents")[:, :, keep[0]:keep[1]].clone()  # (1, C, 5K, h, w), normalized
        return components, state

    layout_cls.__call__, cond_cls.__call__, after_cls.__call__ = layout_call, cond_call, after_call
    layout_cls._prefix_patched = True


def install_early_cond(te_cls):
    """Class-level patch of the ref2va text-encoder step: (a) stash the normalized garment reference of every
    [video, garment] request (what the setup step made of it) so the early conditioner can reuse it; (b) if an early
    result for this exact window/garment/framing exists (or is in flight), use it instead of running the conditioner
    here; (c) in check mode compute both and compare. Serializes all conditioner calls of a GPU (Qwen3-VL keeps
    per-call state such as rope_deltas on the module)."""
    if getattr(te_cls, "_early_cond_patched", False):
        return
    orig = te_cls.__call__

    def call(self, components, state):
        w = VAE_OWNER.get(id(getattr(components, "vae", None)))
        cur = w.cur if w is not None else None
        if cur is None:
            return orig(self, components, state)
        bs = self.get_block_state(state)
        refs = bs.normalized_references
        if len(refs) == 2 and refs[0].kind == "video" and refs[1].kind == "image":
            w.garment_norm[cur["garment"]] = refs[1]
        pre = None
        fut = w.pre.pop(cur["cond_key"], None)
        if fut is not None:
            try:
                pre = fut.result(timeout=5.0)
            except Exception as e:  # noqa: BLE001
                print(f"[gpu{w.device}] early conditioner failed: {e!r}", flush=True)
        if pre is not None and not w.pre_check:
            bs.prompt_embeds, bs.text_token_tags = pre
            self.set_block_state(state, bs)
            cur["cond"] = "early"
            return components, state
        with w.te_lock:
            components, state = orig(self, components, state)
        cur["cond"] = "inline"
        if pre is not None and w.pre_check:
            got = self.get_block_state(state)
            same_tags = torch.equal(got.text_token_tags.cpu(), pre[1].cpu())
            same_shape = got.prompt_embeds.shape == pre[0].shape
            diff = (got.prompt_embeds.float() - pre[0].float()).abs().max().item() if same_shape else float("inf")
            scale = got.prompt_embeds.float().abs().max().item() if same_shape else 1.0
            ok = same_tags and same_shape and diff <= 1e-2 * max(scale, 1.0)
            print(f"[gpu{w.device}] early-conditioner self-check: shape {tuple(pre[0].shape)} vs "
                  f"{tuple(got.prompt_embeds.shape)}, tags equal {same_tags}, max|diff| {diff:.4g} (max|x| {scale:.3g}) "
                  f"-> {'OK' if ok else 'MISMATCH, early conditioner DISABLED'}", flush=True)
            w.pre_checked = True
            if not ok:
                w.early_cond = False
        return components, state

    te_cls.__call__ = call
    te_cls._early_cond_patched = True


class GpuWorker:
    """One pipeline instance on one GPU: ring of intake latents, generation, progressive JPEG emission."""

    def __init__(self, mirror, device, args):
        self.mirror, self.device, self.args = mirror, device, args
        torch.cuda.set_device(device)
        self.pipe = build_pipeline(args)
        vae = self.pipe.vae
        VAE_OWNER[id(vae)] = self
        self.ring = LatentRing(vae, self.pipe.pixel_mean, self.pipe.pixel_std)
        self.ring.tails = {}  # prefix mode: tail start -> moments of a window's own padded 5-frame tail
        self.ref_len = mirror.T  # reference frames = timeline frames per window (= W without a prefix)
        self.lpc = self.pipe.vae_latents_per_chunk  # latents per 17-frame clip
        self.lm = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        self.ls = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        self.pm = torch.tensor(self.pipe.pixel_mean, device="cuda").view(1, -1, 1, 1, 1)
        self.ps = torch.tensor(self.pipe.pixel_std, device="cuda").view(1, -1, 1, 1, 1)
        self.current = {"moments": None}
        self.enc_stream = torch.cuda.Stream()
        self.progressive = not args.no_progressive
        self.gpu_jpeg = tv_encode_jpeg is not None and not args.cpu_jpeg
        self.cur = None  # per-window emission state
        self.check = None  # warm-up self-check buffer
        self.windows = 0
        self._install_progressive_decode()
        # every generate() of this GPU runs on this one thread (warm-up too): CUDA graphs are recorded per thread
        self.gen_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gen{device}")
        # intake VAE encode precision (the recipe is fp32; the decode already runs under fp16 autocast)
        self.enc_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.enc_autocast)
        self.enc_checked = self.enc_dtype is None
        # early conditioner
        self.te_block = next((b for b in leaf_blocks(self.pipe.blocks) if hasattr(b, "video_sample_fps")), None)
        self.te_lock = threading.Lock()
        self.pre = {}  # cond_key -> Future[(prompt_embeds, text_token_tags)]
        self.pre_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cond{device}")
        self.pre_stream = torch.cuda.Stream()
        self.garment_norm = {}  # garment name -> normalized MiniMaxH3ImageReference (from the setup step)
        self.pre_check = False
        self.pre_checked = False
        self.early_cond = bool(args.early_cond) and self.te_block is not None
        self.sample_idx = None  # window-relative frame indices the conditioner reads
        if self.early_cond:
            try:
                tp = self.pipe.processor.video_processor.temporal_patch_size
                idx, _ = type(self.te_block)._sample_video_condition_frames(
                    np.arange(self.ref_len), float(self.pipe.fps), self.te_block.video_sample_fps, tp)
                self.sample_idx = [int(i) for i in idx]
            except Exception as e:  # noqa: BLE001
                print(f"[gpu{device}] early conditioner unavailable: {e!r}", flush=True)
                self.early_cond = False
        print(f"[gpu{device}] early conditioner: {self.early_cond} (reads window frames {self.sample_idx} at "
              f"{getattr(self.te_block, 'video_sample_fps', '?')} fps)", flush=True)
        if self.te_block is not None:
            install_early_cond(type(self.te_block))  # before StageTimer, so the timer wraps it
        StageTimer.install(self.pipe)

    # --- progressive decode: re-implementation of AutoencoderKLMiniMaxH3._decode (diffusers 0.40.0) that hands every
    #     finalised (blended) chunk to _emit_chunk as soon as it exists, instead of only after torch.cat at the end.
    def _install_progressive_decode(self):
        vae = self.pipe.vae
        worker = self
        orig_decode = vae._decode

        def _decode(z):
            if worker.cur is None or not worker.progressive:
                return orig_decode(z)
            tokens_chunk_size = vae.tokens_chunk_size
            token_drop = vae.config.token_drop
            temporal_ratio = vae.temporal_compression_ratio
            chunk_num_frames = tokens_chunk_size * temporal_ratio
            num_tokens = z.shape[2] + token_drop
            pad_tokens = (-num_tokens) % tokens_chunk_size
            num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
            if pad_tokens > 0:
                z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
            decoded_chunks = []
            overlap = None
            for i in range(num_chunks):
                start = i * tokens_chunk_size
                clip = vae._decode_clip(z[:, :, start: start + tokens_chunk_size + vae.token_overlap])
                for j in range(int(token_drop > 0) + 1):
                    frame_start = j * chunk_num_frames
                    chunk = clip[:, :, frame_start: frame_start + chunk_num_frames]
                    chunk = chunk[:, :, vae.frame_pre_padding:]
                    if j == 0:
                        if overlap is not None:
                            chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                        decoded_chunks.append(chunk)
                        worker._emit_chunk(chunk)
                    else:
                        overlap = chunk
            if overlap is not None:
                decoded_chunks.append(overlap)
                worker._emit_chunk(overlap)
            dec = torch.cat(decoded_chunks, dim=2)
            if pad_tokens > 0:
                intra_tail = vae.config.clip_length % temporal_ratio
                num_tokens_before_pad = z.shape[2] - pad_tokens
                pad_frames = sum(
                    intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                    for k in range(pad_tokens))
                dec = dec[:, :, :-pad_frames]
            return dec

        vae._decode = _decode

    def _jpegs(self, u8_tchw):
        """uint8 (t, 3, h, w) CUDA tensor -> list of JPEG bytes (GPU nvjpeg via torchvision; PIL fallback)."""
        q = self.args.jpeg_q
        if self.gpu_jpeg:
            try:
                enc_list = tv_encode_jpeg(list(u8_tchw), quality=q)
                return [e.cpu().numpy().tobytes() for e in enc_list]
            except Exception as e:  # noqa: BLE001
                print(f"[gpu{self.device}] GPU JPEG failed ({e!r}); falling back to PIL", flush=True)
                self.gpu_jpeg = False
        arr = u8_tchw.permute(0, 2, 3, 1).cpu().numpy()

        def one(a):
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="JPEG", quality=q)
            return buf.getvalue()

        return list(JPEG_POOL.map(one, list(arr)))

    def _emit_chunk(self, chunk):
        """Called inside the VAE decoder for every finalised chunk (1, 3, t, h, w) in the VAE's pixel space."""
        cur = self.cur
        pos = cur["decoded"]  # decoded frames so far; only [skip, skip + W) of them are emitted
        cur["decoded"] = pos + chunk.shape[2]
        lo, hi = max(cur["skip"] - pos, 0), min(cur["skip"] + cur["W"] - pos, chunk.shape[2])
        if hi <= lo:
            return
        x = (chunk.float() * self.ps + self.pm).clamp(0, 1)[0, :, lo:hi]  # (3, t, h, w)
        u8 = (x * 255.0 + 0.5).to(torch.uint8).permute(1, 0, 2, 3).contiguous()  # (t, 3, h, w)
        if self.check is not None:
            self.check.append(u8.permute(0, 2, 3, 1).cpu().numpy())
        jpegs = self._jpegs(u8)
        torch.cuda.synchronize()
        now = time.time()
        if cur["t_first_clip"] is None:
            cur["t_first_clip"] = now
        offset = cur["emitted"]
        cur["emitted"] += len(jpegs)
        msg = {"type": "clip", "window": cur["index"], "gpu": self.device, "offset": offset, "frames": len(jpegs),
               "garment": cur.get("garment"), "framing": cur.get("framing"),
               "t_first_capture": cur["t_first_capture"], "t_last_capture": cur["t_last_capture"],
               "t_last_emit_capture": cur["t_last_emit_capture"], "t_recv_last": cur["t_recv_last"], "t_gen_start": cur["t_gen_start"], "t_send": now,
               "first_clip_s": round(cur["t_first_clip"] - cur["t_gen_start"], 3)}
        self.mirror.emit(cur["index"], msg, [pack_jpegs(jpegs)])

    def encode_clip_side(self, cs, clip_u8, tail=False):
        """VAE-encode one clip on the side stream (overlaps with a running denoise), store in this GPU's ring
        (tail=True: a prefix-mode window's own 5-frame tail, kept apart from the full clip that starts there)."""
        torch.cuda.set_device(self.device)
        ring = self.ring
        if clip_u8.shape[0] < CLIP:
            clip_u8 = np.concatenate([clip_u8, np.repeat(clip_u8[-1:], CLIP - clip_u8.shape[0], axis=0)])
        t = time.perf_counter()
        side = os.environ.get("H3_ENC_SIDE", "1") == "1"
        with torch.inference_mode(), torch.cuda.stream(self.enc_stream if side else torch.cuda.current_stream()):
            x = ring._prep(clip_u8)
            with torch.autocast("cuda", dtype=self.enc_dtype or torch.float32, enabled=self.enc_dtype is not None):
                m = ring.vae._encode_clip(x).float()
            if not self.enc_checked:  # first clip: compare with the fp32 recipe once
                ref = ring.vae._encode_clip(x).float()
                c = ref.shape[1] // 2  # moments = [mean | logvar]; the mean is what conditions (after fp16 rounding)
                err = ((m[:, :c] - ref[:, :c]).norm() / ref[:, :c].norm()).item()
                amax = (m[:, :c] - ref[:, :c]).abs().max().item()
                ok = err < 1e-2
                print(f"[gpu{self.device}] intake encode autocast {self.args.enc_autocast}: rel err of latent mean "
                      f"{err:.2e}, max|diff| {amax:.3g} -> {'OK' if ok else 'too large, back to fp32'}", flush=True)
                if not ok:
                    self.enc_dtype = None
                    m = ref
                self.enc_checked = True
        if side:
            self.enc_stream.synchronize()
            m = m.clone()  # owned by the default stream from here on
        else:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        with ring.lock:
            (ring.tails if tail else ring.clips)[cs] = m
            ring.encode_time += dt
        return dt

    def has_clips(self, start, W):
        with self.ring.lock:
            return all(s in self.ring.clips for s in range(start, start + W, CLIP))

    def _prefix_tail(self, r):
        return r + self.ref_len - self.ref_len % CLIP

    def has_prefix_clips(self, r):
        tail = self._prefix_tail(r)
        with self.ring.lock:
            return (all(s in self.ring.clips for s in range(r, tail, CLIP))
                    and (tail in self.ring.tails or tail in self.ring.clips))

    def prefix_moments(self, r):
        """Reference latents of a prefix-mode window: the full clips on the global grid from r, then its own tail
        (the causal encoder makes a padded tail's latents those of the full clip's first frames, so either serves)."""
        tail = self._prefix_tail(r)
        with self.ring.lock:
            parts = [self.ring.clips[s] for s in range(r, tail, CLIP)]
            parts.append(self.ring.tails[tail] if tail in self.ring.tails else self.ring.clips[tail])
        m = torch.cat(parts, dim=2)
        drop = self.ring.vae.config.token_drop
        return m[:, :, : m.shape[2] - drop] if drop else m

    @staticmethod
    def cond_key(session, index, garment, framing, prev):
        return (session, index, garment, framing, bool(prev))

    def schedule_early_cond(self, session, index, sampled, garment, framing):
        """Called by the intake once the conditioner's frames of window `index` have arrived. `sampled` = list of
        (window-relative index, frame). Skipped when the previous-frame reference is in use (its normalized image only
        exists after the previous window) or before the setup step has produced this garment's normalized reference."""
        if not self.early_cond or self.args.prev_ref or garment not in self.garment_norm:
            return
        key = self.cond_key(session, index, garment, framing, False)
        if key in self.pre:
            return
        for k in list(self.pre):  # list(): the generation thread pops concurrently
            if k[0] != session or k[1] < index - 2:
                self.pre.pop(k, None)
        self.pre[key] = self.pre_exec.submit(self._early_cond, sampled, self.garment_norm[garment], framing)

    def _early_cond(self, sampled, garment_ref, framing):
        torch.cuda.set_device(self.device)
        pipe, te = self.pipe, self.te_block
        t = time.perf_counter()
        frames = np.zeros((self.ref_len,) + sampled[0][1].shape, np.uint8)  # only the sampled rows are read
        for i, f in sampled:
            frames[i] = f
        refs = [MiniMaxH3VideoReference(frames=frames, fps=float(pipe.fps)), garment_ref]
        with self.te_lock, torch.inference_mode(), torch.cuda.stream(self.pre_stream):
            vi, ic, vc, vt = te._gather_vision_features(pipe.processor, refs, pipe.fps)
            ids, tags = te._build_presentation(pipe.tokenizer, build_prompt(framing, prev=False, task=self.args.task, swatch=self.args.swatch), refs, ic, vc, vt,
                                               text_tag=pipe.text_tag, video_tag=pipe.video_tag)
            emb = enc.get_qwen3vl_prompt_embeds(pipe.text_encoder, pipe.processor, ids, vi,
                                                text_encoder_layer=pipe.text_encoder_layer,
                                                device=pipe._execution_device, dtype=pipe.text_encoder.dtype)
        self.pre_stream.synchronize()  # values ready; record_stream keeps the allocator from reusing the block on the
        emb.record_stream(torch.cuda.default_stream(self.device))  # side stream while the default stream reads it
        self.last_early_s = time.perf_counter() - t
        return emb, torch.tensor(tags, dtype=torch.long)

    def generate(self, index, start, frames, t_first_capture, t_last_capture, t_recv_last, queue_windows, t_ready=None):
        """Generate window `index` (reference = frames[start:start+L]) on this GPU and stream its S new frames.
        Returns gen seconds. Without a prefix L = S = W."""
        torch.cuda.set_device(self.device)
        L, W, K = self.ref_len, self.mirror.S, self.args.prefix_clips  # W = frames emitted per window
        t_gen_start = time.time()
        t0 = time.perf_counter()
        try:
            self.current["moments"] = self.prefix_moments(start) if K else self.ring.window_moments(start, L)
        except KeyError as e:
            print(f"[gpu{self.device}] window {index} at {start}: missing ring clip {e}; have {sorted(self.ring.clips)}",
                  flush=True)
            self.mirror.emit(index, {"type": "window_done", "window": index, "gpu": self.device, "gen_s": 0,
                                     "first_clip_s": 0, "queue_windows": queue_windows, "frames_sent": 0,
                                     "skipped": True, "t_send": time.time()}, [])
            return None
        garment_name, framing = self.mirror.current_config()
        # latent prefix: usable only if it is the previous window's (no catch-up skip in between), same session,
        # same garment (a prefix wearing the old garment would fight the new reference)
        pre = self.mirror.prefix if K else None
        use_pre = (pre is not None and pre["index"] == index - 1 and pre["session"] == self.mirror.session
                   and pre["garment"] == garment_name)
        n_new = W // CLIP
        skip = 0 if (use_pre or not K) else CLIP * K  # without a usable prefix the context clips are generated too
        _PFX.latents = pre["latents"] if use_pre else None
        # next window's prefix = timeline clips [n_new, n_new + K); generated latents start after the clamped prefix
        _PFX.keep = (((n_new - K) * self.lpc, n_new * self.lpc) if use_pre
                     else (n_new * self.lpc, (n_new + K) * self.lpc)) if K else None
        _PFX.saved = None
        t_last_emit = t_last_capture
        if K:
            try:
                t_last_emit = self.mirror.frames[start + CLIP * K + W - 1][1]
            except IndexError:  # session reset meanwhile
                pass
        self.cur = {"index": index, "W": W, "emitted": 0, "skip": skip, "decoded": 0,
                    "t_first_capture": t_first_capture, "t_last_capture": t_last_capture,
                    "t_last_emit_capture": t_last_emit, "t_recv_last": t_recv_last, "t_gen_start": t_gen_start,
                    "t_first_clip": None, "prefix": use_pre}
        ref = MiniMaxH3VideoReference(frames=np.stack(frames), fps=24.0)
        garment = self.mirror.garment_ref(garment_name)
        self.cur["garment"] = garment_name
        self.cur["framing"] = framing
        prev = self.mirror.prev_ref if (self.args.prev_ref and not self.mirror.force_no_prev) else None
        refs = [ref, garment] + ([prev] if prev is not None else [])
        self.cur["prev"] = prev is not None
        self.cur["cond_key"] = self.cond_key(self.mirror.session, index, garment_name, framing, prev is not None)
        self.cur["cond"] = None
        seed = self.args.seed if self.args.same_seed else self.args.seed + index
        prof = self._profiler(index)
        StageTimer.start()
        t_call = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self.pipe(prompt=build_prompt(framing, prev=prev is not None, task=self.args.task, swatch=self.args.swatch), references=refs,
                                height=self.mirror.size[1], width=self.mirror.size[0], num_frames=L,
                                num_inference_steps=self.args.steps + 1, generator=torch.Generator().manual_seed(seed),
                                output_type="pt", output=["videos"])
            saved = _PFX.saved
        finally:
            _PFX.latents = _PFX.keep = _PFX.saved = None
        if K:
            self.mirror.prefix = ({"index": index, "session": self.mirror.session, "garment": garment_name,
                                   "latents": saved} if saved is not None else None)
        t_ret = time.perf_counter()
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        stages = StageTimer.collect()
        attributed = sum(v[0] for v in stages.values())
        stages["pre_pipe"] = [round(t_call - t0, 4), 0.0, 1]  # ring assembly + reference objects
        stages["unattributed"] = [round(t_ret - t_call - attributed, 4), 0.0, 1]  # pipeline glue between blocks
        stages["tail_sync"] = [round(gen_s - (t_ret - t0), 4), 0.0, 1]  # GPU still busy after the pipeline returned
        cur, self.cur = self.cur, None
        if prof is not None:
            self._finish_profile(prof, index, gen_s)
        video = self._video_u8(out["videos"][0])  # (frames, 3, h, w) uint8 on the GPU
        video = video[skip:skip + W]  # the emitted frames (no-op without a prefix)
        self.windows += 1
        if self.args.prev_ref:
            last = video[-1].permute(1, 2, 0).cpu().numpy()
            self.mirror.prev_ref = MiniMaxH3ImageReference(image=Image.fromarray(last))
        first_clip_s = (cur["t_first_clip"] - t_gen_start) if cur["t_first_clip"] else gen_s
        with self.ring.lock:
            for k in [k for k in self.ring.clips if k < start]:
                self.ring.clips.pop(k, None)
            for k in [k for k in self.ring.tails if k < start + L]:
                self.ring.tails.pop(k, None)
        if self.check is not None:  # warm-up self-check of the progressive path against the pipeline output
            vnp = video.permute(0, 2, 3, 1).cpu().numpy()
            prog = np.concatenate(self.check, axis=0) if self.check else np.zeros((0,) + vnp.shape[1:], np.uint8)
            self.check = None
            ok = prog.shape == vnp.shape
            diff = int(np.abs(prog.astype(np.int16) - vnp.astype(np.int16)).max()) if ok else -1
            print(f"[gpu{self.device}] progressive self-check: emitted {prog.shape[0]}/{vnp.shape[0]} frames, "
                  f"max|diff| = {diff}", flush=True)
            if not ok or diff > 4:
                print(f"[gpu{self.device}] progressive output DISABLED (mismatch)", flush=True)
                self.progressive = False
        sent = cur["emitted"] if self.progressive else 0
        if sent < W:  # fallback / remainder: v1 whole-window protocol
            jpegs = self._jpegs(video[sent:].contiguous())
            msg = {"type": "window", "index": index, "window": index, "gpu": self.device, "frames": len(jpegs),
                   "offset": sent, "t_first_capture": t_first_capture, "t_last_capture": t_last_capture,
                   "t_last_emit_capture": cur["t_last_emit_capture"], "t_recv_last": t_recv_last, "t_gen_start": t_gen_start, "t_send": time.time(),
                   "gen_s": round(gen_s, 3), "queue_windows": queue_windows}
            self.mirror.emit(index, msg, jpegs)
        t_ready = t_ready or t_recv_last
        done = {"type": "window_done", "window": index, "gpu": self.device, "gen_s": round(gen_s, 3),
                "first_clip_s": round(first_clip_s, 3), "queue_windows": queue_windows, "frames_sent": W,
                "ready_after_recv_s": round(t_ready - t_recv_last, 3), "start_after_ready_s": round(t_gen_start - t_ready, 3),
                "t_send": time.time()}
        done["garment"], done["framing"], done["prev"] = cur.get("garment"), cur.get("framing"), cur.get("prev")
        done["cond"], done["stages"], done["prefix"] = cur.get("cond"), stages, cur.get("prefix")
        print(f"window {index} [gpu{self.device}] {cur.get('garment')}/{cur.get('framing')}"
              f"{'/prev' if cur.get('prev') else ''}{'/prefix' if cur.get('prefix') else ''}: first clip {first_clip_s:.2f}s "
              f"gen {gen_s:.2f}s queue {queue_windows} "
              f"uplink-last {t_recv_last - t_last_capture:+.2f}s ready+{t_ready - t_recv_last:.2f}s "
              f"start+{t_gen_start - t_ready:.2f}s first-capture-age {time.time() - t_first_capture:.2f}s "
              f"cond={cur.get('cond')}", flush=True)
        if self.args.stage_log:
            print(f"  stages cpu/gpu s: {fmt_stages(stages)}", flush=True)
        self.mirror.record_stages(stages, gen_s, first_clip_s)
        self.mirror.emit(index, done, [])
        return gen_s

    @staticmethod
    def _video_u8(v):
        """Pipeline video output ('pt': (F, 3, h, w) float [0, 1] on the GPU; also tolerates (F, h, w, 3), numpy and
        uint8) -> uint8 (F, 3, h, w) CUDA tensor."""
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        v = v.cuda()
        if v.shape[1] != 3 and v.shape[-1] == 3:
            v = v.permute(0, 3, 1, 2)
        if v.dtype != torch.uint8:
            v = (v.float().clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
        return v.contiguous()

    def _profiler(self, index):
        """torch.profiler around one session window (--profile-window K; warm-up windows are never profiled)."""
        k = self.args.profile_window
        if k is None or index != k or not self.mirror.live:
            return None
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.__enter__()
        print(f"[gpu{self.device}] profiling window {index} ...", flush=True)
        return prof

    def _finish_profile(self, prof, index, gen_s):
        prof.__exit__(None, None, None)
        out = Path(self.args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"trace_gpu{self.device}_s{self.mirror.session}_w{index}.json"
        prof.export_chrome_trace(str(path))
        ka = prof.key_averages()
        dev = sum(getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0) for e in ka) / 1e6
        print(f"[gpu{self.device}] profile of window {index}: gen {gen_s:.3f}s, summed device time {dev:.3f}s "
              f"(~{100 * dev / gen_s:.0f}% of gen; includes side-stream intake encodes) -> {path}", flush=True)
        for key in ("self_device_time_total", "self_cuda_time_total"):
            try:
                print(ka.table(sort_by=key, row_limit=25), flush=True)
                break
            except Exception:  # noqa: BLE001
                continue


class Mirror:
    """Frame intake, window assignment (round-robin over GPUs), output sequencing."""

    def __init__(self, args):
        self.args = args
        self.W = args.window
        self.devices = [int(d) for d in args.gpus.split(",")]
        # window geometry: window k reads frames [k*S, k*S + T) and emits frames [k*S + E0, k*S + E0 + S)
        self.K = args.prefix_clips
        if self.K:
            if args.window % CLIP != 5:
                raise SystemExit(f"--prefix-clips needs a 17n+5 window, got {args.window}")
            if not 1 <= self.K <= args.window // CLIP:
                raise SystemExit(f"--prefix-clips must be 1..{args.window // CLIP} for a {args.window}-frame window")
            if len(self.devices) != 1 or args.prev_ref:
                raise SystemExit("--prefix-clips is single-GPU (window k+1 needs window k's latents) and excludes --prev-ref")
            install_latent_prefix()
        self.S = args.window - 5 if self.K else args.window  # new frames per window (stride)
        self.T = args.window + CLIP * self.K  # timeline = reference frames per window
        self.E0 = CLIP * self.K  # offset of the first emitted frame in the window
        self.prefix = None  # {"index", "session", "garment", "latents"} of the last generated window
        self.frames = []  # (uint8 frame (H, W, 3) or None once consumed, t_capture, t_recv)
        self.lock = threading.RLock()  # re-entrant: catch-up emits from inside _take_window
        self.cv = threading.Condition(self.lock)
        self.next_window = 0  # next window index to be taken by a worker
        self.out_next = 0  # next window index whose output may be sent
        self.out_pending = {}  # window index -> list of (msg, blobs) waiting for their turn
        self.ready_t = {}  # window index -> time its last clip latents were ready
        self.prev_ref = None  # last generated frame of the previous window (MiniMaxH3ImageReference)
        self.force_no_prev = False  # warm-up control
        self.catchup_on = False  # enabled after warm-up
        self.live = False  # True once warm-up is over (profiling, periodic stage summaries)
        self.stage_hist = collections.deque(maxlen=10)  # (stages, gen_s, first_clip_s) of the last windows
        self.skipped = 0
        self.out_q = None  # asyncio queue set per connection
        self.loop = None
        self.stats = {"windows": 0, "last_gen_s": 0.0, "last_first_clip_s": 0.0, "queue_windows": 0}
        self.stop = False
        self.session = 0
        self.canvas = None  # (h, w) reference canvas, derived from the first frame of a session
        self.sizes = canvas_sizes(args.w, args.h, args.aspects)  # output sizes offered; [0] is the default
        self.size = self.sizes[0]  # (w, h) of the current session
        # garment library: default image + everything in --garments-dir (name -> path); references are loaded lazily
        # lipstick task: the "garments" are rendered lip swatches, one per shade (same name -> path library)
        self.framings = framings_for(args.task)
        if args.task == "lipstick":
            self.garments = write_swatches(load_shades(args.shades), Path(args.out_dir) / "shades", self.GARMENT_CANVAS,
                                           args.swatch)
        else:
            self.garments = {"default": args.image}
            gd = Path(args.garments_dir)
            if gd.is_dir():
                for f in sorted(gd.iterdir()):
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        self.garments[f.stem] = str(f)
        self._garment_refs = {}
        first = next(iter(self.garments))
        self.config = {"garment": args.garment if args.garment in self.garments else first, "framing": "auto"}
        print(f"garments: {list(self.garments)}; start config {self.config}", flush=True)
        install_latent_injection()
        self.workers = []
        for d in self.devices:
            print(f"== loading pipeline on GPU {d}", flush=True)
            self.workers.append(GpuWorker(self, d, args))
        torch.cuda.set_device(self.devices[0])

    def worker_for(self, index):
        return self.workers[index % len(self.workers)]

    GARMENT_CANVAS = 1024  # every garment is padded to a square on white and resized to this, so all share one token count

    def garment_ref(self, name):
        if name not in self._garment_refs:
            img = Image.open(self.garments[name]).convert("RGB")
            side = max(img.size)
            sq = Image.new("RGB", (side, side), (255, 255, 255))
            sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            sq = sq.resize((self.GARMENT_CANVAS, self.GARMENT_CANVAS), Image.Resampling.LANCZOS)
            self._garment_refs[name] = MiniMaxH3ImageReference(image=sq)
        return self._garment_refs[name]

    def current_config_dict(self):
        with self.lock:
            return dict(self.config)

    def current_config(self):
        with self.lock:
            return self.config["garment"], self.config["framing"]

    def set_config(self, garment=None, framing=None):
        with self.lock:
            if garment in self.garments:
                self.config["garment"] = garment
            if framing in self.framings:
                self.config["framing"] = framing
            cfg = dict(self.config)
        print(f"config: {cfg}", flush=True)
        return cfg

    def start_workers(self):
        for i, w in enumerate(self.workers):
            w.gen_exec.submit(self._worker_loop, i)  # same thread as this GPU's warm-up (CUDA graphs are per thread)

    # --- output sequencing: clips of window k+1 never overtake window k
    def emit(self, index, msg, blobs):
        with self.lock:
            if index != self.out_next:
                self.out_pending.setdefault(index, []).append((msg, blobs))
                return
            items = [(msg, blobs)]
            while items:
                for m, b in items:
                    if self.loop and self.out_q is not None:
                        self.loop.call_soon_threadsafe(self.out_q.put_nowait, (m, b))
                    if m.get("type") == "window_done":
                        self.stats["windows"] += 1
                        self.stats["last_gen_s"] = m.get("gen_s", 0)
                        self.stats["last_first_clip_s"] = m.get("first_clip_s", 0)
                        self.out_next += 1
                items = self.out_pending.pop(self.out_next, [])
                if items and not any(m.get("type") == "window_done" for m, _ in items):
                    for m, b in items:
                        if self.loop and self.out_q is not None:
                            self.loop.call_soon_threadsafe(self.out_q.put_nowait, (m, b))
                    items = []

    def record_stages(self, stages, gen_s, first_clip_s):
        """Every 10 live windows: mean per-stage cpu/gpu seconds + host load, for A/B runs."""
        if not self.live:
            return
        self.stage_hist.append((stages, gen_s, first_clip_s))
        if self.stats["windows"] % 10 != 9 or len(self.stage_hist) < 5:
            return
        n = len(self.stage_hist)
        mean = {}
        for st, _, _ in self.stage_hist:
            for name, (cpu, gpu, calls) in st.items():
                m = mean.setdefault(name, [0.0, 0.0, 0])
                m[0] += cpu / n
                m[1] += gpu / n
                m[2] = calls
        g = sum(h[1] for h in self.stage_hist) / n
        f = sum(h[2] for h in self.stage_hist) / n
        print(f"STAGES mean of {n} windows: gen {g:.3f}s first clip {f:.3f}s | {fmt_stages(mean)} | "
              f"host {host_info()}", flush=True)

    def reset(self, size=None):
        """New session. size = one of self.sizes (default: the first); a change of size drops the reference canvas."""
        with self.lock:
            size = tuple(size) if size and tuple(size) in self.sizes else self.sizes[0]
            if size != self.size:
                self.size = size
                self.canvas = None
            self.frames = []
            for w in self.workers:
                with w.ring.lock:
                    w.ring.clips.clear()
                    w.ring.tails.clear()
                w.pre.clear()
            self.prefix = None
            self.next_window = 0
            self.out_next = 0
            self.out_pending = {}
            self.ready_t = {}
            self.prev_ref = None
            self.skipped = 0
            self.stats["windows"] = 0
            self.session += 1

    def normalize(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.convert("RGB"))
        if self.canvas is None:
            p = self.workers[0].pipe
            out = be.MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                arr[None], 24.0, 1, p.canvas_multiple, p.config.canvas_short_edge, p.config.canvas_max_pixels, 24.0)
            self.canvas = out.shape[1:3]
            print("reference canvas:", self.canvas, flush=True)
        h, w = self.canvas
        if arr.shape[:2] != (h, w):
            arr = np.asarray(img.convert("RGB").resize((w, h), Image.Resampling.BILINEAR))
        return arr

    def push_frame(self, img: Image.Image, t_capture: float):
        try:
            self._push_frame(img, t_capture)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            raise

    def _push_frame(self, img: Image.Image, t_capture: float):
        arr = self.normalize(img)
        t_recv = time.time()
        with self.lock:
            self.frames.append((arr, t_capture, t_recv))
            idx = len(self.frames) - 1
        if idx % 24 == 0:
            print(f"intake: frame {idx} (capture age {t_recv - t_capture:.2f}s)", flush=True)
        if self.K:
            return self._intake_prefix(idx, t_recv)
        # windows are fixed W-frame blocks from frame 0; encode each completed 17-frame clip (or the tail) on arrival,
        # on the GPU that will generate this window
        k = idx // self.W
        w0 = k * self.W
        rel = idx - w0
        wk = self.worker_for(k)
        if wk.early_cond and wk.sample_idx and rel == wk.sample_idx[-1]:
            with self.lock:
                sampled = [(i, self.frames[w0 + i][0]) for i in wk.sample_idx]
                garment, framing = self.config["garment"], self.config["framing"]
                session = self.session
            if all(f is not None for _, f in sampled):
                wk.schedule_early_cond(session, k, sampled, garment, framing)
        if (rel + 1) % CLIP == 0 or rel == self.W - 1:
            cs = w0 + (rel // CLIP) * CLIP
            ce = min(cs + CLIP, w0 + self.W)
            with self.lock:
                clip = np.stack([f[0] for f in self.frames[cs:ce]])
            dt = self.worker_for(k).encode_clip_side(cs, clip)
            if rel == self.W - 1:
                self.ready_t[k] = time.time()
                if k % 5 == 0:
                    print(f"intake: window {k} ready {self.ready_t[k] - t_recv:.3f}s after its last frame (tail encode {dt:.3f}s)",
                          flush=True)
            with self.cv:
                self.cv.notify_all()  # only after the clip latents exist (the worker races on the last clip otherwise)

    def _intake_prefix(self, idx, t_recv):
        """--prefix-clips intake. Windows overlap (stride S < T), but S is a multiple of 17, so every window's clips sit
        on one global 17-frame grid: each full clip is encoded once as it completes and shared by the windows that
        read it; only a window's 5-frame tail [k*S + T - 5, k*S + T) is its own (padded) encode, kept apart from the
        full clip that later starts at the same frame."""
        wk = self.workers[0]
        if wk.early_cond and wk.sample_idx:
            d = idx - wk.sample_idx[-1]
            if d >= 0 and d % self.S == 0:
                with self.lock:
                    sampled = [(i, self.frames[d + i][0]) for i in wk.sample_idx]
                    garment, framing, session = self.config["garment"], self.config["framing"], self.session
                if all(f is not None for _, f in sampled):
                    wk.schedule_early_cond(session, d // self.S, sampled, garment, framing)
        if (idx + 1) % CLIP == 0:
            cs = idx + 1 - CLIP
            with self.lock:
                clip = np.stack([f[0] for f in self.frames[cs:idx + 1]])
            wk.encode_clip_side(cs, clip)
        d = idx + 1 - self.T
        if d >= 0 and d % self.S == 0:
            k, ts = d // self.S, idx + 1 - self.T % CLIP
            with self.lock:
                clip = np.stack([f[0] for f in self.frames[ts:idx + 1]])
            dt = wk.encode_clip_side(ts, clip, tail=True)
            self.ready_t[k] = time.time()
            if k % 5 == 0:
                print(f"intake: window {k} ready {self.ready_t[k] - t_recv:.3f}s after its last frame (tail encode {dt:.3f}s)",
                      flush=True)
        with self.cv:
            self.cv.notify_all()

    def _window_ready(self, k):
        if self.K:
            r = k * self.S
            return len(self.frames) >= r + self.T and self.workers[0].has_prefix_clips(r)
        start = k * self.W
        if len(self.frames) < start + self.W:
            return False
        return self.worker_for(k).has_clips(start, self.W)

    def _take_window(self, i):
        """Worker i (GPU index) waits for window k with k % N == i to be complete, takes it. Returns None on stop."""
        n = len(self.workers)
        with self.cv:
            while not self.stop:
                k = self.next_window
                if self.catchup_on and n == 1 and self._window_ready(k):
                    # catch-up: if newer complete windows are already waiting, skip the stale ones (a mirror shows now)
                    newest = k
                    while self._window_ready(newest + 1):
                        newest += 1
                    for j in range(k, newest):
                        for f in range(j * self.S, (j + 1) * self.S):
                            self.frames[f] = (None,) + self.frames[f][1:]
                        self.skipped += 1
                        self.emit(j, {"type": "window_done", "window": j, "gpu": -1, "gen_s": 0, "first_clip_s": 0,
                                      "queue_windows": 0, "frames_sent": 0, "skipped": True, "t_send": time.time()}, [])
                    if newest != k:
                        print(f"catch-up: skipped windows {k}..{newest - 1}", flush=True)
                        self.next_window = newest
                        k = newest
                if k % n == i and self._window_ready(k):
                    start = k * self.S
                    frames = [f[0] for f in self.frames[start:start + self.T]]
                    t_first, t_last, t_recv_last = self.frames[start + self.E0][1], self.frames[start + self.T - 1][1], \
                        self.frames[start + self.T - 1][2]
                    self.next_window += 1
                    self.stats["queue_windows"] = (len(self.frames) - self.next_window * self.S - self.T + self.S) // self.S
                    for j in range(start, start + self.S):  # frames no later window reads
                        self.frames[j] = (None,) + self.frames[j][1:]
                    return k, start, frames, t_first, t_last, t_recv_last, self.stats["queue_windows"], self.session,                         self.ready_t.get(k, t_recv_last)
                self.cv.wait(timeout=0.5)
        return None

    def _worker_loop(self, i):
        w = self.workers[i]
        torch.cuda.set_device(w.device)
        while not self.stop:
            item = self._take_window(i)
            if item is None:
                return
            k, start, frames, t_first, t_last, t_recv_last, q, session, t_ready = item
            try:
                w.generate(k, start, frames, t_first, t_last, t_recv_last, q, t_ready)
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
            with self.cv:
                self.cv.notify_all()


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.get("/garments/{name}")
async def garment_file(name: str):
    m: Mirror = STATE["mirror"]
    if name not in m.garments:
        return HTMLResponse("not found", status_code=404)
    return FileResponse(m.garments[name])


@app.get("/stats")
async def stats():
    m: Mirror = STATE["mirror"]
    return {**m.stats, "frames": len(m.frames), "next_window": m.next_window, "out_next": m.out_next,
            "skipped": m.skipped, "prev_ref": m.prev_ref is not None,
            "prefix": None if m.prefix is None else m.prefix["index"],
            "session": m.session, "gpus": m.devices,
            "workers": [{"gpu": w.device, "progressive": w.progressive, "gpu_jpeg": w.gpu_jpeg, "windows": w.windows,
                         "ring_clips": sorted(w.ring.clips)} for w in m.workers]}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    if websocket.query_params.get("token") != STATE["token"]:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    if STATE.get("live_ws"):  # one viewer: a second stream would push its frames into the same session
        await websocket.close(code=1013, reason="busy: another viewer is connected")
        return
    STATE["live_ws"] = True
    m: Mirror = STATE["mirror"]
    try:  # ?cw=&ch= is the camera frame size: the session runs at the offered size closest to that aspect ratio
        size = closest_size(m.sizes, int(websocket.query_params["cw"]), int(websocket.query_params["ch"]))
    except (KeyError, ValueError, ZeroDivisionError):
        size = None
    m.reset(size)
    m.loop = asyncio.get_running_loop()
    m.out_q = asyncio.Queue()
    try:
        await websocket.send_text(json.dumps({"type": "hello", "window": m.S, "gen_frames": m.T, "prefix_clips": m.K,
                                              "fps": FPS, "w": m.size[0], "h": m.size[1], "sizes": m.sizes,
                                              "server_time": time.time(), "progressive": m.workers[0].progressive,
                                              "jpeg_q": m.args.jpeg_q, "protocol": 2, "gpus": len(m.workers),
                                              "garments": list(m.garments), "framings": list(m.framings), "task": m.args.task,
                                              "config": m.current_config_dict(), "same_seed": m.args.same_seed,
                                              "prev_ref": m.args.prev_ref, "catchup": m.args.catchup}))
    except Exception:  # noqa: BLE001  (client left before the hello)
        STATE["live_ws"] = False
        raise

    async def sender():
        while True:
            msg, blobs = await m.out_q.get()
            await websocket.send_text(json.dumps(msg))
            for b in blobs:
                await websocket.send_bytes(b)

    send_task = asyncio.create_task(sender())
    print(f"ws: client connected {websocket.client} size {m.size[0]}x{m.size[1]}", flush=True)
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                txt = msg.get("text") or ""
                if '"ping"' in txt[:20]:
                    t = json.loads(txt)
                    await websocket.send_text(json.dumps({"type": "pong", "t_client": t.get("t_client"),
                                                          "t_server": time.time()}))
                elif '"config"' in txt[:20]:
                    t = json.loads(txt)
                    cfg = m.set_config(t.get("garment"), t.get("framing"))
                    await websocket.send_text(json.dumps({"type": "config", **cfg}))
                else:
                    print(f"ws: ignoring text message {txt[:80]}", flush=True)
                continue
            t_capture, = struct.unpack("<d", data[:8])
            img = Image.open(io.BytesIO(data[8:]))
            img.load()
            await asyncio.get_running_loop().run_in_executor(INTAKE, m.push_frame, img, t_capture)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    finally:
        print(f"ws: client gone after {len(m.frames)} frames", flush=True)
        send_task.cancel()
        m.out_q = None
        STATE["live_ws"] = False


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--image", default="/workspace/inputs/garment.jpg")
    p.add_argument("--garments-dir", default="/workspace/garments")
    p.add_argument("--garment", default="default", help="garment (or, with --task lipstick, shade) name to start with")
    p.add_argument("--task", default="garment", choices=["garment", "lipstick"],
                   help="what <Picture 1> swaps in: a garment image, or a lipstick shade (rendered lip swatch)")
    p.add_argument("--shades", default=None, help="lipstick task: JSON list of {name, hex, finish: matte|satin|gloss} "
                   "(default: 8 built-in shades)")
    p.add_argument("--swatch", default="chip", choices=["chip", "lips"],
                   help="lipstick task: shade reference = shape-free color chip, or drawn lips (got pasted as lips)")
    p.add_argument("--lora", default="minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
    p.add_argument("--video-shift", type=float, default=12.0, help="12 for Turbo v0.1, 6 for the v1.0 768p LoRAs")
    p.add_argument("--audio-shift", type=float, default=3.0)
    p.add_argument("--w", type=int, default=288)
    p.add_argument("--h", type=int, default=512)
    p.add_argument("--aspects", default="", help="extra output aspect ratios, e.g. 16:9,4:3,3:4,1:1: a session runs at "
                   "the one closest to the camera of the client, at the pixel budget of --w x --h (each is compiled in "
                   "the warm-up, ~80 s per aspect per framing). Default: only --w x --h")
    p.add_argument("--ref", type=int, default=224)
    p.add_argument("--window", type=int, default=39)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--gpus", default="0", help="comma-separated CUDA device indices, windows are round-robined")
    p.add_argument("--token", default=None)
    p.add_argument("--jpeg-q", type=int, default=70)
    p.add_argument("--no-progressive", action="store_true")
    p.add_argument("--cpu-jpeg", action="store_true")
    p.add_argument("--warm-framings", default="all", help="comma list of framings to precompile (default all); e.g. auto,selfie")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--same-seed", action="store_true", default=True)
    p.add_argument("--vary-seed", dest="same_seed", action="store_false", help="seed + window index (old behaviour)")
    # off by default since 2026-09-23: on one B300 it cost ~0.7 s per window and did not keep up (report section 11)
    p.add_argument("--prev-ref", action="store_true", default=False, help="feed the previous window's last frame as <Picture 2>")
    p.add_argument("--no-prev-ref", dest="prev_ref", action="store_false")
    p.add_argument("--prefix-clips", type=int, default=0,
                   help="latent prefix: keep the last K generated 17-frame clips of each window as clean conditioning "
                        "at the start of the next (single GPU; windows overlap, stride = window - 5)")
    p.add_argument("--catchup", action="store_true", default=True, help="skip stale complete windows (single GPU)")
    p.add_argument("--no-catchup", dest="catchup", action="store_false")
    # v4 levers
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                   help="torch.compile mode for the transformer; reduce-overhead = CUDA graphs")
    p.add_argument("--threads", type=int, default=int(_THREADS), help="torch intra-op CPU threads (env H3_THREADS "
                   "sets this and the OMP/MKL pools, which must be fixed before import)")
    p.add_argument("--no-tf32", dest="tf32", action="store_false", default=True)
    p.add_argument("--enc-autocast", default="off", choices=["off", "fp16", "bf16"],
                   help="intake VAE encode precision (recipe: fp32); checked against fp32 on the first clip")
    p.add_argument("--no-early-cond", dest="early_cond", action="store_false", default=True,
                   help="run the conditioner in the text-encoder step instead of as soon as its frames arrive")
    p.add_argument("--no-stage-log", dest="stage_log", action="store_false", default=True,
                   help="per-window stage line (the 10-window STAGES summary is always printed)")
    p.add_argument("--profile-window", type=int, default=None, help="torch.profiler trace of this live window index")
    p.add_argument("--out-dir", default="/workspace/outputs/live")
    args = p.parse_args(argv)
    if args.task == "lipstick" and args.prev_ref:
        p.error("--prev-ref is garment-only (its prompt clauses describe the garment)")
    return args


def build_mirror(args):
    """Torch settings + pipeline load on every GPU. Returns the Mirror (also STATE["mirror"]); not yet warmed up."""
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(args.threads, 4))
    except RuntimeError:
        pass
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"host: {host_info()}  tf32={args.tf32} compile_mode={args.compile_mode} enc_autocast={args.enc_autocast} "
          f"early_cond={args.early_cond} prev_ref={args.prev_ref} prefix_clips={args.prefix_clips}", flush=True)
    STATE["token"] = args.token or secrets.token_urlsafe(8)
    m = Mirror(args)
    STATE["mirror"] = m
    if threading.current_thread() is threading.main_thread():  # signal handlers can only be installed from main
        faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid> dumps all thread stacks to stderr
    return m


def warm_up(m, args):
    """Compile every framing shape, self-check progressive decode and the early conditioner, then start the GPU
    worker loops. Blocks for minutes; generation itself runs on each GPU's gen thread (the one the live loop reuses)."""
    # warm-up, in the main thread: every framing prompt is a different token count = a different torch.compile
    # shape (~75 s each, CPU-bound, stalls everything) so compile them ALL now, per GPU; the last (warm) window
    # of each GPU also runs the progressive self-check. Garments share one canvas, so they never recompile.
    n = len(m.workers)
    fr_all = framings_for(args.task)
    warm = [f for f in args.warm_framings.split(",") if f in fr_all] if args.warm_framings != "all" else list(fr_all)
    if "auto" not in warm:
        warm = ["auto"] + warm
    # CUDA graph modes record a graph on the 2nd call of a shape: run every framing twice so none records live
    reps = 2 if args.compile and args.compile_mode in ("reduce-overhead", "max-autotune") else 1
    plan = [f for f in warm for _ in range(reps)] + ["auto"] if args.compile else ["auto", "auto"]
    print(f"warm-up ({n} GPU(s)) sizes {m.sizes} framings {plan} ...", flush=True)
    rng = np.random.default_rng(0)
    total = len(plan) * n
    n_push = (total - 1) * m.S + m.T  # overlapping windows (prefix mode) need T frames for the first one
    for size in m.sizes[1:] + m.sizes[:1]:  # the default size last: its last window runs the progressive self-check
        m.reset(size)
        dummy = rng.integers(0, 255, size=(args.window, size[1], size[0], 3), dtype=np.uint8)
        t = time.perf_counter()
        for i in range(n_push):
            m.push_frame(Image.fromarray(dummy[i % args.window]), time.time())
        print(f"warm-up {size[0]}x{size[1]}: {n_push} frames pushed + encoded in {time.perf_counter() - t:.2f}s",
              flush=True)
        m.force_no_prev = True  # window 0 of every session has no previous frame: compile that shape first
        for k in range(total):
            w = m.worker_for(k)
            framing = plan[k // n]
            m.set_config(framing=framing)
            if k >= n:
                m.force_no_prev = False
            if k // n == len(plan) - 1 and size == m.sizes[0]:
                w.check = []
            item = m._take_window(k % n)
            k_, start, frames, t_first, t_last, t_recv_last, q, _, t_ready = item
            g = w.gen_exec.submit(w.generate, k_, start, frames, t_first, t_last, t_recv_last, q, t_ready).result()
            print(f"warm-up {size[0]}x{size[1]} window {k_} on gpu{w.device} framing={framing} "
                  f"prev={bool(m.args.prev_ref and k >= n)}: {g:.2f}s", flush=True)
        m.set_config(framing="auto")
        m.force_no_prev = False
    # early-conditioner self-check: one more window per GPU, pushed now that the garment's normalized reference is
    # stashed, so the intake schedules the early conditioner; the text-encoder step computes both and compares
    if any(w.early_cond for w in m.workers) and not args.prev_ref:
        base = len(m.frames)
        for i in range(n * m.S):
            m.push_frame(Image.fromarray(dummy[(base + i) % args.window]), time.time())
        for w in m.workers:
            w.pre_check = True
        for k in range(n):
            item = m._take_window(m.next_window % n)
            w = m.worker_for(item[0])
            w.gen_exec.submit(w.generate, *item[:7], item[8]).result()
        for w in m.workers:
            w.pre_check = False
            if w.early_cond and not w.pre_checked:
                print(f"[gpu{w.device}] early conditioner was not exercised in warm-up -> DISABLED", flush=True)
                w.early_cond = False
    m.reset()
    m.catchup_on = bool(args.catchup)
    m.live = True
    m.start_workers()


def main():
    args = parse_args()
    m = build_mirror(args)
    warm_up(m, args)
    print(f"READY  token={STATE['token']}  gpus={m.devices} window={args.window} stride={m.S} timeline={m.T} "
          f"prefix_clips={m.K} "
          f"progressive={[w.progressive for w in m.workers]} gpu_jpeg={[w.gpu_jpeg for w in m.workers]} "
          f"early_cond={[w.early_cond for w in m.workers]} enc={[str(w.enc_dtype) for w in m.workers]} "
          f"compile_mode={args.compile_mode} threads={torch.get_num_threads()} "
          f"jpeg_q={args.jpeg_q}  open: https://<podid>-{args.port}.proxy.runpod.net/?token={STATE['token']}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
