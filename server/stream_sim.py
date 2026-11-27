"""Simulated live feed for the mirror loop, on one GPU.

The source clip (looped) is "captured" at real 24 fps by a feed thread. Every 17 captured frames form a VAE clip
that the feed thread encodes on arrival (competing for the same GPU as the generation, as in production). When a
window of W = 17n+5 frames is complete, the main thread assembles the reference latent from the ring buffer (plus
the padded 5-frame tail, the only encode left on the critical path), runs conditioner + denoise + decode, and
records:
  P_first = time from the window's last captured frame to the first decoded 17-frame clip (what a player shows)
  P_total = time to the whole decoded window
  lag     = W/24 + P_first  (a real-world moment is displayed this long after it happened)
  GPU busy per window = P_total + chunk encodes during capture (keep-up needs this <= W/24)
No frame is touched before it has "arrived": nothing is precomputed from the future.

  cd /workspace/Minimax-H3-Turbo && python stream_sim.py --video /workspace/inputs/mirror1.mp4 \
      --image /workspace/inputs/garment.jpg --window 39 --windows 6 --w 320 --h 576 --ref 256 --steps 4 --compile
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/Minimax-H3-Turbo")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import before_encoder as be  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import encoders as enc  # noqa: E402
from diffusers.utils.export_utils import encode_video  # noqa: E402
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration  # noqa: E402

from bench_ref2va import DEFAULT_PROMPT, fast_patch_embed, install_gpu_resize  # noqa: E402
from inference_minimax_h3 import load_lora_adapter  # noqa: E402
from minimax_h3_ref2va_pipeline import load_minimax_h3_ref2va_pipeline  # noqa: E402

MODEL_ID = os.environ.get("H3_MODEL_PATH", "MiniMaxAI/MiniMax-H3")
FPS = 24
CLIP = 17


def build_pipeline(args):
    t0 = time.perf_counter()
    pipe = load_minimax_h3_ref2va_pipeline(MODEL_ID)
    # The nf4 text encoder is registered BEFORE load_components: it only loads components that are still unset, so
    # the bf16 text encoder (62 GB) is never read just to be replaced (on a slow volume that read cost ~13 min).
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               llm_int8_skip_modules=["visual", "embed_tokens", "lm_head"]))
    lm = te.model.language_model
    lm.layers = lm.layers[:51]
    te.config.text_config.num_hidden_layers = 51
    lm.config.num_hidden_layers = 51
    te.requires_grad_(False)
    te.eval()
    pipe.update_components(text_encoder=te)
    pipe.load_components(workflow="ref2va", dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_ID)
    if pipe.text_encoder is not te:
        raise RuntimeError("load_components replaced the nf4 text encoder")
    for name in ("transformer_ref", "vae", "audio_vae"):
        getattr(pipe, name).to("cuda")
    # Turbo v0.1 (544p) was distilled at video shift 12; the v1.0 768p LoRAs at 6 (audio 3 for both).
    video_shift, audio_shift = getattr(args, "video_shift", 12.0), getattr(args, "audio_shift", 3.0)
    pipe.scheduler.set_shift(video_shift)
    pipe.audio_scheduler.set_shift(audio_shift)
    print(f"lora: {Path(args.lora).name}  shift video {video_shift} / audio {audio_shift}", flush=True)
    type(pipe).min_duration = property(lambda self: 0.5)
    tr = pipe.transformer_ref
    load_lora_adapter(transformer=tr, lora_path=Path(args.lora), lora_alpha=8, lora_scale=1.0, fuse_lora=True)
    # Hopper: FA3 hub kernel. Blackwell (B200/B300, sm_100): try the FA4 hub kernel first; else SDPA (cuDNN flash).
    backends = [b for b in os.environ.get("H3_ATTN", "_flash_4_hub,_flash_3_hub").split(",") if b]
    chosen = "sdpa (default)"
    for b in backends:
        try:
            tr.set_attention_backend(b)
            chosen = b
            break
        except Exception as e:  # noqa: BLE001
            print(f"attention backend {b} unavailable: {e!r}"[:200], flush=True)
    print(f"attention backend: {chosen}", flush=True)
    tr.requires_grad_(False)
    tr.eval()
    pipe.processor.video_processor.cap_pixels_per_frame = True
    fast_patch_embed(pipe.text_encoder)
    install_gpu_resize()
    stack, seen = [pipe.blocks], set()
    while stack:
        b = stack.pop()
        if id(b) in seen:
            continue
        seen.add(id(b))
        if hasattr(b, "video_sample_fps"):
            b.video_sample_fps = 1.0
        subs = getattr(b, "sub_blocks", None)
        if subs:
            stack.extend(subs.values() if hasattr(subs, "values") else list(subs))
    if args.compile:
        # mode: "default" (no CUDA graphs, what every run up to 2026-09-22 used), "reduce-overhead" (CUDA graphs: one
        # graph per static shape, removes per-kernel launch cost on busy hosts), "max-autotune[-no-cudagraphs]".
        mode = getattr(args, "compile_mode", "default")
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)  # framings x prev shapes
        tr.forward = torch.compile(tr.forward, dynamic=False, mode=None if mode == "default" else mode)
        print(f"torch.compile mode: {mode}", flush=True)
    pipe.register_to_config(canvas_short_edge=args.ref, canvas_max_pixels=round(args.ref * args.ref * 1344 / 768))
    print(f"pipeline ready in {time.perf_counter() - t0:.0f}s, vram {torch.cuda.memory_allocated() / 1e9:.1f} GB",
          flush=True)
    return pipe


class LatentRing:
    """Per-clip VAE moments encoded as frames arrive; assemble the window's reference latent at window end."""

    def __init__(self, vae, pixel_mean, pixel_std):
        self.vae = vae
        self.pm = torch.tensor(pixel_mean, device="cuda").view(1, -1, 1, 1, 1)
        self.ps = torch.tensor(pixel_std, device="cuda").view(1, -1, 1, 1, 1)
        self.clips = {}  # start frame index -> moments (1, 48, 5, h, w)
        self.encode_time = 0.0
        self.lock = threading.Lock()

    def _prep(self, frames_u8):
        x = torch.from_numpy(frames_u8).to("cuda").permute(3, 0, 1, 2)[None].float().div(255.0)
        return ((x - self.pm) / self.ps).to(next(self.vae.parameters()).dtype)

    @torch.inference_mode()
    def encode_clip(self, start, frames_u8):
        if frames_u8.shape[0] < CLIP:  # padded tail, like _encode does
            frames_u8 = np.concatenate([frames_u8, np.repeat(frames_u8[-1:], CLIP - frames_u8.shape[0], axis=0)])
        torch.cuda.synchronize()
        t = time.perf_counter()
        m = self.vae._encode_clip(self._prep(frames_u8))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        with self.lock:
            self.clips[start] = m
            self.encode_time += dt
        return dt

    def window_moments(self, start, n_frames):
        starts = list(range(start, start + n_frames, CLIP))
        with self.lock:
            parts = [self.clips[s] for s in starts]
        m = torch.cat(parts, dim=2)
        drop = self.vae.config.token_drop
        return m[:, :, : m.shape[2] - drop] if drop else m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--lora", default="minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
    p.add_argument("--video-shift", type=float, default=12.0, help="12 for Turbo v0.1, 6 for the v1.0 768p LoRAs")
    p.add_argument("--audio-shift", type=float, default=3.0)
    p.add_argument("--tag", default="", help="suffix for the output file names (e.g. the LoRA under test)")
    p.add_argument("--window", type=int, default=39, help="frames per generated window (17n+5)")
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--w", type=int, default=320)
    p.add_argument("--h", type=int, default=576)
    p.add_argument("--ref", type=int, default=256)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-realtime", action="store_true", help="deliver frames as fast as possible (no sleeps)")
    p.add_argument("--out", default="/workspace/outputs/stream")
    args = p.parse_args()
    W = args.window
    assert (W - 5) % CLIP == 0, "window must be 17n+5"

    pipe = build_pipeline(args)
    vae = pipe.vae

    # Source feed: the clip looped, pre-resized to the reference canvas (= capture at canvas size).
    vref = MiniMaxH3VideoReference.from_file(args.video)
    src = vref.frames
    need = W * args.windows
    reps = -(-need // src.shape[0])
    feed = np.concatenate([src] * reps, axis=0)[:need]
    feed = be.MiniMaxH3Ref2VASetupStep._normalize_video_condition(
        feed, 24.0, need, pipe.canvas_multiple, pipe.config.canvas_short_edge, pipe.config.canvas_max_pixels, 24.0)
    print(f"feed: {feed.shape} ({need / FPS:.1f} s, {args.windows} windows of {W} frames = {W / FPS:.2f} s)",
          flush=True)
    garment = MiniMaxH3ImageReference.from_file(args.image)
    ring = LatentRing(vae, pipe.pixel_mean, pipe.pixel_std)

    # --- garment latent library + ring-buffer injection into the reference encoder step
    orig_encode_cond = enc.encode_vae_condition
    garment_cache = {}
    current = {"moments": None}
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    def encode_cond(vae_, pixels, pixel_mean, pixel_std, encode_seed=42):
        if pixels.shape[2] == 1:  # garment image: encode once, reuse (garment library)
            key = (pixels.shape, int(pixels.float().sum().item()))
            if key not in garment_cache:
                garment_cache[key] = orig_encode_cond(vae_, pixels, pixel_mean, pixel_std, encode_seed)
            return garment_cache[key]
        moments = current["moments"]
        assert moments is not None
        posterior = DiagonalGaussianDistribution(moments)
        lat = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
        lat = lat.to(torch.float16).float().cpu()
        return (lat - latents_mean) / latents_std

    enc.encode_vae_condition = encode_cond

    # --- first-clip decode hook (what a streaming player would show first)
    first_clip = {"t": None}
    orig_decode_clip = vae._decode_clip

    def decode_clip(z):
        out = orig_decode_clip(z)
        if first_clip["t"] is None:
            torch.cuda.synchronize()
            first_clip["t"] = time.perf_counter()
        return out

    vae._decode_clip = decode_clip

    def generate(start, gen_seed):
        current["moments"] = ring.window_moments(start, W)
        first_clip["t"] = None
        ref = MiniMaxH3VideoReference(frames=feed[start : start + W], fps=24.0)
        with torch.inference_mode():
            out = pipe(prompt=DEFAULT_PROMPT, references=[ref, garment], height=args.h, width=args.w,
                       num_frames=W, num_inference_steps=args.steps + 1,
                       generator=torch.Generator().manual_seed(gen_seed), output_type="np",
                       output=["videos", "audio", "sampling_rate"])
        torch.cuda.synchronize()
        return out

    # warm-up (pays compilation) on window 0's frames, encoded directly (not part of the measurement)
    print("warm-up ...", flush=True)
    for s in range(0, W, CLIP):
        ring.encode_clip(s, feed[s : min(s + CLIP, W)])
    t = time.perf_counter()
    generate(0, 1)
    print(f"warm-up done in {time.perf_counter() - t:.1f}s", flush=True)
    ring.clips.clear()
    ring.encode_time = 0.0

    # --- feed thread: frames arrive at 24 fps; clips encoded on arrival; windows announced when complete
    ready = []  # (window index, start, t_last_frame_arrival, encode seconds spent on this window)
    ready_cv = threading.Condition()
    t0 = time.perf_counter()

    def feed_thread():
        enc_spent = 0.0
        for i in range(need):
            if not args.no_realtime:
                due = t0 + i / FPS
                now = time.perf_counter()
                if due > now:
                    time.sleep(due - now)
            wstart = (i // W) * W
            rel = i - wstart
            # a clip [wstart+17k, wstart+17k+17) completes at rel = 17k+16; the 5-frame tail at rel = W-1
            if (rel + 1) % CLIP == 0 or rel == W - 1:
                cs = wstart + (rel // CLIP) * CLIP
                enc_spent += ring.encode_clip(cs, feed[cs : min(cs + CLIP, wstart + W)])
            if rel == W - 1:
                with ready_cv:
                    ready.append((i // W, wstart, time.perf_counter(), enc_spent))
                    ready_cv.notify()
                enc_spent = 0.0

    th = threading.Thread(target=feed_thread, daemon=True)
    th.start()

    results, videos = [], []
    for k in range(args.windows):
        with ready_cv:
            while len(ready) <= k:
                ready_cv.wait()
            widx, wstart, t_arrived, enc_spent = ready[k]
        t_start = time.perf_counter()
        wait_for_gpu = t_start - t_arrived  # >0 means the previous window was still being processed
        out = generate(wstart, 42 + k)
        t_done = time.perf_counter()
        p_first = first_clip["t"] - t_arrived
        p_total = t_done - t_arrived
        gpu_busy = (t_done - t_start) + enc_spent
        rec = {"window": k, "start_frame": wstart, "queue_wait_s": round(wait_for_gpu, 3),
               "P_first_s": round(p_first, 3), "P_total_s": round(p_total, 3), "encode_during_capture_s": round(enc_spent, 3),
               "gpu_busy_s": round(gpu_busy, 3), "window_s": round(W / FPS, 3),
               "lag_first_frame_s": round(W / FPS + p_first, 3), "keeps_up": gpu_busy <= W / FPS}
        print(json.dumps(rec), flush=True)
        results.append(rec)
        videos.append(out["videos"][0])
    th.join()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = f"{args.w}x{args.h}_W{W}_ref{args.ref}_s{args.steps}{'_compiled' if args.compile else ''}{'_' + args.tag if args.tag else ''}"
    stitched = np.concatenate(videos, axis=0)
    encode_video(stitched, fps=FPS, output_path=f"{args.out}/{tag}_stitched.mp4")
    Path(f"{args.out}/{tag}.json").write_text(json.dumps(results, indent=1))
    steady = results[1:] if len(results) > 1 else results
    avg = lambda key: sum(r[key] for r in steady) / len(steady)
    print(f"STEADY STATE ({len(steady)} windows): P_first {avg('P_first_s'):.2f}s  P_total {avg('P_total_s'):.2f}s  "
          f"gpu_busy {avg('gpu_busy_s'):.2f}s per {W / FPS:.2f}s window  lag {avg('lag_first_frame_s'):.2f}s  "
          f"keeps_up={all(r['keeps_up'] for r in steady)}", flush=True)
    print("SIM DONE", flush=True)


if __name__ == "__main__":
    main()
