"""Stage-timed MiniMax-H3 Ref2VA (+Turbo LoRA) requests: Qwen3-VL conditioning, reference VAE
encode, denoise (per step), video decode, audio decode.  Run from /workspace/Minimax-H3-Turbo.

  python bench_ref2va.py --video /workspace/inputs/mirror1.mp4 --image /workspace/inputs/garment.jpg \
      --lora minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors --steps 4 \
      --config 960x544x124@768 --config 960x544x124@544 --config 960x544x73@544 --min-duration 3 \
      --runs 2 --out-dir /workspace/outputs/run3 --json /workspace/outputs/run3.json

A config is WIDTHxHEIGHTxFRAMES[@REF_SHORT_EDGE]; REF_SHORT_EDGE overrides the canvas the reference
*video* is resized onto before VAE encoding (model default 768, max pixels 768*1344).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_DIR = Path("/workspace/Minimax-H3-Turbo")
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(Path.cwd()))
from diffusers import ComponentsManager  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import (  # noqa: E402
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.utils.export_utils import encode_video  # noqa: E402
from inference_minimax_h3 import load_lora_adapter  # noqa: E402
from minimax_h3_ref2va_pipeline import load_minimax_h3_ref2va_pipeline  # noqa: E402

# Set H3_MODEL_PATH to the local HF snapshot directory to avoid any hub API calls (rate limits, offline pods).
import os  # noqa: E402

MODEL_ID = os.environ.get("H3_MODEL_PATH", "MiniMaxAI/MiniMax-H3")

DEFAULT_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the person in <Video 1>, preserving their face, hair, skin, body proportions, trousers, "
    "shoes and every movement exactly as in <Video 1>.\n"
    "<Subject 2> is the garment in <Picture 1>, preserving its exact color, fabric, cut, texture and print.\n"
    "<Subject 3> is the location in <Video 1>, preserving its exact background, walls, floor, objects, "
    "camera position, framing and lighting.\n\n"
    "summary:\n"
    "[reference generation] Reproduce <Video 1> shot-for-shot and frame-for-frame: <Subject 1> performs exactly "
    "the same movements at the same timing inside <Subject 3>, seen from exactly the same camera; the only "
    "change is that <Subject 1> is now wearing <Subject 2> instead of the original top.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears throughout): fully_preserved - identical face, hair, body, pose, motion timing and "
    "position in frame as <Video 1>.\n"
    "<Subject 2> (appears throughout): fully_preserved - worn by <Subject 1>, keeps its color, cut, texture "
    "and print, fitted naturally to the body and following the motion.\n"
    "<Subject 3> (appears throughout): fully_preserved - identical background, framing, camera and lighting "
    "as <Video 1>; no mirror, no new objects, no camera movement beyond what <Video 1> has.\n\n"
    "detailed_description:\n"
    "This is the same video as <Video 1> with a garment swap. Same camera, same framing, same background, "
    "same lighting, same person, same walk and same timing. Nothing else changes.\n\n"
    "overall_soundscape:\nquiet room tone.\n"
)


class StageTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.totals, self.calls, self.per_call = {}, {}, {}

    def _add(self, name, dt):
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.calls[name] = self.calls.get(name, 0) + 1
        self.per_call.setdefault(name, []).append(round(dt, 3))

    def wrap(self, obj, attr, name):
        fn = getattr(obj, attr)

        def timed(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            self._add(name, time.perf_counter() - t)
            return out

        setattr(obj, attr, timed)

    def hook_module(self, module, name):
        state = {}

        def pre(m, args, kwargs):
            torch.cuda.synchronize()
            state["t"] = time.perf_counter()

        def post(m, args, kwargs, out):
            torch.cuda.synchronize()
            self._add(name, time.perf_counter() - state["t"])

        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.register_forward_hook(post, with_kwargs=True)

    def report(self):
        return {
            k: {"seconds": round(v, 3), "calls": self.calls[k], "per_call": self.per_call[k]}
            for k, v in self.totals.items()
        }


def parse_config(text):
    """'960x544x124', '960x544x124@544', or key=value form 'w=864,h=480,f=73,ref=480,video=/path.mp4'."""
    if "=" in text:
        kv = dict(item.split("=", 1) for item in text.split(","))
        return {
            "w": int(kv["w"]),
            "h": int(kv["h"]),
            "frames": int(kv["f"]),
            "ref_short_edge": int(kv["ref"]) if "ref" in kv else None,
            "video": kv.get("video"),
            "steps": int(kv["s"]) if "s" in kv else None,
        }
    ref_edge = None
    if "@" in text:
        text, ref_edge = text.split("@", 1)
        ref_edge = int(ref_edge)
    w, h, f = (int(x) for x in text.lower().split("x"))
    return {"w": w, "h": h, "frames": f, "ref_short_edge": ref_edge, "video": None}


def fast_patch_embed(text_encoder):
    """Replace the Qwen3-VL vision patch embedding's Conv3d (kernel == stride == patch extent, so it is a
    plain matmul over each flattened 3x2x16x16 patch) with F.linear. The cuDNN bf16 Conv3d path takes ~85 s
    for ~20k patches on an H200; the matmul takes milliseconds. Numerically equivalent up to accumulation order."""
    import torch.nn.functional as F

    pe = text_encoder.model.visual.patch_embed
    conv = pe.proj
    weight = conv.weight.detach().reshape(conv.out_channels, -1)  # (embed, C*T*H*W), same order as the view below
    bias = conv.bias.detach() if conv.bias is not None else None
    checked = {"done": False}

    def forward(hidden_states):
        x = hidden_states.to(dtype=weight.dtype)  # (N, C*T*H*W) already flattened in (C, T, H, W) order
        out = F.linear(x, weight, bias)
        if not checked["done"]:
            checked["done"] = True
            with torch.no_grad():
                sample = x[:64].view(-1, pe.in_channels, pe.temporal_patch_size, pe.patch_size, pe.patch_size)
                ref = conv(sample).view(-1, pe.embed_dim)
                print(f"fast_patch_embed check: max|linear-conv| = {(out[:64] - ref).abs().max().item():.3e} "
                      f"(scale {ref.abs().max().item():.2f})", flush=True)
        return out

    pe.forward = forward
    print("patched vision patch_embed Conv3d -> F.linear", flush=True)


def install_gpu_resize():
    """Replace the reference-video normalization's per-frame PIL LANCZOS resize (CPU, ~1.2 s for 73 frames) with a
    batched GPU bicubic+antialias resize. Same contract: uint8 (num_frames, height, width, 3) at the canvas the
    clip's own aspect ratio resolves to, resampled to 24 fps and truncated to num_frames first."""
    import math

    import numpy as np
    import torch.nn.functional as F
    from diffusers.modular_pipelines.minimax_h3 import before_encoder as be
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

    cls = be.MiniMaxH3Ref2VASetupStep

    def normalize(frames, fps, num_frames, canvas_multiple, canvas_short_edge, canvas_max_pixels, target_fps):
        if isinstance(frames, list):
            frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
        if isinstance(frames, torch.Tensor):
            frames = frames.movedim(-3, -1).cpu().numpy()
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
        if fps != target_fps:
            scale = target_fps / fps
            slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
            frames = np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)
        frames = frames[:num_frames]
        height, width = resolve_canvas_size(
            frames.shape[2], frames.shape[1], canvas_multiple, canvas_short_edge, canvas_max_pixels
        )
        if frames.shape[1:3] == (height, width):
            return frames
        x = torch.from_numpy(frames).to("cuda", non_blocking=True).permute(0, 3, 1, 2).float()
        y = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
        return y.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    cls._normalize_video_condition = staticmethod(normalize)
    print("reference video resize: GPU bicubic+antialias (replaces per-frame PIL LANCZOS)", flush=True)


def install_block_timing(pipe, store):
    """Time every leaf block of the modular pipeline (patches each leaf block class's __call__)."""
    patched = set()

    def leaves(block):
        subs = getattr(block, "sub_blocks", None)
        if subs:
            for sub in (subs.values() if hasattr(subs, "values") else list(subs)):
                yield from leaves(sub)
        else:
            yield block

    for leaf in leaves(pipe.blocks):
        cls = type(leaf)
        if cls in patched:
            continue
        patched.add(cls)
        orig = cls.__call__

        def make(orig, name):
            def timed_call(self, *a, **k):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = orig(self, *a, **k)
                torch.cuda.synchronize()
                store[name] = store.get(name, 0.0) + time.perf_counter() - t
                return out

            return timed_call

        cls.__call__ = make(orig, cls.__name__)
    print(f"block timing installed on {len(patched)} block classes", flush=True)


def pin_module(module, name):
    """Materialize a (possibly mmapped) module in pinned host RAM so device moves run at PCIe speed."""
    t = time.perf_counter()
    n_bytes = 0
    for p in list(module.parameters()) + list(module.buffers()):
        if p.device.type == "cpu" and not p.is_pinned():
            p.data = p.data.pin_memory()
        n_bytes += p.numel() * p.element_size()
    print(f"pinned {name}: {n_bytes / 1e9:.1f} GB in {time.perf_counter() - t:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--video-shift", type=float, default=12.0)
    p.add_argument("--audio-shift", type=float, default=3.0)
    p.add_argument("--config", action="append", default=None, help="WxHxF[@refShortEdge], repeatable")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--attention", default="_flash_3_hub")
    p.add_argument(
        "--te-attn",
        default=None,
        help="attn_implementation for the Qwen3-VL conditioner (e.g. kernels-community/flash-attn3, sdpa). "
        "Without flash, its vision tower takes ~90 s per request.",
    )
    p.add_argument(
        "--reserve",
        default="40GB",
        help="GPU memory the ComponentsManager keeps free; 40GB forces the 62GB text encoder off before the "
        "62GB transformer comes on, avoiding offload thrash on a 141GB card.",
    )
    p.add_argument(
        "--offload",
        action="store_true",
        help="Use the ComponentsManager CPU-offload path (bf16 text encoder swapped in/out) instead of the "
        "default resident path (8-bit text encoder + bf16 transformer/VAEs all on the GPU).",
    )
    p.add_argument("--pin", action="store_true", help="Offload path only: pin weights in host RAM (needs >250GB).")
    p.add_argument("--no-cap-vlm-pixels", action="store_true", help="Don't cap Qwen3-VL per-frame pixels.")
    p.add_argument("--no-fast-patch-embed", action="store_true",
                   help="Keep the stock Conv3d vision patch embedding (~85 s per request on H200).")
    p.add_argument("--vlm-fps", type=float, default=None,
                   help="Rate the Qwen3-VL conditioner samples the reference video at (block default 2.0). "
                   "Fewer samples = fewer conditioner tokens in the denoised sequence.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer's forward (first pass of each config pays compilation).")
    p.add_argument("--block-timing", action="store_true", help="Also report seconds per modular-pipeline block.")
    p.add_argument("--gpu-resize", action="store_true",
                   help="Resize the reference video on the GPU (bicubic+antialias) instead of per-frame PIL LANCZOS.")
    p.add_argument("--te-mode", default="int8",
                   help="Resident conditioner: int8 | int8-trunc51 | nf4 | nf4-trunc51 | bf16 | bf16-trunc51 "
                   "(measured text-encoder step: int8 0.96 s, nf4-trunc51 0.23 s, bf16 0.19 s).")
    p.add_argument("--out-dir", default="/workspace/outputs")
    p.add_argument("--json", default=None)
    p.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Override the pipeline's 5 s floor (experimental: lets num_frames < 124 through).",
    )
    args = p.parse_args()

    configs = [parse_config(c) for c in args.config] if args.config else [
        {"w": args.width, "h": args.height, "frames": args.frames, "ref_short_edge": None}
    ]
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else DEFAULT_PROMPT

    t0 = time.perf_counter()
    if args.offload:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=args.reserve)
        pipe = load_minimax_h3_ref2va_pipeline(MODEL_ID, components_manager=manager)
        pipe.load_components(workflow="ref2va", dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_ID)
    else:
        # Resident path: the 62 GB bf16 Qwen3-VL conditioner is replaced by an 8-bit copy (~36 GB) loaded
        # straight onto the GPU, so conditioner + transformer_ref + VAEs (~115 GB) all fit on one 141 GB card
        # and no weights move between host and device during a request.
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        pipe = load_minimax_h3_ref2va_pipeline(MODEL_ID)
        # quantized text encoder first: load_components only loads components that are still unset, so the bf16
        # copy is never read just to be replaced (as in stream_sim.build_pipeline)
        t_te = time.perf_counter()
        te_kwargs = {"attn_implementation": args.te_attn} if args.te_attn else {}
        skip = ["visual", "embed_tokens", "lm_head"]
        mode = args.te_mode
        if mode.startswith("int8"):
            te_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        elif mode.startswith("nf4"):
            te_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip)
        elif not mode.startswith("bf16"):
            raise ValueError(f"unknown --te-mode {mode}")
        text_encoder_8bit = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, subfolder="text_encoder", dtype=torch.bfloat16, device_map={"": 0}, **te_kwargs
        )
        if mode.endswith("-trunc51"):
            # MiniMax-H3 reads hidden_states[50]; decoder layers 51..63 never influence it.
            lm = text_encoder_8bit.model.language_model
            lm.layers = lm.layers[:51]
            text_encoder_8bit.config.text_config.num_hidden_layers = 51
            lm.config.num_hidden_layers = 51
        print(f"text encoder mode: {mode}", flush=True)
        print("text encoder attn:", text_encoder_8bit.config._attn_implementation,
              "vision attn:", getattr(text_encoder_8bit.config.vision_config, "_attn_implementation", None),
              flush=True)
        text_encoder_8bit.requires_grad_(False)
        text_encoder_8bit.eval()
        pipe.update_components(text_encoder=text_encoder_8bit)
        print(f"8-bit text encoder loaded on GPU in {time.perf_counter() - t_te:.1f}s, "
              f"vram now {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        pipe.load_components(workflow="ref2va", dtype=torch.bfloat16, pretrained_model_name_or_path=MODEL_ID)
        if pipe.text_encoder is not text_encoder_8bit:
            raise RuntimeError("load_components replaced the quantized text encoder")
        for name in ("transformer_ref", "vae", "audio_vae"):
            t_m = time.perf_counter()
            getattr(pipe, name).to("cuda")
            print(f"{name} -> cuda in {time.perf_counter() - t_m:.1f}s, vram {torch.cuda.memory_allocated() / 1e9:.1f} GB",
                  flush=True)
    pipe.scheduler.set_shift(args.video_shift)
    pipe.audio_scheduler.set_shift(args.audio_shift)
    if args.min_duration is not None:
        type(pipe).min_duration = property(lambda self: args.min_duration)
        print(f"min_duration overridden to {pipe.min_duration}s")
    tr = pipe.transformer_ref
    if args.lora:
        load_lora_adapter(
            transformer=tr, lora_path=Path(args.lora), lora_alpha=args.lora_alpha, lora_scale=1.0, fuse_lora=True
        )
    if args.attention and args.attention != "none":
        try:
            tr.set_attention_backend(args.attention)
            print("attention backend:", args.attention)
        except Exception as e:  # noqa: BLE001
            print("attention backend fallback (default sdpa):", repr(e))
    tr.requires_grad_(False)
    tr.eval()
    if not args.no_cap_vlm_pixels:
        pipe.processor.video_processor.cap_pixels_per_frame = True
        print("Qwen3-VL video processor: cap_pixels_per_frame=True")
    if not args.no_fast_patch_embed:
        fast_patch_embed(pipe.text_encoder)
    if args.vlm_fps is not None:
        n = 0
        stack = [pipe.blocks]
        seen = set()
        while stack:
            blk = stack.pop()
            if id(blk) in seen:
                continue
            seen.add(id(blk))
            if hasattr(blk, "video_sample_fps"):
                blk.video_sample_fps = args.vlm_fps
                n += 1
            subs = getattr(blk, "sub_blocks", None)
            if subs:
                stack.extend(subs.values() if hasattr(subs, "values") else list(subs))
        print(f"conditioner video_sample_fps set to {args.vlm_fps} on {n} block(s)", flush=True)
    if args.compile:
        tr.forward = torch.compile(tr.forward, dynamic=False)
        print("transformer forward compiled (torch.compile, dynamic=False)", flush=True)
    print(f"load time {time.perf_counter() - t0:.1f}s", flush=True)
    if args.offload and args.pin:
        for name in ("text_encoder", "transformer_ref", "vae", "audio_vae"):
            pin_module(getattr(pipe, name), name)
    print(f"execution device: {pipe._execution_device}", flush=True)
    default_canvas = (pipe.config.canvas_short_edge, pipe.config.canvas_max_pixels)
    print("default reference canvas:", default_canvas)

    timer = StageTimer()
    block_times = {}
    if args.block_timing:
        install_block_timing(pipe, block_times)
    if args.gpu_resize:
        install_gpu_resize()
    # diffusers calls `text_encoder.model(...)` directly (reads hidden_states[50]), so hook the inner model.
    timer.hook_module(pipe.text_encoder.model, "conditioner_qwen3vl")
    timer.hook_module(tr, "denoise_transformer")
    timer.wrap(pipe.vae, "encode", "video_vae_encode")
    timer.wrap(pipe.vae, "decode", "video_vae_decode")
    timer.wrap(pipe.audio_vae, "encode", "audio_vae_encode")
    timer.wrap(pipe.audio_vae, "decode", "audio_vae_decode")

    garment = MiniMaxH3ImageReference.from_file(args.image)
    ref_cache = {}

    def references_for(video_path):
        if video_path not in ref_cache:
            t = time.perf_counter()
            video_ref = MiniMaxH3VideoReference.from_file(video_path)
            video_ref.audio = None  # motion only: no soundtrack conditioning
            video_ref.sample_rate = None
            print(f"reference decode {video_path}: {time.perf_counter() - t:.2f}s frames={len(video_ref.frames)} "
                  f"fps={video_ref.fps}", flush=True)
            ref_cache[video_path] = [video_ref, garment]
        return ref_cache[video_path]

    results = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        if cfg["ref_short_edge"] is not None:
            edge = cfg["ref_short_edge"]
            pipe.register_to_config(canvas_short_edge=edge, canvas_max_pixels=round(edge * edge * 1344 / 768))
        else:
            pipe.register_to_config(canvas_short_edge=default_canvas[0], canvas_max_pixels=default_canvas[1])
        video_path = cfg.get("video") or args.video
        refs = references_for(video_path)
        in_frames = len(refs[0].frames)
        steps = cfg.get("steps") or args.steps
        tag = (f"{cfg['w']}x{cfg['h']}x{cfg['frames']}_ref{pipe.config.canvas_short_edge}"
               f"_in{in_frames}f_s{steps}")
        print(f"=== config {tag}  reference canvas={pipe.config.canvas_short_edge}/{pipe.config.canvas_max_pixels} "
              f"video={video_path}", flush=True)
        out = None
        for run in range(args.runs):
            timer.reset()
            block_times.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                with torch.inference_mode():
                    out = pipe(
                        prompt=prompt,
                        references=refs,
                        height=cfg["h"],
                        width=cfg["w"],
                        num_frames=cfg["frames"],
                        num_inference_steps=steps + 1,
                        generator=torch.Generator().manual_seed(42 + run),
                        output_type="np",
                        output=["videos", "audio", "sampling_rate"],
                    )
            except Exception as e:  # noqa: BLE001
                print(f"config {tag} run {run} FAILED: {e!r}", flush=True)
                results.append({"config": tag, "run": run, "error": repr(e)})
                break
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            rep = timer.report()
            other = wall - sum(v["seconds"] for v in rep.values())
            rec = {
                "config": tag,
                "run": run,
                "wall_s": round(wall, 3),
                "stages": rep,
                "unattributed_s": round(other, 3),
                "blocks_s": {k: round(v, 3) for k, v in sorted(block_times.items(), key=lambda kv: -kv[1])},
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                "settings": {**cfg, "video": video_path, "input_frames": in_frames, "te_attn": args.te_attn,
                             "vlm_fps": args.vlm_fps, "compile": args.compile, "gpu_resize": args.gpu_resize,
                             "te_mode": args.te_mode,
                             "steps": steps, "lora": args.lora, "video_shift": args.video_shift,
                             "attention": args.attention, "ref_canvas": pipe.config.canvas_short_edge,
                             "mode": "offload" if args.offload else "resident_te8bit",
                             "cap_vlm_pixels": not args.no_cap_vlm_pixels},
            }
            print(json.dumps(rec, indent=1), flush=True)
            results.append(rec)
            if args.json:
                Path(args.json).write_text(json.dumps(results, indent=1))
        if out is not None:
            video = out["videos"][0]
            path = out_dir / f"{tag}.mp4"
            encode_video(video, fps=24, output_path=str(path), audio=out["audio"][0],
                         audio_sample_rate=out["sampling_rate"])
            print(f"saved {path} shape={getattr(video, 'shape', None)}", flush=True)
    print("ALL CONFIGS DONE", flush=True)


if __name__ == "__main__":
    main()
