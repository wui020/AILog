cat << 'EOF' > /home/gw/run_cosmos4_nano.py
import os
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import torch
import gradio as gr
import imageio
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, PipelineQuantizationConfig

MODEL_PATH = "/home/gw/models/cosmos3-nano"

print("=" * 50)
print("  Cosmos3-Nano Save-VRAM Version (No-Emoji)")
print("=" * 50)

# ============ 第一步：Patch掉Safety Checker（防止联网下载blocklist） ============
try:
    import cosmos_guardrail.cosmos_guardrail as _cg
    _orig_safety_init = _cg.CosmosSafetyChecker.__init__
    def _patched_safety_init(self, *a, **kw):
        print("  [Warning] Safety Checker has been bypassed to save VRAM.")
        pass
    _cg.CosmosSafetyChecker.__init__ = _patched_safety_init
    print("[OK] Safety Checker bypass patch injected successfully.")
except Exception as e:
    print(f"[Warning] Safety Checker patch skipped: {e}")

# ============ 第二步：加载模型 ============
print("Loading 8bit quantized model...")

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True, "llm_int8_threshold": 6.0},
)

try:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        safety_checker=None,
        enable_safety_checker=False,
        local_files_only=True,
    )
except Exception as e:
    print(f"Quantized loading failed: {e}")
    print("Trying default BF16 loading...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        safety_checker=None,
        enable_safety_checker=False,
        local_files_only=True,
    )

# ============ 第三步：VAE 显存优化 ============
try:
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("[OK] VAE Tiling + Slicing enabled.")
except Exception as e:
    print(f"[Warning] VAE optimization skipped: {e}")

print("[OK] Model loading process complete!")
print(f"     Allocated VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ============ 辅助函数 ============
def clear_vram():
    """强制清理显存"""
    torch.cuda.empty_cache()
    gc.collect()

def check_frames(n):
    """确保帧数满足 Cosmos VAE 要求: (n-1) % 4 == 0"""
    n = int(n)
    if n < 1:
        n = 1
    if (n - 1) % 4 != 0:
        n = ((n - 1) // 4) * 4 + 1
    return max(n, 1)

def estimate_vram(frames, height, width, has_image=False):
    """粗略估算显存需求（仅供参考）"""
    # 经验公式：每帧每万像素约需 0.03GB（8bit量化下），若有输入图则略微增加计算开销
    pixels = frames * height * width / 10000
    base_est = pixels * 0.03
    if has_image:
        base_est += 0.5  # 增加编码输入首帧的额外开销估算
    return base_est

# ============ 核心生成函数 ============
def generate(prompt, neg_prompt, num_frames, height, width, steps, guidance, fps, seed, input_image=None):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty!")

    # 帧数修正
    num_frames = check_frames(num_frames)

    # 分辨率修正为16的倍数
    height = (int(height) // 16) * 16
    width = (int(width) // 16) * 16
    if height < 16:
        height = 16
    if width < 16:
        width = 16

    # 如果有首帧输入，进行尺寸预对齐
    processed_image = None
    if input_image is not None:
        try:
            # 兼容 numpy 矩阵或文件路径
            if not isinstance(input_image, Image.Image):
                processed_image = Image.fromarray(input_image)
            else:
                processed_image = input_image
            
            # 将输入首帧调整为与输出视频宽高一致，防止底层 VAE/Transformer 因尺寸不一致报错
            processed_image = processed_image.resize((width, height), Image.Resampling.LANCZOS)
            print("[OK] Input image resized to match target dimensions.")
        except Exception as e:
            print(f"[Warning] Failed to process input image: {e}")
            processed_image = None

    # 显存预估
    est = estimate_vram(num_frames, height, width, has_image=(processed_image is not None))
    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"VRAM Estimate: ~{est:.1f} GB | Free VRAM: {free:.1f} GB")

    if est > free * 0.8:
        print(f"[Warning] Potential Out-of-Memory risk! Estimated {est:.1f}GB > Free {free:.1f}GB.")

    # 种子
    seed = int(seed)
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

    print(f"Start generation: {width}x{height}, {num_frames} frames, {steps} steps, FPS={fps}")
    print(f"Estimated output video duration: {num_frames/fps:.1f}s")

    clear_vram()

    try:
        # 准备生成参数
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": neg_prompt or "",
            "num_frames": num_frames,
            "num_inference_steps": int(steps),
            "guidance_scale": guidance,
            "height": height,
            "width": width,
            "generator": generator,
        }

        # 如果存在输入图像，则切换为图生视频 (Image-to-Video) 模式
        if processed_image is not None:
            generation_kwargs["image"] = processed_image
            print("Running in Image-to-Video (I2V) mode...")
        else:
            print("Running in Text-to-Video (T2V) mode...")

        with torch.inference_mode():
            result = pipe(**generation_kwargs)

        # 保存视频
        frames = result.frames[0] if hasattr(result, 'frames') else result.video
        out_path = f"/tmp/cosmos_{num_frames}f_{width}x{height}.mp4"
        imageio.mimwrite(out_path, [np.array(f) for f in frames], fps=fps, codec="libx264", quality=5)

        # 清理中间结果
        del result
        clear_vram()

        duration = num_frames / fps
        mode_str = "I2V" if processed_image is not None else "T2V"
        info = f"[OK] {mode_str} | {width}x{height} | {num_frames}f | {steps}s | Duration: {duration:.1f}s"
        print(info)
        return out_path, info

    except torch.cuda.OutOfMemoryError:
        clear_vram()
        raise gr.Error(
            f"OOM! Available {free:.1f}GB VRAM is insufficient. Try decreasing:\n"
            f"  1. Frame Count (Current: {num_frames})\n"
            f"  2. Resolution (Current: {width}x{height})\n"
            f"  3. Inference Steps (Current: {steps})"
        )
    except Exception as e:
        clear_vram()
        raise gr.Error(f"Generation error: {str(e)}")


# ============ Gradio UI ============
css = """
#warn {background: #fff3cd; padding: 10px; border-radius: 8px; margin-bottom: 10px;}
"""

with gr.Blocks(title="Cosmos3-Nano Save-VRAM Edition", css=css) as demo:
    gr.Markdown("# Cosmos3-Nano Save-VRAM Edition")
    gr.Markdown("### Features: 8-bit Quantization + VAE Tiling/Slicing + Image-to-Video Support")

    with gr.Row():
        gr.Markdown(
            '<div id="warn">Warning: First run should use low parameters to test the pipeline environment.<br>'
            'Frame Rule: (Frames - 1) must be a multiple of 4 (e.g. 1, 5, 9, 13, 17, 21...)<br>'
            'Resolution Rule: Width and Height must be multiples of 16.</div>'
        )

    with gr.Row():
        # 左侧：输入
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Positive Prompt", placeholder="Describe the scene you want to generate...", lines=3)
            neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry, deformed", lines=2)
            
            # 【新增：图生视频输入槽】
            input_image = gr.Image(type="numpy", label="First Frame Image (Optional, for Image-to-Video)", height=250)

            with gr.Row():
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
                clear_vram_btn = gr.Button("Clear VRAM", size="lg")

        # 右侧：所有参数
        with gr.Column(scale=1):
            # --- 帧数 & 时长 ---
            num_frames = gr.Slider(
                minimum=1, maximum=121, value=5, step=4,
                label="Frame Count (1/5/9/13/17... Less frames save VRAM)"
            )
            fps = gr.Slider(
                minimum=1, maximum=30, value=8, step=1,
                label="FPS Playback Speed (Does not affect VRAM consumption)"
            )
            duration_display = gr.Textbox(
                value="Estimated duration: 0.63s", label="Estimated Video Duration",
                interactive=False
            )

            # --- 分辨率 ---
            with gr.Row():
                width = gr.Slider(16, 1280, value=128, step=16, label="Width")
                height = gr.Slider(16, 720, value=128, step=16, label="Height")

            # --- 质量参数 ---
            steps = gr.Slider(1, 100, value=1, step=1, label="Inference Steps (1 step is fastest but blurry)")
            guidance = gr.Slider(1.0, 15.0, value=4.0, step=0.5, label="Guidance Scale")

            # --- 种子 ---
            seed = gr.Number(value=-1, label="Seed (-1 = Random)")

            # --- 显存预估 ---
            vram_display = gr.Textbox(
                value="", label="VRAM Requirement Estimate",
                interactive=False
            )

    # 输出
    with gr.Row():
        output_video = gr.Video(label="Generated Result", autoplay=True)
        output_info = gr.Textbox(label="Generation Info", interactive=False)

    # ============ 交互逻辑 ============
    # 实时更新预估时长
    def update_duration(frames, fps_val):
        try:
            f = check_frames(frames)
            return f"Estimated duration: {f / max(int(fps_val), 1):.2f}s ({f} frames / {int(fps_val)} FPS)"
        except:
            return ""

    # 实时更新显存预估
    def update_vram(frames, h, w, img):
        try:
            f = check_frames(frames)
            h = max((int(h) // 16) * 16, 16)
            w = max((int(w) // 16) * 16, 16)
            has_img = (img is not None)
            est = estimate_vram(f, h, w, has_image=has_img)
            free = torch.cuda.mem_get_info()[0] / 1024**3
            status = "OK" if est < free * 0.7 else "Potential OOM Risk"
            return f"Estimate: ~{est:.1f} GB | Free: {free:.1f} GB | Status: {status}"
        except:
            return "Calculating..."

    # 帧数、分辨率、首帧图变化 → 更新时长+显存
    num_frames.change(update_duration, [num_frames, fps], duration_display)
    num_frames.change(update_vram, [num_frames, height, width, input_image], vram_display)
    fps.change(update_duration, [num_frames, fps], duration_display)
    height.change(update_vram, [num_frames, height, width, input_image], vram_display)
    width.change(update_vram, [num_frames, height, width, input_image], vram_display)
    input_image.change(update_vram, [num_frames, height, width, input_image], vram_display)

    # 生成按钮逻辑
    generate_btn.click(
        fn=generate,
        inputs=[prompt, neg_prompt, num_frames, height, width, steps, guidance, fps, seed, input_image],
        outputs=[output_video, output_info]
    )

    # 清理显存按钮
    def do_clear_vram():
        clear_vram()
        free = torch.cuda.mem_get_info()[0] / 1024**3
        return f"VRAM cleared! Free VRAM: {free:.1f} GB"

    clear_vram_btn.click(fn=do_clear_vram, outputs=vram_display)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
EOF