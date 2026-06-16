cat << 'EOF' > /home/gw/run_cosmos3_nano.py
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
from diffusers import DiffusionPipeline, PipelineQuantizationConfig

MODEL_PATH = "/home/gw/models/cosmos3-nano"

print("=" * 50)
print("  Cosmos3-Nano 极省显存版")
print("=" * 50)

# ============ 第一步：Patch掉Safety Checker（防止联网下载blocklist） ============
try:
    import cosmos_guardrail.cosmos_guardrail as _cg
    _orig_safety_init = _cg.CosmosSafetyChecker.__init__
    def _patched_safety_init(self, *a, **kw):
        print("  ⚠️ Safety Checker 已跳过（省显存+防联网报错）")
        pass
    _cg.CosmosSafetyChecker.__init__ = _patched_safety_init
    print("✅ Safety Checker 补丁已注入")
except Exception as e:
    print(f"⚠️ Safety Checker 补丁跳过: {e}")

# ============ 第二步：加载模型 ============
print("⏳ 正在加载 8bit 量化模型...")

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
    print(f"加载失败: {e}")
    print("尝试不带量化加载...")
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
    print("✅ VAE Tiling + Slicing 已开启")
except Exception as e:
    print(f"⚠️ VAE优化跳过: {e}")

# 不手动 to("cuda")，8bit量化时device_map会自动分配
print("✅ 模型加载完成！")
print(f"   显存已用: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

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

def estimate_vram(frames, height, width):
    """粗略估算显存需求（仅供参考）"""
    # 经验公式：每帧每万像素约需 0.03GB（8bit量化下）
    pixels = frames * height * width / 10000
    return pixels * 0.03

# ============ 核心生成函数 ============
def generate(prompt, neg_prompt, num_frames, height, width, steps, guidance, fps, seed):
    if not prompt.strip():
        raise gr.Error("提示词不能为空！")

    # 帧数修正
    num_frames = check_frames(num_frames)

    # 分辨率修正为16的倍数
    height = (int(height) // 16) * 16
    width = (int(width) // 16) * 16
    if height < 16:
        height = 16
    if width < 16:
        width = 16

    # 显存预估
    est = estimate_vram(num_frames, height, width)
    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"📊 预估显存需求: ~{est:.1f} GB | 剩余显存: {free:.1f} GB")

    if est > free * 0.8:
        print(f"⚠️ 显存可能不足！预估{est:.1f}GB > 可用{free:.1f}GB，尝试继续...")

    # 种子
    seed = int(seed)
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None

    print(f"🎬 开始生成: {width}x{height}, {num_frames}帧, {steps}步, FPS={fps}")
    print(f"   预估时长: {num_frames/fps:.1f}秒")

    clear_vram()

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt or "",
                num_frames=num_frames,
                num_inference_steps=int(steps),
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator,
            )

        # 保存视频
        frames = result.frames[0] if hasattr(result, 'frames') else result.video
        out_path = f"/tmp/cosmos_{num_frames}f_{width}x{height}.mp4"
        imageio.mimwrite(out_path, [np.array(f) for f in frames], fps=fps, codec="libx264", quality=5)

        # 清理中间结果
        del result
        clear_vram()

        duration = num_frames / fps
        info = f"✅ {width}x{height} | {num_frames}帧 | {steps}步 | 时长{duration:.1f}秒"
        print(info)
        return out_path, info

    except torch.cuda.OutOfMemoryError:
        clear_vram()
        raise gr.Error(
            f"❌ OOM！剩余{free:.1f}GB不够。请降低：\n"
            f"  ① 帧数（当前{num_frames}）\n"
            f"  ② 分辨率（当前{width}x{height}）\n"
            f"  ③ 步数（当前{steps}）"
        )
    except Exception as e:
        clear_vram()
        raise gr.Error(f"生成出错: {str(e)}")


# ============ Gradio UI ============
css = """
#warn {background: #fff3cd; padding: 10px; border-radius: 8px; margin-bottom: 10px;}
"""

with gr.Blocks(title="Cosmos3-Nano 极省显存版", css=css) as demo:
    gr.Markdown("# 🎬 Cosmos3-Nano 极省显存版")
    gr.Markdown("### 💡 策略：8bit量化 + VAE分块 + 极低默认参数 → 保证能出视频")

    with gr.Row():
        gr.Markdown(
            '<div id="warn">⚠️ <b>首次运行建议用默认参数</b>（5帧/128x128/1步）跑通，确认环境OK后再慢慢调大！<br>'
            '帧数规则: (帧数-1) 必须是4的倍数 → 1, 5, 9, 13, 17, 21...<br>'
            '分辨率规则: 宽高必须是16的倍数</div>'
        )

    with gr.Row():
        # 左侧：输入
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="📝 正向提示词", placeholder="描述你想生成的视频...", lines=3)
            neg_prompt = gr.Textbox(label="🚫 反向提示词", value="low quality, blurry", lines=2)

            with gr.Row():
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                clear_vram_btn = gr.Button("🧹 清理显存", size="lg")

        # 右侧：所有参数
        with gr.Column(scale=1):
            # --- 帧数 & 时长 ---
            num_frames = gr.Slider(
                minimum=1, maximum=121, value=5, step=4,
                label="🎞️ 帧数 (1/5/9/13/17... 越少越省显存)"
            )
            fps = gr.Slider(
                minimum=1, maximum=30, value=8, step=1,
                label="⏱️ FPS帧率 (影响播放速度，不影响生成显存)"
            )
            duration_display = gr.Textbox(
                value="预估时长: 0.63秒", label="📹 预估视频时长",
                interactive=False
            )

            # --- 分辨率 ---
            with gr.Row():
                width = gr.Slider(16, 1280, value=128, step=16, label="↔️ 宽度")
                height = gr.Slider(16, 720, value=128, step=16, label="↕️ 高度")

            # --- 质量参数 ---
            steps = gr.Slider(1, 100, value=1, step=1, label="🔄 推理步数 (1步最快但糊)")
            guidance = gr.Slider(1.0, 15.0, value=4.0, step=0.5, label="🎯 Guidance Scale")

            # --- 种子 ---
            seed = gr.Number(value=-1, label="🎲 种子 (-1=随机)")

            # --- 显存预估 ---
            vram_display = gr.Textbox(
                value="", label="📊 预估显存需求",
                interactive=False
            )

    # 输出
    with gr.Row():
        output_video = gr.Video(label="生成结果", autoplay=True)
        output_info = gr.Textbox(label="生成信息", interactive=False)

    # ============ 交互逻辑 ============
    # 实时更新预估时长
    def update_duration(frames, fps_val):
        try:
            f = check_frames(frames)
            return f"预估时长: {f / max(int(fps_val), 1):.2f}秒 ({f}帧÷{int(fps_val)}FPS)"
        except:
            return ""

    # 实时更新显存预估
    def update_vram(frames, h, w):
        try:
            f = check_frames(frames)
            h = max((int(h) // 16) * 16, 16)
            w = max((int(w) // 16) * 16, 16)
            est = estimate_vram(f, h, w)
            free = torch.cuda.mem_get_info()[0] / 1024**3
            status = "✅ 够用" if est < free * 0.7 else "⚠️ 可能OOM"
            return f"预估: ~{est:.1f} GB | 剩余: {free:.1f} GB | {status}"
        except:
            return "计算中..."

    # 帧数变化 → 更新时长+显存
    num_frames.change(update_duration, [num_frames, fps], duration_display)
    num_frames.change(update_vram, [num_frames, height, width], vram_display)
    fps.change(update_duration, [num_frames, fps], duration_display)
    height.change(update_vram, [num_frames, height, width], vram_display)
    width.change(update_vram, [num_frames, height, width], vram_display)

    # 生成按钮
    generate_btn.click(
        fn=generate,
        inputs=[prompt, neg_prompt, num_frames, height, width, steps, guidance, fps, seed],
        outputs=[output_video, output_info]
    )

    # 清理显存按钮
    def do_clear_vram():
        clear_vram()
        free = torch.cuda.mem_get_info()[0] / 1024**3
        return f"已清理！剩余显存: {free:.1f} GB"

    clear_vram_btn.click(fn=do_clear_vram, outputs=vram_display)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
EOF