## 视觉输入的处理方式

是的，**用 VAE 编码**。具体流程：

1. **eval 端预处理**（在 `_obs_to_model_input` 中）：
   - 从环境拿到 RGB 图 → 旋转 180° → center-crop + resize → 多相机拼接 → 归一化到 `[-1, 1]`，得到 `[1, 3, H, W]` 的 tensor。

2. **模型内部**（在 `infer_action` 中）：
   - 调用 `self._encode_input_image_latents_tensor(input_image)` — 底层走的是 `self.vae.encode()`，把单帧 RGB 编码成 latent：
     ```python
     image = input_image[0].unsqueeze(1)  # [3, 1, H, W]
     z = self.vae.encode([image], ...)     # VAE latent
     ```
   - 得到的 `first_frame_latents` 作为 video expert 的条件输入（`pre_dit` 的 `x` 参数），video expert 用它做 KV prefill，生成 attention cache 供 action expert 交叉注意力使用。

所以视觉条件的本质是：**单帧 RGB → Wan2.2 VAE 编码为 latent → video DiT 做一次 prefill 得到 KV cache → action DiT 在去噪循环中通过 MoT 混合注意力看到这个视觉 cache**。

---

## 文本 prompt 的处理方式

用的是 **Wan2.2 自带的 text encoder + tokenizer**：

```python
def encode_prompt(self, prompt):
    ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    prompt_emb = self.text_encoder(ids, mask)  # [B, L, D]
    # 超过实际 token 长度的位置 zero-pad
    seq_lens = mask.gt(0).sum(dim=1).long()
    for i, v in enumerate(seq_lens):
        prompt_emb[i, v:] = 0
    return prompt_emb, mask  # context, context_mask
```

得到的 `context` 和 `context_mask` 会：
1. 如果有 proprio（本体感知），通过 `proprio_encoder`（一个 Linear 层）编码后 **concat 到 context 序列末尾** 作为额外一个 token。
2. 最终传给 video expert 的 `pre_dit` 和 action expert 的去噪步骤，作为 cross-attention 的条件。

**在 libero eval 中额外做了 prompt caching**：每个 task 只 encode 一次 prompt，所有 trial 和 replan 步都复用同一个 `(cached_context, cached_context_mask)`，避免重复计算。

---

总结一句话：**视觉走 VAE latent → video DiT prefill 提供 KV cache；文本走 text encoder 得到 embedding 序列；两者在 MoT 架构中通过注意力机制共同条件化 action DiT 的去噪过程。**

##

**LIBERO 视觉输入的完整路径：**

1. **环境采集观测** — `experiments/libero/eval_libero_single.py` 中调用 `get_libero_image(obs)` 获取 RGB numpy 图像（主摄像头 + wrist 摄像头）。

2. **图像预处理** — `_obs_to_model_input()` (line ~233) 将多摄像头图像做 center-crop-resize 然后拼接（水平/垂直），归一化到 `[-1, 1]`：
   ```python
   x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
   x = x * (2.0 / 255.0) - 1.0
   ```
   产出 shape 为 `[1, 3, H, W]` 的 tensor。

3. **调用模型推理** — 传入 `model.infer_action(input_image=image, ...)` (在 `_predict_action_chunk()` 里, line ~480)。

4. **VAE 编码图像** — `fastwam.py` 的 `infer_action()` (line 1031) 内部：
   ```python
   first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
   ```
   该方法 (line 256) 把 `[1, 3, H, W]` 转成 `[3, 1, H, W]`（单帧视频），调用 `self.vae.encode([image], device=self.device)` 得到 latent `z`。

5. **VAE 内部** — `WanVideoVAE.encode()` → `VideoVAE_.encode()` → `Encoder3d` 逐帧 causal 编码，输出 latent shape `[B, z_dim, T, H/8, W/8]`。

6. **Latent 作为 video token 进入 MoT** — `first_frame_latents` 通过 `video_expert.pre_dit()` 转为 video tokens，再经过 MoT 的 prefill 缓存住，与 action denoising 交互。

---

**关键文件和位置：**

| 步骤 | 文件 | 行号 |
|------|------|------|
| 环境图像获取 & 预处理 | `experiments/libero/eval_libero_single.py` | ~233 `_obs_to_model_input()` |
| VAE 编码入口 | `src/fastwam/models/wan22/fastwam.py` | ~256 `_encode_input_image_latents_tensor()` |
| VAE 编码实现 | `src/fastwam/models/wan22/wan_video_vae.py` | ~519 `Encoder3d` / ~618 `Encoder3d_38` |
| 顶层 VAE 封装 | `src/fastwam/models/wan22/wan_video_vae.py` | ~1060 `WanVideoVAE` / `WanVideoVAE38` |

核心流程就是：LIBERO 环境 RGB 图片 → 归一化到 `[-1,1]` → `VAE.encode()` → 得到 latent → 作为 video expert 的 first-frame conditioning 进入 MoT。

