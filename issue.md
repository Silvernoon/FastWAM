## 1

失败时会如同失去视觉一样在空处乱抓

或者在已经抓过的位置反复抓

残留？视觉没刷新？

过去的视觉影响过大

**视觉 conditioning 失效或过度锚定**

## 输入图像有刷新吗？

**有，但只在每次 replan 时刷新。** 默认 `replan_steps=10`，即每 10 步动作才重新拍一帧、重新 VAE encode、重新 prefill video cache。在这 10 步之间，action denoising 一直 attend 到**同一个 video KV cache**（来自 replan 开头那一帧）。

---

## 你描述的两种失败模式的根因

从 MoT 的 `_forward_action_with_video_cache_inner` (mot.py line 399) 可以看到：

```python
k_cat = torch.cat([k_video, k_action], dim=1)
v_cat = torch.cat([v_video, v_action], dim=1)
mixed = self._mixed_attention(q_cat=q_action, k_cat=k_cat, v_cat=v_cat, ...)
```

**每一层** action tokens 都 attend 到 video KV cache。而 video KV cache 是 prefill 时 video expert 对**单帧 latent** 做完整 transformer forward 后缓存的 K/V。

问题在于：

### 1. "在空处乱抓" — 视觉 conditioning 失效

当 `first_frame_latents` 的 timestep 被设为 0（line 1128）：
```python
timestep_video = torch.zeros((first_frame_latents.shape[0],), ...)
```
这意味着 video expert 把输入当作"完全干净的 latent"（t=0 = no noise）。但 `pre_dit` 里的 `token_timesteps[:, 0, :] = 0` 只对第一帧设 t=0，其余帧用传入的 timestep。推理时只有一帧，所以整个 video 都是 t=0。

如果模型训练时 video expert 主要学的是"从噪声预测视频"的 denoising 任务，那 t=0 的 prefill 可能产出的 KV 表征**不够有信息量**——模型没见过这种"完全干净输入"的 pattern，导致 action expert 实际上没有有效的视觉 grounding。

### 2. "在已经抓过的位置反复抓" — 视觉锚定过强

这是相反的情况：video KV cache 太强地编码了"物体在某个位置"的信息，即使物体已经被移走了（因为 10 步内不刷新图像），action expert 仍然被旧的视觉 cache 引导去那个位置。

---

## 修复方案

### 方案 A：减小 replan 间隔（最简单，立即见效）

```yaml
EVALUATION:
  replan_steps: 3  # 从 10 降到 3，更频繁刷新视觉
```

这直接减少 stale observation 的影响。代价是推理次数增加（每次 replan 都要跑一次 VAE encode + prefill + action denoising loop）。

### 方案 B：缩短 action_horizon 配合 replan

如果 `action_horizon` 很长但 `replan_steps` 很短，模型预测了很多步但只用前几步。可以让 `action_horizon = replan_steps`，减少浪费并让模型专注于短期预测。

### 方案 C：在 attention mask 中降低 video cache 的权重（需要改代码）

在 `_forward_action_with_video_cache_inner` 中，给 video KV 加一个 scaling factor，降低其对 action 的影响：

```python
# mot.py, 在 mixed attention 之前
v_video = video_cache_v[layer_idx] * self.video_attn_scale  # e.g. 0.5
```

或者用 attention bias 的方式：在 attention score 上对 video 部分加一个负偏置。

### 方案 D：训练时加入 observation noise / augmentation

如果是训练阶段能改的话，对输入图像加随机 augmentation（crop jitter、color jitter、dropout 某些 patch），让模型不要过度依赖单帧的精确像素信息。

---

你想我实现哪个方案？如果是快速验证，方案 A 改一行 config 就行。如果想从根本上解决，方案 C 需要改 MoT 代码。

---
Edited `src/fastwam/datasets/lerobot/transforms/image.py`
Add TemporalConsistentColorJitter and TemporalConsistentGaussianNoise augmentation classes to prevent the model from over-relying on exact pixel values during training.



Now update the LIBERO data config to use these augmentations during training:

Edited `configs/data/libero_2cam.yaml`
Add color jitter and Gaussian noise augmentation to training transforms to reduce over-reliance on exact pixel values.



### C：推理时支持多帧历史输入

现在修改 `infer_action` 让它可以接受最近 N 帧而不是单帧：

Read file `src/fastwam/models/wan22/fastwam.py` (1028 - 1100)



Now let me look at `_encode_input_image_latents_tensor` more carefully and the `_build_mot_attention_mask` to understand how multi-frame would work:

Read file `src/fastwam/models/wan22/fastwam.py` (255 - 270)



Now let me look at how the attention mask for action→video is built during `infer_action` to understand what changes with multi-frame:

Read file `src/fastwam/models/wan22/fastwam.py` (395 - 430)



I see the key constraint: `mask[video_seq_len:, :first_frame_tokens] = True` — action only attends to the **first frame** of video tokens. This means even if we encode multiple frames, the action expert would only see the first frame's tokens unless we change this mask.

Let me now implement the multi-frame approach. The cleanest way is to add a method that encodes multiple frames and modify the attention mask to let action attend to all provided frames:

Read file `src/fastwam/models/wan22/fastwam.py` (243 - 256)



Edited `src/fastwam/models/wan22/fastwam.py`
Add _encode_multi_frame_latents_tensor method to support encoding multiple observation frames for multi-frame history conditioning.



### B：在 MoT attention 中加 video attention scaling

Now let me modify the `_build_mot_attention_mask` and add a configurable `video_attn_scale` parameter:

Read file `src/fastwam/models/wan22/fastwam.py` (20 - 50)



Edited `src/fastwam/models/wan22/fastwam.py`
Add video_attn_scale parameter to FastWAM __init__ to control how strongly video KV cache influences action denoising.



Now I need to store this parameter and use it. Let me find where `self.loss_lambda_action` is stored:

Read file `src/fastwam/models/wan22/fastwam.py` (82 - 92)



Edited `src/fastwam/models/wan22/fastwam.py`
Store video_attn_scale as instance attribute.


Actually, the better approach is to apply the scale at the `_denoise_step_compiled` level in `fastwam.py`, since that's where `video_attn_scale` lives. Let me modify it to scale the video V cache before passing to MoT:


The cleanest approach for `video_attn_scale` is to scale the video V cache values. This effectively controls how much information flows from video to action. Let me apply it before passing to MoT:

Edited `src/fastwam/models/wan22/fastwam.py`
Apply video_attn_scale to video V cache before passing to MoT, allowing control over how strongly video conditioning influences action denoising.




### C：修改 `infer_action` 支持多帧输入 + 修改 attention mask

Now let me add the `infer_action_multiframe` method that uses multiple observation frames:

Edited `src/fastwam/models/wan22/fastwam.py`


Now let me add the `infer_action_multiframe` method right after `infer_action`:

Edited `src/fastwam/models/wan22/fastwam.py`



Now I'll modify `_predict_action_chunk` to support multi-frame history with a frame buffer:

Edited `experiments/libero/eval_libero_single.py`

Now I need to add the frame buffer initialization and pass it to `_predict_action_chunk`:

Edited `experiments/libero/eval_libero_single.py`
Add obs_frame_buffer initialization for multi-frame observation history in run_single_episode.



Edited `experiments/libero/eval_libero_single.py`
Pass obs_frame_buffer to _predict_action_chunk for multi-frame history support.



Now let me add the `num_obs_history` and `video_attn_scale` config options to the LIBERO eval config:

Edited `configs/sim_libero.yaml`
Add num_obs_history and video_attn_scale documentation to the LIBERO eval config.




##

世界模型的准度
- VAE?
- VideoDiT?

actionDiT的准度

挽回？

速度？
