pi0.5在经过微调后对长时任务成功率明显提升

* 常规FastWam，一次Forward Pass后共享KV Cache

* Joint 即做联合去噪

* fast-wam-idm 即先生成future video，后action再基于生成出的视频表征来预测

* Fast-WAM w.o. video co-train 去除视频训练

## ation-dim

经过最后action-DiT去噪得到的**机器人低维控制信号**

通常是关节角度/速度、末端执行器站姿、夹爪开合等拼接在一起的向量。

从 `action_state_merger.py` 可以看到，它把多个控制维度（如各关节 + gripper）concat 成一个统一的 `[T, action_dim]` 张量，还支持 padding 到固定维度。

所以输出形状是 `[action_horizon, action_dim]`——一段时间窗口内每一步的机器人控制指令。

## 和 Video DiT 的关系

这不是简单地"把 Video DiT 的输出换成动作"，而是一个 **Mixture-of-Transformers (MoT)** 架构：

1. **Video Expert** — 原始 Wan2.2 视频 DiT，处理视频 latent tokens
2. **Action Expert** — 这个 `ActionDiT`，处理动作 tokens
3. **MoT** — 把两个 expert 的 token 拼在一起做 **混合 self-attention**，然后各自分开做 cross-attention 和 FFN

具体来说，每一层：
- 各 expert 独立计算 Q/K/V
- Q/K/V 拼接后做一次联合 attention（video tokens 和 action tokens 互相看到对方）
- attention 输出按 expert 切回去，各自走自己的 cross-attn + FFN

这样 action expert 的骨干权重可以直接从视频 DiT 初始化（`from_pretrained` 里加载 `backbone_state_dict`，跳过 `action_encoder` 和 `head`），然后通过混合 attention 让动作生成和视频理解共享表征。

## 推理流程 (`infer_action`)

1. 输入一张图片 → VAE 编码为 first-frame latent
2. Video expert 做一次前向（timestep=0，即无噪声），prefill video KV cache
3. 采样纯噪声作为初始 action latent `[1, action_horizon, action_dim]`
4. 循环 N 步去噪：
   - Action expert 计算 Q/K/V
   - 与 cached video K/V 做混合 attention（action 能看到 video，但不重新计算 video）
   - 预测噪声 → flow matching scheduler 更新 action latent
5. 最终输出去噪后的 `[action_horizon, action_dim]` 作为机器人控制序列

本质上就是：**用视频模型的视觉理解能力（通过混合 attention）来引导动作扩散生成**，video expert 提供场景理解的 context，action expert 负责在这个 context 下生成合理的控制信号。

## 

## 调整 attention 的具体方案

**方案 A：Action-only self-attention（去掉 video KV 参与）**

最简单的验证方式——把 action → video 的 attention 关掉，看性能掉多少：

```python
# _build_mot_attention_mask 中
# 注释掉这行：
# mask[video_seq_len:, :first_frame_tokens] = True
```

如果性能几乎不掉，说明 ActionDiT 确实自己就够了。

**方案 B：把 first-frame 信息从 mixed attention 移到 cross-attention**

不让 action tokens 在 self-attention 里看 video KV，而是把 first-frame latent 编码后拼到 action 的 cross-attention context 里（和 text embedding 一起）。这样 ActionDiT 仍然能获取视觉条件，但不依赖 video expert 的前向计算：

```python
# 在 action_expert.pre_dit 中，把 first_frame_feature 拼到 context
action_context = torch.cat([text_context, first_frame_feature], dim=1)
```

**方案 C：Per-layer learnable gate（渐进式）**

在 `forward_action_with_video_cache` 中，给每层加一个 scalar gate（初始化为 0 或小值），控制 video KV 的混入程度：

```python
# MoT.__init__ 中
self.video_gate = nn.Parameter(torch.zeros(self.num_layers))

# forward_action_with_video_cache 中
gate = torch.sigmoid(self.video_gate[layer_idx])
k_cat = torch.cat([gate * k_video, k_action], dim=1)
v_cat = torch.cat([gate * v_video, v_action], dim=1)
```

训练后检查 gate 值——如果大部分层 gate → 0，就验证了你的假设。

## 思路

AtionDiT实际上已经学会了世界变化吗，因为它是从VideoDiT改的

那为什么还要从VideoDiT中再生成然后传递呢

从这个项目数据可以明显得到VideoDiT不做生成性能也很不错，是否是因为ActionDiT自己就能理解未来

那为什么数据差一点

是不是attention的问题，videoDiT提交一次影响了attention


