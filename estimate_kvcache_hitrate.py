"""
估算 ShareGPT 数据集在 vLLM serve 上的 KV Cache 命中率
硬件假设：2x 华为昇腾 910B2（单卡 64GB HBM，共 128GB）
依赖：无第三方库，仅使用 Python 标准库

用法：
    python estimate_kvcache_hitrate.py [--dataset PATH]
"""

import json
import math
import argparse
from pathlib import Path
from collections import Counter


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def est_tokens(text: str) -> int:
    """粗略估算 token 数（英文约 1 token = 4 字符）"""
    return max(1, len(text) // 4)


# ── 数据集分析 ────────────────────────────────────────────────────────────────

def analyze_dataset(data: list) -> dict:
    """分析数据集基本统计信息"""
    total_tokens_per_conv = []
    human_turns_per_conv = []
    mid_ctx_per_conv = []      # 非首轮请求的平均输入长度（可命中 cache 的部分）

    for d in data:
        convs = d["conversations"]
        cumulative = 0
        human_turns = []

        for c in convs:
            t = est_tokens(c["value"])
            cumulative += t
            if c["from"] == "human":
                human_turns.append(cumulative)

        total_tokens_per_conv.append(cumulative)
        human_turns_per_conv.append(len(human_turns))

        # 非首轮的平均输入 token 数（首轮无 cache 可用）
        if len(human_turns) > 1:
            mid_ctx_per_conv.append(sum(human_turns[1:]) / (len(human_turns) - 1))

    n = len(data)
    tokens_sorted = sorted(total_tokens_per_conv)

    return {
        "total_convs": n,
        "avg_tokens": sum(total_tokens_per_conv) / n,
        "median_tokens": tokens_sorted[n // 2],
        "p90_tokens": tokens_sorted[int(n * 0.9)],
        "max_tokens": tokens_sorted[-1],
        "avg_human_turns": sum(human_turns_per_conv) / n,
        "avg_cacheable_tokens": sum(mid_ctx_per_conv) / len(mid_ctx_per_conv) if mid_ctx_per_conv else 0,
    }


# ── 理论命中率 ────────────────────────────────────────────────────────────────

def theoretical_hit_rate(avg_turns: float) -> tuple[float, float]:
    """
    计算对话内前缀缓存的两种理论命中率。

    vLLM 上报的命中率是 token 加权指标：
        缓存命中的 token 数 / 总输入 token 数

    对于 N 轮对话，设每条消息长度近似相等：
    - 第 i 轮输入规模 ∝ i（累积 context 线性增长）
    - 第 i 轮可命中 token = 第 i 轮输入 − 本轮新 human 消息

    Token 加权命中率：
        Σ(cumulative_i - h_i) / Σ(cumulative_i) ≈ (N-1)/N

    每请求等权均值（调和级数，仅供参考）：
        (1/N) * Σ(i-1)/i = 1 - H(N)/N
    """
    N = int(avg_turns)
    token_weighted = (avg_turns - 1) / avg_turns          # vLLM 实际上报的口径
    harmonic_N = sum(1 / i for i in range(1, N + 1))
    per_request_avg = (N - harmonic_N) / N                # 每请求等权均值（偏低）
    return token_weighted, per_request_avg


# ── 模型配置 ──────────────────────────────────────────────────────────────────

MODELS = [
    # (display_name,  model_size_gb, num_layers, num_kv_heads, head_dim, kv_dtype_bytes)
    # 架构参数来源：各模型 HuggingFace 官方仓库 config.json
    # Qwen3-8B 架构参数来源：
    #   https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json
    #   num_hidden_layers=36, num_key_value_heads=8, head_dim=128
    #   参数量约 8.2B → fp16 约 16GB
    ("Qwen3-8B    (fp16)",        16,  36,  8, 128, 2),
    # Qwen3-32B 架构参数来源：
    #   https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json
    #   num_hidden_layers=64, num_key_value_heads=8, head_dim=128
    #   参数量约 32.8B → fp16 约 66GB，INT4 约 17GB
    ("Qwen3-32B   (fp16)",        66,  64,  8, 128, 2),
    ("LLaMA3-70B  (INT4+fp16kv)", 35,  80,  8, 128, 2),
]

TOTAL_HBM_GB = 128          # 2x 64GB
HBM_UTILIZATION = 0.85      # 预留 15% 给激活值、框架开销等
CONCURRENCIES = [1, 8, 32, 64, 128, 256]


# ── 主分析逻辑 ────────────────────────────────────────────────────────────────

def analyze_kvcache(stats: dict) -> None:
    avg_cacheable_tokens = stats["avg_cacheable_tokens"]
    avg_turns = stats["avg_human_turns"]
    base_hit_rate, per_request_hit_rate = theoretical_hit_rate(avg_turns)

    print("=" * 70)
    print("KV Cache 命中率估算")
    print(f"硬件：2x 昇腾 910B2，共 {TOTAL_HBM_GB}GB HBM")
    print("=" * 70)

    print("\n【数据集统计】")
    print(f"  总对话数          : {stats['total_convs']}")
    print(f"  平均 token / 对话 : {stats['avg_tokens']:.0f}")
    print(f"  中位数 token      : {stats['median_tokens']}")
    print(f"  P90 token         : {stats['p90_tokens']}")
    print(f"  最长 context      : {stats['max_tokens']} tokens")
    print(f"  平均 human 轮数   : {avg_turns:.1f}")
    print(f"  平均可命中 token  : {avg_cacheable_tokens:.0f}（非首轮平均输入长度）")

    N = int(avg_turns)
    print(f"\n【理论最大命中率（对话内 prefix 复用）】")
    print(f"  Token 加权命中率（vLLM 实际上报口径）：(N-1)/N = {N-1}/{N} ≈ {base_hit_rate:.1%}")
    print(f"  每请求等权均值  （调和级数，仅供参考）：1-H({N})/{N}  ≈ {per_request_hit_rate:.1%}")
    print(f"  差异原因：越晚的 turn 携带更大的累积 context，token 加权后高权重 turn 拉高整体命中率")
    print(f"  （首轮请求无 cache，后续轮次输入 = 全量历史，仅新 human 消息部分需重新计算）\n")

    print(f"{'模型':<28} {'模型占用':>6} {'KV预算':>6} {'单对话KV':>8} "
          f"{'最大并发缓存':>10}  " + "  ".join(f"并发={c:3d}" for c in CONCURRENCIES))
    print("-" * 120)

    for (name, model_gb, layers, kv_heads, head_dim, kv_dtype) in MODELS:
        kv_budget_gb = TOTAL_HBM_GB * HBM_UTILIZATION - model_gb
        if kv_budget_gb <= 0:
            print(f"{name:<28}  !! 显存不足，无法加载 !!")
            continue

        # 单 token 占用 KV 字节数 = 2(K+V) × kv_heads × head_dim × layers × dtype
        kv_bytes_per_token = 2 * kv_heads * head_dim * layers * kv_dtype
        per_conv_kv_mb = kv_bytes_per_token * avg_cacheable_tokens / 1e6
        max_convs_cached = int(kv_budget_gb * 1e9 / (kv_bytes_per_token * avg_cacheable_tokens))

        row = (f"{name:<28} {model_gb:>5}GB {kv_budget_gb:>5.0f}GB "
               f"{per_conv_kv_mb:>7.1f}MB {max_convs_cached:>10}  ")

        for c in CONCURRENCIES:
            if c <= max_convs_cached:
                rate = base_hit_rate
                flag = "   "
            else:
                # cache 不足：命中率按可覆盖比例线性衰减
                rate = (max_convs_cached / c) * base_hit_rate
                flag = " ⚠ "
            row += f"  {rate:>5.1%}{flag}"
        print(row)

    print()
    print("说明：")
    print("  ⚠  表示该并发数超出 KV cache 容量，命中率因 eviction 下降")
    print("  命中率基于对话内前缀复用（intra-conversation prefix caching）")
    print("  跨对话（inter-conversation）命中率忽略不计（内容差异大）")
    print("  需在 vllm serve 中启用：--enable-prefix-caching")
    print("  910B2 需使用 vllm-ascend 后端，请确认该版本已支持 APC")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="估算 vLLM KV Cache 命中率")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent.parent / "datasets" / "ShareGPT_multiturn_long_200_30-60.json"),
        help="数据集 JSON 文件路径"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"错误：找不到数据集文件 {dataset_path}")
        return

    print(f"加载数据集：{dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = analyze_dataset(data)
    analyze_kvcache(stats)


if __name__ == "__main__":
    main()
