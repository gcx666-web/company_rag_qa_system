# -*- coding: utf-8 -*-
"""OpenAIEmbeddings 的真实调用集成测试（OpenAI 兼容接口，供应商可切换）。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    # 1. 在项目根目录 .env 中配置（.env 已被 .gitignore 忽略），默认硅基流动：
    EMBEDDING_API_KEY=sk-xxxx
    EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
    EMBEDDING_MODEL=BAAI/bge-m3

    # 2. 直接运行本文件
    .venv/Scripts/python.exe tests/test_openai_embedding.py

切换供应商只需改 .env，例如阿里云百炼：
    EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    EMBEDDING_MODEL=text-embedding-v4

如果只配置了某一家的 key（未设置 EMBEDDING_BASE_URL/MODEL），会自动匹配默认配置：
    SILICONFLOW_API_KEY -> 硅基流动 + BAAI/bge-m3
    DASHSCOPE_API_KEY  -> 阿里云百炼 + text-embedding-v4

说明：
1. 本文件会真实请求配置的 embedding 服务，未配置 key 时打印 [SKIP] 并以退出码 1 结束。
2. 默认配置使用硅基流动的 BAAI/bge-m3（免费）；一次运行只消耗几十个 Token。
3. 走 OpenAI 兼容接口，因此 DashScope、硅基流动等都能直接复用。
4. 测试3（dimensions 参数）只对支持的模型执行：
   text-embedding-v3/v4、text-embedding-3-small/large、Qwen/Qwen3-Embedding 系列；
   其他模型（如 BAAI/bge-m3）会打印 [SKIP] 并跳过。
5. 本文件是脚本式集成测试，顶层函数不会被 `python -m unittest` 收集，
   请直接运行本文件（与参考写法一致）。
"""

import math
import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path，保证从任意目录运行都能导入 src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from tenacity import RetryError

from src.embeddings.openai_enbeddings import OpenAIEmbeddings


# 各供应商的默认配置
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "BAAI/bge-m3"       # 免费，1024 维，不支持 dimensions
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "text-embedding-v4"   # 默认 1024 维，支持 dimensions
# 用于验证 dimensions 参数生效的维度；需为所选模型支持的取值
CUSTOM_DIMENSION = 512


def resolve_config():
    """按 .env / 环境变量解析出 (api_key, base_url, model, key_source)。

    优先级：
    1. EMBEDDING_API_KEY -> 默认硅基流动配置（可用 EMBEDDING_BASE_URL/MODEL 覆盖）
    2. SILICONFLOW_API_KEY -> 默认硅基流动配置
    3. DASHSCOPE_API_KEY -> 默认阿里云百炼配置
    这样只配置某一家的 key 时，不会把 key 误发到另一家的接口上。
    """
    load_dotenv(str(PROJECT_ROOT / ".env"))

    explicit_key = os.getenv("EMBEDDING_API_KEY")
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")

    if explicit_key:
        api_key, key_source = explicit_key, "EMBEDDING_API_KEY"
        default_url, default_model = SILICONFLOW_BASE_URL, SILICONFLOW_MODEL
    elif siliconflow_key:
        api_key, key_source = siliconflow_key, "SILICONFLOW_API_KEY"
        default_url, default_model = SILICONFLOW_BASE_URL, SILICONFLOW_MODEL
    elif dashscope_key:
        api_key, key_source = dashscope_key, "DASHSCOPE_API_KEY"
        default_url, default_model = DASHSCOPE_BASE_URL, DASHSCOPE_MODEL
    else:
        return None

    base_url = os.getenv("EMBEDDING_BASE_URL", default_url)
    model = os.getenv("EMBEDDING_MODEL", default_model)
    return api_key, base_url, model, key_source


def supports_dimensions(model: str) -> bool:
    """判断模型是否支持 OpenAI 兼容接口的 dimensions 参数。

    - 阿里云百炼：text-embedding-v3 / v4
    - OpenAI：text-embedding-3-small / 3-large
    - 硅基流动：Qwen/Qwen3-Embedding 系列
    BAAI/bge-* 等模型不支持该参数。
    """
    if model in {
        "text-embedding-v3",
        "text-embedding-v4",
        "text-embedding-3-small",
        "text-embedding-3-large",
    }:
        return True
    return "Qwen3-Embedding" in model


def print_vector_stats(vector, label=""):
    """打印向量的维度与数值分布，便于人工确认返回结果是否正常。

    :param vector: 单个文本的向量（list[float]）
    :param label: 输出前缀，用于区分多条文本
    """
    prefix = f"  {label} " if label else "  "
    if not vector:
        print(f"{prefix}向量为空")
        return

    values = [float(x) for x in vector]
    dimension = len(values)
    norm = math.sqrt(sum(x * x for x in values))
    mean = sum(values) / dimension
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / dimension)

    print(f"{prefix}维度: {dimension}")
    print(f"{prefix}前10个值: {[round(x, 6) for x in values[:10]]}")
    print(f"{prefix}L2范数: {norm:.6f}")
    print(f"{prefix}最小值/最大值: {min(values):.6f} / {max(values):.6f}")
    print(f"{prefix}均值/标准差: {mean:.6f} / {std:.6f}")
    finite = all(math.isfinite(x) for x in values)
    print(f"{prefix}NaN/Inf: {'无' if finite else '存在'}")


def cosine_similarity(left, right):
    """计算两个向量的余弦相似度（纯标准库实现，避免引入额外依赖）。"""
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def test_embedding() -> bool:
    """真实调用配置的 embedding 服务，返回是否全部通过。"""
    config = resolve_config()
    if config is None:
        print("[SKIP] 未设置 EMBEDDING_API_KEY 环境变量")
        print("也接受 SILICONFLOW_API_KEY / DASHSCOPE_API_KEY")
        print("请在 .env 文件中添加：EMBEDDING_API_KEY=your-key")
        return False

    api_key, base_url, model, key_source = config

    try:
        # 初始化
        print("\n初始化模型...")
        print(f"  Key 来源: {key_source}（不显示具体值）")
        if key_source == "DASHSCOPE_API_KEY" and "dashscope" not in base_url:
            print("[WARN] 当前使用 DASHSCOPE_API_KEY，但 base_url 不是百炼地址")
        if key_source == "SILICONFLOW_API_KEY" and "siliconflow" not in base_url:
            print("[WARN] 当前使用 SILICONFLOW_API_KEY，但 base_url 不是硅基流动地址")
        embedder = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        print(f"  模型: {embedder.get_model_name()}")
        print(f"  向量维度: {embedder.get_dimension()}")
        print(f"  接口地址: {embedder.base_url}")

        # 测试1：单条文本向量化
        print("\n--- 测试1：单条文本向量化 ---")
        text = "人工智能是计算机科学的一个分支"
        vector = embedder.embed_text(text)
        print(f"  输入文本: {text}")
        print(f"  向量前10个值: {[round(x, 6) for x in vector[:10]]}")
        print_vector_stats(vector)

        assert vector, "单条文本返回了空向量"
        assert len(vector) == embedder.get_dimension(), (
            f"实际维度 {len(vector)} 与 get_dimension() "
            f"{embedder.get_dimension()} 不一致"
        )
        assert all(math.isfinite(x) for x in vector), "向量中存在 NaN/Inf"
        print("  [OK] 维度与数值检查通过")

        # 测试2：批量文本向量化（text-embedding-v4 单次最多 10 条）
        print("\n--- 测试2：批量文本向量化 ---")
        texts = [
            "员工如何申请年假",
            "请假需要走什么流程",
            "苹果手机是电子产品",
        ]
        vectors = embedder.embed_batch(texts)
        assert len(vectors) == len(texts), (
            f"批量返回 {len(vectors)} 条向量，期望 {len(texts)} 条"
        )
        dimensions = {len(item) for item in vectors}
        assert dimensions == {embedder.get_dimension()}, (
            f"批量向量维度不一致：{dimensions}"
        )
        print(f"  共 {len(vectors)} 条，每条维度 {embedder.get_dimension()}")
        for index, (item, item_vector) in enumerate(zip(texts, vectors)):
            print(f"\n  [{index}] 输入文本: {item}")
            print_vector_stats(item_vector)
        print("  [OK] 批量条数与维度检查通过")

        # 语义检查：同义句相似度应高于无关句（只比较相对高低，不写死阈值）
        print("\n--- 语义相似度检查 ---")
        sim_related = cosine_similarity(vectors[0], vectors[1])
        sim_unrelated = cosine_similarity(vectors[0], vectors[2])
        print(f"  {texts[0]} ↔ {texts[1]}: {sim_related:.6f}")
        print(f"  {texts[0]} ↔ {texts[2]}: {sim_unrelated:.6f}")
        assert sim_related > sim_unrelated, (
            "同义句的相似度应高于无关句，"
            f"当前 {sim_related:.6f} <= {sim_unrelated:.6f}"
        )
        print("  [OK] 同义句相似度高于无关句")

        # 测试3：显式指定 dimensions 参数（仅部分模型支持）
        print("\n--- 测试3：指定 dimensions 参数 ---")
        if supports_dimensions(embedder.get_model_name()):
            embedder_custom = OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base_url,
                dimension=CUSTOM_DIMENSION,
            )
            custom_vector = embedder_custom.embed_text("向量维度测试")
            print(f"  get_dimension(): {embedder_custom.get_dimension()}")
            print(f"  实际返回维度: {len(custom_vector)}")
            assert embedder_custom.get_dimension() == CUSTOM_DIMENSION
            assert len(custom_vector) == CUSTOM_DIMENSION, (
                f"dimensions={CUSTOM_DIMENSION} 未生效，实际返回 {len(custom_vector)} 维"
            )
            print("  [OK] dimensions 参数生效")
        else:
            print(
                f"  [SKIP] {model} 不支持 dimensions 参数，"
                "如需验证请改用 text-embedding-v4 或 Qwen/Qwen3-Embedding-*"
            )

        print("\n==== 全部测试通过 ====")
        return True
    except Exception as exc:  # 集成测试需要把失败原因完整打印出来
        # tenacity 会把真实异常包进 RetryError，这里拆开显示鉴权/网络等真实原因
        detail = exc
        if isinstance(exc, RetryError):
            detail = exc.last_attempt.exception() or exc
        print(f"\n[FAIL] 测试失败: {type(detail).__name__}: {detail}")
        return False


if __name__ == "__main__":
    sys.exit(0 if test_embedding() else 1)
