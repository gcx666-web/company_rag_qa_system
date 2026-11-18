# -*- coding: utf-8 -*-
"""LLM 与 RAG 模块的测试（Mock 单元测试 + 真实模型集成测试）。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    # 1. 真实模型用例依赖项目根目录 .env 里的 key（.env 已被 .gitignore 忽略），
    #    当前仓库已配置好的 SILICONFLOW_API_KEY 即可直接使用
    SILICONFLOW_API_KEY=sk-xxxx
    # 如需换成别家，可额外配置（优先级最高）：
    # LLM_API_KEY=sk-xxxx
    # LLM_BASE_URL=https://api.deepseek.com/v1
    # LLM_MODEL=deepseek-chat

    # 2. 运行全部用例（单元 + 真实调用）
    .venv/Scripts/python.exe -m unittest tests.test_llm_rag -v

    # 也可以直接运行本文件
    .venv/Scripts/python.exe tests/test_llm_rag.py

配置解析优先级（resolve_llm_config）：
    LLM_API_KEY         -> 显式配置，base_url/model 可用 LLM_BASE_URL / LLM_MODEL 覆盖，
                           未覆盖时默认硅基流动 + deepseek-ai/DeepSeek-V4.1-Flash
    SILICONFLOW_API_KEY -> https://api.siliconflow.cn/v1 + deepseek-ai/DeepSeek-V4.1-Flash
    DASHSCOPE_API_KEY   -> 阿里云百炼兼容模式 + qwen-plus
    都没有              -> 真实调用用例统一 [SKIP]，单元用例照常通过

说明：
1. 单元部分用 mock 替换 openai.OpenAI / client.chat.completions.create，
   不联网、毫秒级，验证入参透传、返回结构、异常不被吞、格式化与提示词拼装；
2. 真实部分真实请求模型服务：4 次问答调用 + 1 次故意用无效 key 触发鉴权失败，
   DeepSeek-V4.1-Flash 属 Flash 级模型，总消耗可忽略；
3. src/llm/llm.py 的默认 base_url（https://api.deepseek.ai/v1/）与默认模型名
   （硅基流动的 deepseek-ai/ 命名）并不匹配，因此真实用例一律显式传 base_url，
   不使用构造函数默认值，避免把请求发到错误的地址；
4. 用标准库 unittest 编写，无需额外安装 pytest。
"""

import os
import sys
import unittest
from unittest import mock

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import openai
from dotenv import load_dotenv

from src.llm.llm import LLM
from src.llm.rag import build_prompt, format_context, generate_answer

# 各供应商的默认配置
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen-plus"


def resolve_llm_config():
    """按 .env / 环境变量解析出 (api_key, base_url, model, key_source)。

    优先级：
    1. LLM_API_KEY -> 可用 LLM_BASE_URL / LLM_MODEL 覆盖（默认硅基流动 Flash）
    2. SILICONFLOW_API_KEY -> 硅基流动默认配置
    3. DASHSCOPE_API_KEY -> 阿里云百炼默认配置
    这样只配置某一家的 key 时，不会把 key 误发到另一家的接口上。
    返回 None 表示没有任何可用配置，真实调用用例会跳过。
    """
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    explicit_key = os.getenv("LLM_API_KEY")
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")

    if explicit_key:
        api_key, key_source = explicit_key, "LLM_API_KEY"
        default_url, default_model = SILICONFLOW_BASE_URL, SILICONFLOW_MODEL
    elif siliconflow_key:
        api_key, key_source = siliconflow_key, "SILICONFLOW_API_KEY"
        default_url, default_model = SILICONFLOW_BASE_URL, SILICONFLOW_MODEL
    elif dashscope_key:
        api_key, key_source = dashscope_key, "DASHSCOPE_API_KEY"
        default_url, default_model = DASHSCOPE_BASE_URL, DASHSCOPE_MODEL
    else:
        return None

    base_url = os.getenv("LLM_BASE_URL", default_url)
    model = os.getenv("LLM_MODEL", default_model)
    return api_key, base_url, model, key_source


class FakeUsage:
    """模拟 openai 返回的 usage 对象（真实对象带 prompt_tokens 等属性）。"""

    def __init__(self, prompt_tokens=10, completion_tokens=5, total_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = (
            prompt_tokens + completion_tokens if total_tokens is None else total_tokens
        )

    def __repr__(self):
        return (
            f"FakeUsage(prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"total_tokens={self.total_tokens})"
        )


def make_completion(content="测试回答", finish_reason="stop", usage=None):
    """构造与 chat.completions.create 返回值同构的假响应对象。"""
    response = mock.Mock()
    response.choices = [mock.Mock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = finish_reason
    response.usage = FakeUsage() if usage is None else usage
    return response


def describe_api_error(exc):
    """把真实调用的异常转成带排障提示的可读文本。"""
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, openai.NotFoundError):
        text += "\n提示：模型名或接口地址可能不对，可用 LLM_MODEL / LLM_BASE_URL 指定"
    elif isinstance(exc, openai.AuthenticationError):
        text += "\n提示：API Key 无效，或 key 与 base_url 不属于同一平台"
    elif isinstance(exc, openai.PermissionDeniedError):
        text += "\n提示：当前 key 没有该模型权限，可换模型（LLM_MODEL）"
    elif isinstance(exc, openai.APIConnectionError):
        text += "\n提示：网络不可达，请检查 base_url / 代理设置"
    return text


class StubLLM:
    """假 LLM：记录每次 generate 的入参并返回预设响应，不联网。

    只实现 generate 与 model 属性，因为 generate_answer 仅依赖这两者。
    """

    def __init__(self, content="测试回答", usage=None, response=None):
        self.model = "stub-model"
        self.calls = []
        if response is not None:
            self.response = dict(response)
        else:
            self.response = {
                "content": content,
                "finish_reason": "stop",
                "token_usage": {} if usage is None else usage,
            }

    def generate(self, message, temperature=0.7, max_tokens=2000):
        self.calls.append(
            {
                "message": message,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return dict(self.response)

    @property
    def last_call(self):
        return self.calls[-1]


# ======================================================================
# 一、resolve_llm_config：配置优先级（决定真实用例是否 skip）
# ======================================================================
class TestResolveLLMConfig(unittest.TestCase):
    """没有 key 时必须返回 None，避免把 key 发到不匹配的平台。"""

    def resolve(self, **env):
        """在清空的临时环境变量中解析配置，不真的读 .env。"""
        with mock.patch.dict(globals(), {"load_dotenv": lambda *a, **k: None}):
            with mock.patch.dict(os.environ, env, clear=True):
                return resolve_llm_config()

    def test_无任何key返回None(self):
        self.assertIsNone(self.resolve())

    def test_LLM_API_KEY优先级最高且可覆盖地址与模型(self):
        config = self.resolve(
            LLM_API_KEY="sk-explicit",
            SILICONFLOW_API_KEY="sk-siliconflow",
            LLM_BASE_URL="https://other.example/v1",
            LLM_MODEL="other-model",
        )

        self.assertEqual(
            config,
            ("sk-explicit", "https://other.example/v1", "other-model", "LLM_API_KEY"),
        )

    def test_SILICONFLOW默认配置(self):
        config = self.resolve(SILICONFLOW_API_KEY="sk-siliconflow")

        self.assertEqual(
            config,
            (
                "sk-siliconflow",
                SILICONFLOW_BASE_URL,
                SILICONFLOW_MODEL,
                "SILICONFLOW_API_KEY",
            ),
        )

    def test_DASHSCOPE默认配置(self):
        config = self.resolve(DASHSCOPE_API_KEY="sk-dashscope")

        self.assertEqual(
            config,
            (
                "sk-dashscope",
                DASHSCOPE_BASE_URL,
                DASHSCOPE_MODEL,
                "DASHSCOPE_API_KEY",
            ),
        )

    def test_地址与模型可单独覆盖(self):
        config = self.resolve(
            SILICONFLOW_API_KEY="sk-siliconflow", LLM_MODEL="deepseek-ai/DeepSeek-V3.2"
        )

        self.assertEqual(
            config,
            (
                "sk-siliconflow",
                SILICONFLOW_BASE_URL,
                "deepseek-ai/DeepSeek-V3.2",
                "SILICONFLOW_API_KEY",
            ),
        )


# ======================================================================
# 二、LLM 单元测试：不联网，用 mock 替换 openai.OpenAI
# ======================================================================
class TestLLMInit(unittest.TestCase):
    """构造函数：字段保存 + OpenAI 客户端初始化参数。"""

    def test_构造函数保存参数并创建客户端(self):
        with mock.patch("src.llm.llm.openai.OpenAI") as fake_openai:
            llm = LLM(
                api_key="sk-test",
                model="test-model",
                base_url="https://example.com/v1",
            )

        self.assertEqual(llm.api_key, "sk-test")
        self.assertEqual(llm.model, "test-model")
        self.assertEqual(llm.base_url, "https://example.com/v1")
        # base_url 必须真的传给客户端，否则请求会发往 openai 官方地址
        fake_openai.assert_called_once_with(
            api_key="sk-test", base_url="https://example.com/v1"
        )
        self.assertIs(llm.client, fake_openai.return_value)

    def test_默认模型与默认接口地址(self):
        """固化当前默认值：默认地址与默认模型名不属于同一平台，真实用例须显式传参。"""
        with mock.patch("src.llm.llm.openai.OpenAI"):
            llm = LLM(api_key="sk-test")

        self.assertEqual(llm.model, "deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(llm.base_url, "https://api.deepseek.ai/v1/")
        self.assertEqual(llm.api_key, "sk-test")


class TestLLMGenerateMocked(unittest.TestCase):
    """generate：入参透传、返回结构、异常不被吞。"""

    def setUp(self):
        with mock.patch("src.llm.llm.openai.OpenAI"):
            self.llm = LLM(
                api_key="sk-test",
                model="test-model",
                base_url="https://example.com/v1",
            )
        self.create = self.llm.client.chat.completions.create
        self.messages = [{"role": "user", "content": "你好"}]

    def test_返回内容与用量透传(self):
        usage = FakeUsage(prompt_tokens=7, completion_tokens=3)
        self.create.return_value = make_completion(
            content="你好，有什么可以帮你？", finish_reason="stop", usage=usage
        )

        result = self.llm.generate(self.messages)

        self.assertEqual(set(result), {"content", "finish_reason", "token_usage"})
        self.assertEqual(result["content"], "你好，有什么可以帮你？")
        self.assertEqual(result["finish_reason"], "stop")
        # usage 原样透传，不做二次包装
        self.assertIs(result["token_usage"], usage)
        self.assertEqual(result["token_usage"].total_tokens, 10)

    def test_请求参数默认值(self):
        self.create.return_value = make_completion()

        self.llm.generate(self.messages)

        self.create.assert_called_once_with(
            model="test-model",
            messages=self.messages,
            temperature=0.7,
            max_tokens=2000,
        )

    def test_请求参数可覆盖(self):
        self.create.return_value = make_completion()

        self.llm.generate(self.messages, temperature=0.0, max_tokens=64)

        kwargs = self.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["messages"], self.messages)

    def test_空内容原样返回(self):
        """模型只返回 reasoning/被截断时 content 可能为空，generate 不做兜底改写。"""
        self.create.return_value = make_completion(content="", finish_reason="length")

        result = self.llm.generate(self.messages, max_tokens=8)

        self.assertEqual(result["content"], "")
        self.assertEqual(result["finish_reason"], "length")

    def test_接口异常不被吞(self):
        self.create.side_effect = RuntimeError("模拟接口失败")

        with self.assertRaises(RuntimeError):
            self.llm.generate(self.messages)


# ======================================================================
# 三、format_context：上下文格式化（纯函数，无网络）
# ======================================================================
class TestFormatContext(unittest.TestCase):
    """按正文去重、跳过空白、字符预算、双换行拼接。"""

    def test_空输入返回空字符串(self):
        cases = [
            ([], "空列表"),
            ([{"content": ""}], "空内容"),
            ([{"content": "   \n\t "}], "纯空白内容"),
            ([{}], "缺少 content 字段"),
        ]
        for result, label in cases:
            with self.subTest(label=label):
                self.assertEqual(format_context(result), "")

    def test_按正文去重(self):
        result = [
            {"content": "公司年假15天"},
            {"content": "公司年假15天"},
            {"content": "请假提前3天"},
        ]

        context = format_context(result)

        self.assertEqual(context, "公司年假15天\n\n请假提前3天")
        self.assertEqual(context.count("公司年假15天"), 1)

    def test_双换行拼接并保持顺序(self):
        result = [{"content": "第一条"}, {"content": "第二条"}, {"content": "第三条"}]

        self.assertEqual(format_context(result), "第一条\n\n第二条\n\n第三条")

    def test_内容会先strip(self):
        result = [{"content": "  第一条 \n"}, {"content": "\t第二条  "}]

        self.assertEqual(format_context(result), "第一条\n\n第二条")

    def test_字符预算包含分隔符(self):
        # 每条 4 个字；max_token 按字符计，第 2 条起额外占 2 个换行符
        result = [{"content": "第一句话"}, {"content": "第二句话"}]

        self.assertEqual(format_context(result, max_token=8), "第一句话")
        # 4 + 4 + 2 = 10，预算 12 时两条都能放下
        self.assertEqual(
            format_context(result, max_token=12), "第一句话\n\n第二句话"
        )

    def test_单条超预算被整条丢弃(self):
        result = [{"content": "0123456789"}]

        self.assertEqual(format_context(result, max_token=9), "")

    def test_跳过超预算条目后继续尝试后续短文本(self):
        result = [
            {"content": "四字"},          # 2 字，收入，total = 2
            {"content": "0123456789"},   # 2 + 10 + 2 = 14 > 12，跳过
            {"content": "结尾"},          # 2 + 2 + 2 = 6 <= 12，收入
        ]

        self.assertEqual(format_context(result, max_token=12), "四字\n\n结尾")

    def test_默认预算为2000字符(self):
        # 单条约 1500 字符，两条合计 2502 > 2000，第二条被丢弃
        chunk = "字" * 1500
        result = [{"content": chunk}, {"content": "尾" * 1000}]

        self.assertEqual(format_context(result), chunk)


# ======================================================================
# 四、build_prompt：提示词拼装（纯函数，无网络）
# ======================================================================
class TestBuildPrompt(unittest.TestCase):
    """指令 + 三段标题，问题在前、上下文在后。"""

    def setUp(self):
        self.question = "公司年假有多少天？"
        self.context = "公司员工年假为15天"
        self.prompt = build_prompt(self.question, self.context)

    def test_包含固定指令(self):
        self.assertIn("你是一个知识库问答助手", self.prompt)
        # 指令里约定缺信息时返回固定话术
        self.assertIn("我不知道", self.prompt)

    def test_三段标题齐全且以答案段结尾(self):
        for title in ("【问题】：", "【上下文】：", "【答案】："):
            with self.subTest(title=title):
                self.assertIn(title, self.prompt)
        self.assertTrue(self.prompt.rstrip().endswith("【答案】："))

    def test_问题与上下文出现在各自段落(self):
        question_index = self.prompt.index(self.question)
        context_index = self.prompt.index(self.context)
        self.assertLess(self.prompt.index("【问题】："), question_index)
        self.assertLess(question_index, self.prompt.index("【上下文】："))
        self.assertLess(self.prompt.index("【上下文】："), context_index)
        self.assertLess(context_index, self.prompt.index("【答案】："))

    def test_空上下文也能拼装(self):
        prompt = build_prompt("问题", "")

        self.assertIn("【上下文】：", prompt)
        self.assertIn("问题", prompt)
        self.assertTrue(prompt.rstrip().endswith("【答案】："))


# ======================================================================
# 五、generate_answer 单元测试：用假 LLM 验证编排逻辑（无网络）
# ======================================================================
class TestGenerateAnswerUnit(unittest.TestCase):
    """空结果短路、提示词拼装、返回值字段与用量透传。"""

    def test_空结果直接返回没有找到答案(self):
        stub = StubLLM()

        response = generate_answer(stub, "公司年假有多少天？", [])

        self.assertEqual(response["answer"], "没有找到答案")
        self.assertEqual(response["question"], "公司年假有多少天？")
        # 空结果分支不返回 context/model/tokens_used
        self.assertEqual(set(response), {"answer", "question"})
        # 且不调用模型，避免无意义的费用
        self.assertEqual(stub.calls, [])

    def test_正常路径组织提示词并透传(self):
        usage = FakeUsage(prompt_tokens=30, completion_tokens=8)
        stub = StubLLM(content="公司年假为15天。", usage=usage)
        result = [
            {"content": "公司员工年假为15天"},
            {"content": "请假需提前3天申请", "score": 0.8},
        ]

        response = generate_answer(
            stub, "公司年假有多少天？", result, temperature=0.2, max_token=500
        )

        self.assertEqual(
            set(response),
            {"answer", "question", "context", "tokens_used", "model"},
        )
        self.assertEqual(response["answer"], "公司年假为15天。")
        self.assertEqual(response["question"], "公司年假有多少天？")
        self.assertEqual(response["context"], format_context(result, 500))
        self.assertEqual(
            response["context"], "公司员工年假为15天\n\n请假需提前3天申请"
        )
        self.assertEqual(response["model"], "stub-model")
        self.assertIs(response["tokens_used"], usage)

        call = stub.last_call
        self.assertEqual(call["temperature"], 0.2)
        self.assertEqual(call["max_tokens"], 500)
        self.assertEqual(len(call["message"]), 1)
        self.assertEqual(call["message"][0]["role"], "user")
        prompt = call["message"][0]["content"]
        self.assertIn("公司年假有多少天？", prompt)
        self.assertIn(response["context"], prompt)

    def test_回答会被strip(self):
        stub = StubLLM(content="  15天 \n")

        response = generate_answer(stub, "年假多少天？", [{"content": "年假15天"}])

        self.assertEqual(response["answer"], "15天")

    def test_模型缺少content字段时回答为空(self):
        """真实接口被截断/只返回 reasoning 时没有 content，这里不抛异常也不兜底。"""
        stub = StubLLM(response={"finish_reason": "length"})

        response = generate_answer(stub, "年假多少天？", [{"content": "年假15天"}])

        self.assertEqual(response["answer"], "")
        self.assertEqual(response["tokens_used"], {})
        self.assertEqual(response["context"], "年假15天")

    def test_上下文超预算时只带部分内容(self):
        stub = StubLLM()
        result = [
            {"content": "第一条" * 10},   # 30 字，收入，total = 30
            {"content": "第二条" * 10},   # 30 + 30 + 2 = 62 > 40，跳过
        ]

        response = generate_answer(
            stub, "问题", result, max_token=40
        )

        prompt = stub.last_call["message"][0]["content"]
        self.assertEqual(response["context"], "第一条" * 10)
        self.assertIn("第一条" * 10, prompt)
        self.assertNotIn("第二条", prompt)


# ======================================================================
# 六、真实模型集成测试：真实请求 .env 配置的大模型
# ======================================================================
class TestLLMRagWithRealModel(unittest.TestCase):
    """真实调用 LLM.generate 与 generate_answer，验证端到端链路。

    依赖 .env 中的 key（见文件头），未配置时整类 [SKIP]；
    每个用例都会打印真实返回内容，便于人工确认模型行为。
    """

    @classmethod
    def setUpClass(cls):
        config = resolve_llm_config()
        if config is None:
            raise unittest.SkipTest(
                "未配置 LLM_API_KEY / SILICONFLOW_API_KEY / DASHSCOPE_API_KEY，"
                "请在项目根目录 .env 中配置后重跑真实模型用例"
            )

        api_key, base_url, model, key_source = config
        cls.config = config
        cls.llm = LLM(api_key=api_key, model=model, base_url=base_url)

        print("\n[INFO] 真实模型用例")
        print(f"  Key 来源: {key_source}（不显示具体值）")
        print(f"  模型: {model}")
        print(f"  接口地址: {base_url}")
        if key_source == "SILICONFLOW_API_KEY" and "siliconflow" not in base_url:
            print("[WARN] 使用的是 SILICONFLOW_API_KEY，但 base_url 不是硅基流动地址")
        if key_source == "DASHSCOPE_API_KEY" and "dashscope" not in base_url:
            print("[WARN] 使用的是 DASHSCOPE_API_KEY，但 base_url 不是百炼地址")

    def call_generate(self, messages, **kwargs):
        """真实调用 LLM.generate；失败时把鉴权/网络/模型原因完整带出来。"""
        try:
            return self.llm.generate(messages, **kwargs)
        except Exception as exc:
            self.fail(f"真实模型调用失败 -> {describe_api_error(exc)}")

    def call_answer(self, question, result, **kwargs):
        """真实调用 generate_answer（格式化上下文 + 拼提示词 + 请求模型）。"""
        try:
            return generate_answer(self.llm, question, result, **kwargs)
        except Exception as exc:
            self.fail(f"真实模型问答失败 -> {describe_api_error(exc)}")

    def test_LLM_generate_真实调用(self):
        messages = [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "只回复两个字：收到"},
        ]

        result = self.call_generate(messages, temperature=0.0, max_tokens=512)
        usage = result["token_usage"]

        print(f"\n[真实返回] content={result['content']!r}")
        print(f"[真实返回] finish_reason={result['finish_reason']}")
        print(
            f"[真实返回] usage: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )

        self.assertEqual(set(result), {"content", "finish_reason", "token_usage"})
        self.assertIsInstance(result["content"], str)
        self.assertTrue(result["content"].strip(), "模型返回了空内容")
        self.assertIn(result["finish_reason"], {"stop", "length"})
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.total_tokens, 0)

    def test_LLM_generate_max_tokens生效(self):
        """max_tokens 必须真的传到服务端：要求长回答但上限很小，回答应被截断。

        注意：DeepSeek-V4.1-Flash 属推理模型，服务端把 reasoning token 也计入
        completion_tokens（实测 max_tokens=32 时 completion_tokens 仍可上千），
        因此这里不用 ``completion_tokens <= max_tokens`` 断言，改用
        finish_reason == "length" 证明上限确实生效。
        """
        messages = [
            {"role": "user", "content": "请详细介绍杭州西湖的历史，至少写 800 字。"}
        ]

        result = self.call_generate(messages, temperature=0.0, max_tokens=32)
        usage = result["token_usage"]

        print(
            f"\n[真实返回] finish_reason={result['finish_reason']} "
            f"completion_tokens={usage.completion_tokens} "
            f"reasoning_tokens={getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', None)} "
            f"content={result['content']!r}"
        )

        self.assertEqual(result["finish_reason"], "length")
        self.assertGreater(usage.completion_tokens, 0)

    def test_generate_answer_真实问答(self):
        result = [
            {"content": "公司员工年假为15天，入职满一年后即可享受。"},
            {
                "content": "请假需提前3天通过OA系统提交申请，由直属主管审批。",
                "score": 0.8,
            },
        ]

        response = self.call_answer(
            "公司年假有多少天？", result, temperature=0.0, max_token=1000
        )
        print(f"\n[真实回答] {response['answer']}")
        print(f"[真实回答] context={response['context']!r}")
        print(f"[真实回答] tokens_used={response['tokens_used']}")

        self.assertEqual(
            set(response),
            {"answer", "question", "context", "tokens_used", "model"},
        )
        self.assertEqual(response["question"], "公司年假有多少天？")
        self.assertEqual(response["model"], self.llm.model)
        self.assertEqual(response["context"], format_context(result, 1000))
        self.assertTrue(response["answer"].strip(), "模型返回了空答案")
        self.assertIn("15", response["answer"])
        self.assertIsNotNone(response["tokens_used"])

    def test_generate_answer_上下文无关(self):
        result = [{"content": "苹果手机是电子产品，属于消费类电子。"}]

        response = self.call_answer(
            "公司年假有多少天？", result, temperature=0.0, max_token=1000
        )
        print(f"\n[真实回答-无相关信息] {response['answer']}")

        self.assertTrue(response["answer"].strip(), "模型返回了空答案")
        if "我不知道" not in response["answer"]:
            # 指令遵从存在随机性，这里只做结构断言并提示
            print("[WARN] 未按提示词返回“我不知道”，仅校验结构与调用成功")

    def test_无效key抛出APIStatusError(self):
        """鉴权失败必须真实抛出，不被吞掉（免费，不消耗额度）。"""
        _, base_url, model, _ = self.config
        bad_llm = LLM(
            api_key="sk-invalid-key-for-test", model=model, base_url=base_url
        )

        with self.assertRaises(openai.APIStatusError) as ctx:
            bad_llm.generate([{"role": "user", "content": "你好"}], max_tokens=8)

        print(f"\n[预期失败] {describe_api_error(ctx.exception)}")
        self.assertGreaterEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
