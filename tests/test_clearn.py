# -*- coding: utf-8 -*-
"""TextClearn 文本清洗类的单元测试。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_clearn -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_clearn.py

说明：用标准库 unittest 编写，无需额外安装 pytest；
以后若安装了 pytest，这些用例同样能被 pytest 直接收集执行。
"""

import os
import sys
import unittest

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入 src.parsers.cleaner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.cleaner import TextClearn


# ======================================================================
# 一、构造函数与正则管理
# ======================================================================
class TestInit(unittest.TestCase):
    """__init__ 与 add_pattern 相关用例。"""

    def test_默认只含内置正则(self):
        tc = TextClearn()
        # 默认正则非空；用户没有追加时 user_patterns 应为空列表
        self.assertTrue(tc.default_patterns)
        self.assertEqual(tc.user_patterns, [])

    def test_初始化时可追加正则列表(self):
        tc = TextClearn(extra_patterns=[r"\d{4}-\d{2}-\d{2}", r"广告位招募.{0,20}"])
        self.assertEqual(len(tc.user_patterns), 2)

    def test_初始化时可追加单个正则字符串(self):
        # 图省事直接传字符串也应该被包装成列表登记
        tc = TextClearn(extra_patterns=r"\d{4}-\d{2}-\d{2}")
        self.assertEqual(len(tc.user_patterns), 1)

    def test_add_pattern_追加成功(self):
        tc = TextClearn()
        self.assertTrue(tc.add_pattern(r"\d{4}-\d{2}-\d{2}"))
        self.assertEqual(len(tc.user_patterns), 1)

    def test_add_pattern_重复追加返回False(self):
        tc = TextClearn()
        self.assertTrue(tc.add_pattern(r"\d{4}-\d{2}-\d{2}"))
        # 同一条正则不允许执行两次
        self.assertFalse(tc.add_pattern(r"\d{4}-\d{2}-\d{2}"))
        # 与默认正则重复同样不追加
        html_pattern = tc.default_patterns[0][1]
        self.assertFalse(tc.add_pattern(html_pattern))

    def test_add_pattern_非法正则抛ValueError(self):
        tc = TextClearn()
        with self.assertRaises(ValueError):
            tc.add_pattern("[a-")  # 未闭合的字符组，语法错误

    def test_add_pattern_空串和非字符串抛ValueError(self):
        tc = TextClearn()
        with self.assertRaises(ValueError):
            tc.add_pattern("")
        with self.assertRaises(ValueError):
            tc.add_pattern(123)  # 非字符串


# ======================================================================
# 二、clearn 主清洗流程
# ======================================================================
class TestClearnMain(unittest.TestCase):
    """clearn() 主清洗方法：默认正则 + 空白规范化。"""

    def setUp(self):
        # 每个用例都用一个全新的清洗器，互不影响
        self.tc = TextClearn()

    # ---------- 默认正则：删除类 ----------
    def test_删除HTML标签和注释(self):
        self.assertEqual(self.tc.clearn("<p>你好</p>"), "你好")
        self.assertEqual(self.tc.clearn("<br/>换行"), "换行")
        self.assertEqual(self.tc.clearn("<!-- 注释 -->正文"), "正文")

    def test_删除标签时链接随之消失(self):
        # URL 写在标签属性里，会随标签一起被删掉
        self.assertEqual(
            self.tc.clearn('<a href="https://x.com">链接</a>'), "链接"
        )

    def test_不会误删普通比较文本(self):
        # “a < b > c”不是 HTML 标签，应原样保留
        self.assertEqual(self.tc.clearn("a < b > c"), "a < b > c")

    def test_删除URL链接(self):
        self.assertEqual(
            self.tc.clearn("详见 https://example.com/a?b=1 结束"), "详见 结束"
        )
        self.assertEqual(self.tc.clearn("裸链接 www.example.com 哦"), "裸链接 哦")

    def test_删除邮箱(self):
        self.assertEqual(self.tc.clearn("邮箱 user@example.com 结束"), "邮箱 结束")

    def test_删除电话号码(self):
        self.assertEqual(self.tc.clearn("手机13812345678收到"), "手机收到")
        self.assertEqual(self.tc.clearn("座机010-1234567联系"), "座机联系")

    def test_删除表情符号(self):
        self.assertEqual(self.tc.clearn("你好😀世界✂"), "你好世界")

    def test_删除控制字符但保留制表符换行(self):
        # 其余 ASCII 控制符与 DEL 一律删除
        self.assertEqual(self.tc.clearn("a\x00b\x01c\x7fd"), "abcd")
        # \t \n \r 不在删除范围，只做后面的空白规范化
        self.assertEqual(self.tc.clearn("a\tb"), "a b")

    # ---------- 空白规范化 ----------
    def test_空白规范化_换行符统一(self):
        # Windows 换行 \r\n、老 Mac 换行 \r 统一成 \n
        self.assertEqual(self.tc.clearn("a\r\nb"), "a\nb")
        self.assertEqual(self.tc.clearn("a\rb"), "a\nb")

    def test_空白规范化_全角空格转半角(self):
        # 全角空格(U+3000)、不换行空格(U+00A0) → 半角空格
        self.assertEqual(self.tc.clearn("a\u3000b\u00a0c"), "a b c")

    def test_空白规范化_压缩与去首尾(self):
        self.assertEqual(self.tc.clearn("a    b"), "a b")          # 连续空格压成 1 个
        self.assertEqual(self.tc.clearn("a\n\n\n\nb"), "a\n\nb")   # 3 个以上换行压成 2 个
        self.assertEqual(self.tc.clearn("   你好   "), "你好")      # 去首尾空白

    # ---------- 综合与边界 ----------
    def test_混合噪音一次洗净(self):
        raw = (
            "  <p>你好</p>\r\n\r\n\r\n\r\n world https://a.com/x?y=1 "
            "x@y.com \u3000\U0001F600 end\x00\t\t尾部  "
        )
        self.assertEqual(self.tc.clearn(raw), "你好\n\n world end 尾部")

    def test_用户自定义正则生效_初始化时传入(self):
        tc = TextClearn(extra_patterns=r"\d{4}-\d{2}-\d{2}")
        self.assertEqual(tc.clearn("日期 2024-01-02 保留"), "日期 保留")

    def test_用户自定义正则生效_事后追加(self):
        self.tc.add_pattern(r"\d{4}-\d{2}-\d{2}")
        self.assertEqual(self.tc.clearn("日期 2024-01-02 保留"), "日期 保留")

    def test_None返回空字符串(self):
        self.assertEqual(self.tc.clearn(None), "")

    def test_非字符串抛TypeError(self):
        with self.assertRaises(TypeError):
            self.tc.clearn(123)


# ======================================================================
# 三、其他数据清洗方法（可单独调用）
# ======================================================================
class TestOtherCleanMethods(unittest.TestCase):
    """各独立清洗方法。"""

    def setUp(self):
        self.tc = TextClearn()

    def test_remove_html_tags(self):
        self.assertEqual(self.tc.remove_html_tags("<b>粗体</b>普通"), "粗体普通")

    def test_remove_urls(self):
        # URL 与文字之间没有空格分隔时，\S+ 会贪婪匹配到下一个空白，
        # 链接后面紧跟的文字会被一起删掉（正则无法识别链接的真实边界）
        self.assertEqual(self.tc.remove_urls("看https://a.com好东西"), "看")
        # 用空格分隔时只删链接本身；单独调用不做空白压缩，会留下两个空格
        self.assertEqual(self.tc.remove_urls("看 https://a.com 好东西"), "看  好东西")

    def test_remove_emails(self):
        self.assertEqual(self.tc.remove_emails("联系user@example.com哦"), "联系哦")

    def test_remove_phone_numbers(self):
        self.assertEqual(self.tc.remove_phone_numbers("打13812345678给我"), "打给我")

    def test_remove_emoji(self):
        self.assertEqual(self.tc.remove_emoji("赞👍"), "赞")

    def test_remove_control_chars(self):
        self.assertEqual(self.tc.remove_control_chars("a\x02b\tc"), "ab\tc")

    def test_fullwidth_to_halfwidth(self):
        self.assertEqual(self.tc.fullwidth_to_halfwidth("ＡＢＣ１２３"), "ABC123")
        self.assertEqual(self.tc.fullwidth_to_halfwidth("！ａｂｃ"), "!abc")
        self.assertEqual(self.tc.fullwidth_to_halfwidth("　"), " ")  # 全角空格
        # 句号“。”不在全角 ASCII 区，保持不变；普通中文不受影响
        self.assertEqual(self.tc.fullwidth_to_halfwidth("ＡＢＣ１２３！ａｂｃ，。"), "ABC123!abc,。")
        self.assertEqual(self.tc.fullwidth_to_halfwidth("中文"), "中文")

    def test_normalize_whitespace(self):
        self.assertEqual(self.tc.normalize_whitespace("a   b\n\n\n\n\nc"), "a b\n\nc")
        self.assertEqual(self.tc.normalize_whitespace("  x  "), "x")

    def test_清洗方法非字符串抛TypeError(self):
        with self.assertRaises(TypeError):
            self.tc.remove_html_tags(None)
        with self.assertRaises(TypeError):
            self.tc.fullwidth_to_halfwidth(123)

    def test_未注册的正则说明抛KeyError(self):
        with self.assertRaises(KeyError):
            self.tc._get_compiled("不存在的正则说明")


if __name__ == "__main__":
    # 支持直接运行本文件，-v 输出每条用例的名字
    unittest.main(verbosity=2)
