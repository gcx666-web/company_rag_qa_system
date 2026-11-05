# -*- coding: utf-8 -*-
"""
cleaner.py —— 文本数据清洗模块。

提供 TextClearn 类，对（从 PDF / Word / 网页等来源解析出来的）原始文本做统一清洗：

- 删除 HTML 标签、URL、邮箱、电话、emoji 表情、不可见控制字符等噪音；
- 支持初始化时追加，以及事后随时追加用户自定义正则；
- 另附全角转半角、空白规范化等独立清洗方法，可单独调用。
"""

import re

__all__ = ["TextClearn"]


class TextClearn:
    r"""
    文本清洗类。

    典型用法::

        # 1) 只用内置默认正则
        tc = TextClearn()
        clean_text = tc.clearn(raw_text)

        # 2) 初始化时追加用户自定义正则（支持单个字符串或字符串列表）
        tc = TextClearn(extra_patterns=[r"\d{4}-\d{2}-\d{2}"])

        # 3) 对象创建之后仍可随时追加
        tc.add_pattern(r"广告位招募.{0,20}")

        # 4) 各清洗方法也可以单独调用
        no_html = tc.remove_html_tags(raw_text)
        half    = tc.fullwidth_to_halfwidth(raw_text)

    设计约定：
    - 正则列表里匹配到的内容一律替换为空字符串（全角空格除外，
      它会在空白规范化阶段被替换成普通半角空格，避免相邻词粘在一起）；
    - clearn() 只做“正则删除 + 空白规范化”，全角转半角是可选步骤，
      需要时请显式调用 fullwidth_to_halfwidth()。
    """

    # ==================================================================
    # 一、构造与正则管理
    # ==================================================================
    def __init__(self, extra_patterns=None):
        """
        构造函数：登记默认正则，并可选地追加用户自定义正则。

        :param extra_patterns: 用户自定义正则。可以是单个正则字符串，
            也可以是字符串列表；为 None（默认）时只使用内置默认正则。
            自定义正则统一排在默认正则之后执行。
        :raises ValueError: extra_patterns 中存在非法正则时抛出。
        """
        # ---------------- 默认正则 ----------------
        # 每一项是 (中文说明, 正则字符串) 二元组，列表顺序即执行顺序：
        # 先删结构性噪音（标签、链接），再删字符级噪音（表情、控制符）。
        # 注意：邮箱、电话属于隐私脱敏用的“宁可错杀”项，订单号等纯数字
        # 串可能被电话正则误删；如果业务上需要保留，删掉对应条目即可。
        self.default_patterns = [
            # HTML 注释（<!-- ... -->）或标签（<p>、</div>、<br/> 等）。
            # 要求标签以字母开头，避免把 “a < b > c” 这类普通比较文本误删。
            ("HTML标签", r"<!--[\s\S]*?-->|</?[A-Za-z][^<>]*>"),
            # 网页链接：http(s):// 或 ftp:// 开头，以及裸写的 www. 开头链接。
            ("URL链接", r"(?:https?|ftp)://\S+|www\.\S+"),
            # 电子邮箱：常见的 user@domain.tld 形式。
            ("邮箱", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            # 电话号码：11 位手机号（1 开头、第二位 3-9），或“区号-号码”座机。
            ("电话号码", r"1[3-9]\d{9}|0\d{2,3}-\d{7,8}"),
            # emoji 表情：覆盖常见区间（U+1F000~U+1FAFF、U+2600~U+27BF）。
            # 不是穷举，遇到漏网的生僻表情，往 user_patterns 里补正则即可。
            ("表情符号", r"[\U0001F000-\U0001FAFF\u2600-\u27BF]"),
            # 不可见 ASCII 控制字符与 DEL(0x7F)。
            # 刻意保留 \t(0x09)、\n(0x0A)、\r(0x0D)，它们由空白规范化处理。
            ("控制字符", r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"),
            #第几页码
            ("页码", r"第\d+页")
            
        ]

        # 用户追加的正则：格式与 default_patterns 相同，排在默认正则之后执行
        self.user_patterns = []

        # 编译缓存：[(中文说明, 已编译的 Pattern), ...]。
        # 在构造时统一编译好，clearn() 每次调用直接使用，
        # 避免每次清洗都重新编译正则的开销。
        self._compiled = [(name, re.compile(p)) for name, p in self.default_patterns]

        # 登记用户在初始化时传入的自定义正则（复用 add_pattern 做校验和去重）
        if extra_patterns:
            # 允许图省事直接传一个字符串，这里统一包装成列表再逐条登记
            if isinstance(extra_patterns, str):
                extra_patterns = [extra_patterns]
            for pattern in extra_patterns:
                self.add_pattern(pattern)

    def add_pattern(self, pattern: str, name: str = "用户自定义") -> bool:
        """
        追加一条用户自定义正则（对象创建后仍可随时调用）。

        :param pattern: 正则字符串，必须是非空 str。
        :param name:    该正则的中文说明，仅用于调试时辨认，不影响执行。
        :return:        True 表示追加成功；与已有正则重复时返回 False
                        （重复的正则不会被执行两次）。
        :raises ValueError: pattern 不是非空字符串，或正则语法非法时抛出。
        """
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern 必须是非空字符串")
        # 去重：在默认正则和已追加的用户正则里各查一遍
        existing = [p for _, p in self.default_patterns] + [p for _, p in self.user_patterns]
        if pattern in existing:
            return False
        try:
            compiled = re.compile(pattern)  # 先试编译，语法错误在这一步暴露
        except re.error as exc:
            raise ValueError(f"非法正则表达式 {pattern!r}：{exc}") from exc
        # 同时登记到“用户正则列表”和“编译缓存”，保持两处一致
        self.user_patterns.append((name, pattern))
        self._compiled.append((name, compiled))
        return True

    # ==================================================================
    # 二、主清洗入口
    # ==================================================================
    def clearn(self, text) -> str:
        """
        主清洗方法：依次套用全部正则，最后做空白规范化。

        执行顺序：
            1. 默认正则：HTML标签 → URL → 邮箱 → 电话 → 表情 → 控制字符；
            2. 用户自定义正则：按追加顺序依次执行；
            3. normalize_whitespace()：换行符统一、空白压缩、去首尾空白。

        :param text: 待清洗的原始文本；传 None 时返回空字符串。
        :return:     清洗后的字符串。
        :raises TypeError: text 不是 str/None 时抛出。
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            raise TypeError(f"clearn() 只接受 str，实际收到 {type(text).__name__}")
        # 逐条套用正则，匹配内容一律替换为空字符串
        for name, regex in self._compiled:
            text = regex.sub("", text)
        # 最后做空白规范化，保证输出整齐
        return self.normalize_whitespace(text)

    def _get_compiled(self, name: str):
        """
        内部辅助：按中文说明从编译缓存里取出对应的已编译正则。

        供下方各独立清洗方法复用，保证“单独调用”和“clearn 统一清洗”
        用的是同一份正则定义，改一处即可全局生效。

        :param name: 正则的中文说明，如 "HTML标签"。
        :raises KeyError: 说明不存在时抛出。
        """
        for n, regex in self._compiled:
            if n == name:
                return regex
        raise KeyError(f"未注册的正则说明：{name}")

    @staticmethod
    def _ensure_str(text) -> str:
        """内部辅助：类型检查，非 str 时抛出带实际类型的 TypeError。"""
        if not isinstance(text, str):
            raise TypeError(f"清洗方法只接受 str，实际收到 {type(text).__name__}")
        return text

    # ==================================================================
    # 三、其他数据清洗方法（均可单独调用）
    # ==================================================================
    def remove_html_tags(self, text) -> str:
        """删除 HTML/XML 标签和注释，只保留标签之间的文字。"""
        text = self._ensure_str(text)
        return self._get_compiled("HTML标签").sub("", text)

    def remove_urls(self, text) -> str:
        """删除 http(s)/ftp 链接以及 www. 开头的裸链接。"""
        text = self._ensure_str(text)
        return self._get_compiled("URL链接").sub("", text)

    def remove_emails(self, text) -> str:
        """删除电子邮箱地址（隐私脱敏）。"""
        text = self._ensure_str(text)
        return self._get_compiled("邮箱").sub("", text)

    def remove_phone_numbers(self, text) -> str:
        """删除手机号 / 座机号（隐私脱敏）。"""
        text = self._ensure_str(text)
        return self._get_compiled("电话号码").sub("", text)

    def remove_emoji(self, text) -> str:
        """删除常见区间的 emoji 表情符号。"""
        text = self._ensure_str(text)
        return self._get_compiled("表情符号").sub("", text)

    def remove_control_chars(self, text) -> str:
        """删除不可见控制字符（保留 \\t、\\n、\\r）。"""
        text = self._ensure_str(text)
        return self._get_compiled("控制字符").sub("", text)

    def fullwidth_to_halfwidth(self, text) -> str:
        """
        全角字符转半角（可选步骤，clearn() 默认不做这一步）。

        转换范围：
            - 全角空格（U+3000）→ 半角空格；
            - 全角可见字符区 U+FF01~U+FF5E（全角字母、数字、标点）→
              按固定偏移 0xFEE0 平移成对应半角字符。

        注意：全角逗号“，”位于该区间会被转成“,”，而句号“。”（U+3002）
        不在区间内、保持不变；如果不希望转换中文标点，请只对
        字母数字片段调用本方法。

        :param text: 任意字符串。
        :return:     转换后的字符串。
        """
        text = self._ensure_str(text)
        chars = []
        for ch in text:
            code = ord(ch)
            if code == 0x3000:              # 全角空格 → 半角空格
                chars.append(" ")
            elif 0xFF01 <= code <= 0xFF5E:  # 全角可见字符区，平移即得半角
                chars.append(chr(code - 0xFEE0))
            else:                           # 其余字符原样保留
                chars.append(ch)
        return "".join(chars)

    def normalize_whitespace(self, text) -> str:
        """
        空白规范化（clearn() 的最后一步，也可单独调用）。

        处理内容：
            1. Windows 换行 \\r\\n、老 Mac 换行 \\r → 统一成 \\n；
            2. 全角空格（U+3000）、不换行空格（U+00A0）→ 普通半角空格
               （用“替换”而不是“删除”，避免相邻词直接粘连）；
            3. 连续的空格 / 制表符 → 压缩成一个半角空格；
            4. 连续 3 个及以上的换行 → 压缩成 2 个（段落间最多留一个空行）；
            5. 去掉首尾空白。

        :param text: 任意字符串。
        :return:     规范化后的字符串。
        """
        text = self._ensure_str(text)
        # 1) 统一换行符，后续按 \n 处理
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 2) 全角空格 / 不换行空格 → 半角空格
        text = text.replace("\u3000", " ").replace("\u00a0", " ")
        # 3) 行内空白压缩（显式写 [ \t] 而不用 \s，避免误伤换行）
        text = re.sub(r"[ \t]+", " ", text)
        # 4) 多余空行压缩
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 5) 去首尾空白
        return text.strip()
