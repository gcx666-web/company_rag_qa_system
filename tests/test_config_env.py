# -*- coding: utf-8 -*-
"""配置层测试：枚举型环境变量剥注释 / 白名单回退（不启动服务、不联网）。

运行方式（项目根目录下）::

    .venv/Scripts/python.exe -m unittest tests.test_config_env -v
"""

import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.api.config import _env_choice, settings

# Chroma 支持的相似度空间（与 ChromaVectorStore 的白名单保持一致）
_DISTANCE_CHOICES = ("l2", "ip", "cosine")

# 模型加载模式（与 factory 的分支保持一致）
_EMBEDDING_MODES = ("auto", "online", "offline")


class EnvChoiceTest(unittest.TestCase):
    """_env_choice：行内注释、大小写、非法值与缺省值。"""

    def test_无空格行内注释被剥离(self):
        """python-dotenv 不会把 ``cosine#注释`` 当注释，这里必须自己剥掉。"""
        with mock.patch.dict(
            os.environ, {"CHROMA_DISTANCE": "cosine#向量相似度空间"}
        ):
            self.assertEqual(
                _env_choice("CHROMA_DISTANCE", "cosine", _DISTANCE_CHOICES), "cosine"
            )

    def test_带空格行内注释同样被剥离(self):
        with mock.patch.dict(os.environ, {"CHROMA_DISTANCE": "l2 # 欧氏距离"}):
            self.assertEqual(
                _env_choice("CHROMA_DISTANCE", "cosine", _DISTANCE_CHOICES), "l2"
            )

    def test_大小写与首尾空白被归一化(self):
        with mock.patch.dict(os.environ, {"CHROMA_DISTANCE": "  IP  "}):
            self.assertEqual(
                _env_choice("CHROMA_DISTANCE", "cosine", _DISTANCE_CHOICES), "ip"
            )

    def test_非法取值回退默认值(self):
        with mock.patch.dict(os.environ, {"CHROMA_DISTANCE": "manhattan"}):
            self.assertEqual(
                _env_choice("CHROMA_DISTANCE", "cosine", _DISTANCE_CHOICES), "cosine"
            )

    def test_未配置时使用默认值(self):
        env_without_distance = {
            key: value
            for key, value in os.environ.items()
            if key != "CHROMA_DISTANCE"
        }
        with mock.patch.dict(os.environ, env_without_distance, clear=True):
            self.assertEqual(
                _env_choice("CHROMA_DISTANCE", "cosine", _DISTANCE_CHOICES), "cosine"
            )


class SettingsTest(unittest.TestCase):
    """进程配置的枚举值必须合法（导入时读取 .env）。"""

    def test_向量空间在Chroma白名单内(self):
        """失败通常意味着 .env 里 CHROMA_DISTANCE 带了注释或拼写错误。"""
        self.assertIn(settings.CHROMA_DISTANCE, _DISTANCE_CHOICES)

    def test_向量模式在合法取值内(self):
        self.assertIn(settings.EMBEDDING_MODE, _EMBEDDING_MODES)


if __name__ == "__main__":
    unittest.main()
