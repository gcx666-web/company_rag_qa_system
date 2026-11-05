# -*- coding: utf-8 -*-
"""TableProcessor 表格处理类的单元测试。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_table_parser -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_table_parser.py

说明：只测 TableProcessor 本身（不加载 docling/PdfParser，速度快、无网络依赖），
用标准库 unittest 编写，无需额外安装 pytest。
"""

import os
import sys
import unittest

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.table_parser import TableProcessor


def make_table(header, rows, page=1):
    """构造 PdfParser 形状的表格字典 {'markdown','rows','page','cols'}。"""
    lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(['---'] * len(header)) + ' |',
    ]
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return {
        'markdown': '\n'.join(lines),
        'rows': [list(header)] + [list(r) for r in rows],
        'page': page,
        'cols': len(header),
    }


# ======================================================================
# 一、merge_tables：List[Dict] 进 / List[Dict] 出，跨页拆分表格合并
# ======================================================================
class TestMergeTables(unittest.TestCase):
    """合并规则：列数相同 + 页码连续 → 视为同一张表。"""

    def setUp(self):
        self.tp = TableProcessor()

    def test_页码连续且列数相同才合并(self):
        t1 = make_table(['姓名', '年龄'], [['张三', '28'], ['李四', '35']], page=1)
        t2 = make_table(['姓名', '年龄'], [['王五', '40']], page=2)
        out = self.tp.merge_tables([t1, t2])
        self.assertEqual(len(out), 1)
        # 输出形状与入参一致
        self.assertEqual(set(out[0].keys()), {'markdown', 'rows', 'page', 'cols'})
        self.assertEqual(
            out[0]['rows'],
            [['姓名', '年龄'], ['张三', '28'], ['李四', '35'], ['王五', '40']],
        )
        # 合并后的 markdown 是一整张表，包含全部数据行
        for name in ('张三', '李四', '王五'):
            self.assertIn(name, out[0]['markdown'])
        self.assertEqual(len(out[0]['markdown'].splitlines()), 5)  # 表头+分隔+3行

    def test_连续三页全部合并(self):
        t1 = make_table(['A', 'B'], [['1', '2']], page=1)
        t2 = make_table(['A', 'B'], [['3', '4']], page=2)
        t3 = make_table(['A', 'B'], [['5', '6']], page=3)
        out = self.tp.merge_tables([t1, t2, t3])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['rows'], [['A', 'B'], ['1', '2'], ['3', '4'], ['5', '6']])

    def test_拆分页重复表头时自动去重(self):
        t1 = make_table(['A', 'B'], [['1', '2']], page=1)
        t2 = make_table(['A', 'B'], [['3', '4']], page=2)   # 跨页部分重复了表头
        out = self.tp.merge_tables([t1, t2])
        self.assertEqual(out[0]['rows'], [['A', 'B'], ['1', '2'], ['3', '4']])

    def test_页码不连续不合并(self):
        t1 = make_table(['A', 'B'], [['1', '2']], page=1)
        t2 = make_table(['A', 'B'], [['3', '4']], page=3)
        self.assertEqual(len(self.tp.merge_tables([t1, t2])), 2)

    def test_列数不同不合并(self):
        t1 = make_table(['A', 'B'], [['1', '2']], page=1)
        t2 = make_table(['A', 'B', 'C'], [['3', '4', '5']], page=2)
        self.assertEqual(len(self.tp.merge_tables([t1, t2])), 2)

    def test_页码缺失时防御性不合并(self):
        # page 不是整数（如 None）时不能崩溃，也不合并
        t1 = make_table(['A', 'B'], [['1', '2']], page=1)
        t2 = make_table(['A', 'B'], [['3', '4']], page=2)
        del t2['page']
        self.assertEqual(len(self.tp.merge_tables([t1, t2])), 2)

    def test_单张表原样返回(self):
        t = make_table(['A'], [['1']])
        out = self.tp.merge_tables([t])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['rows'], [['A'], ['1']])

    def test_空入参返回空列表(self):
        self.assertEqual(self.tp.merge_tables([]), [])

    def test_rows缺失时退回用markdown解析(self):
        t1 = {'markdown': '| A | B |\n|---|---|\n| 1 | 2 |', 'rows': [], 'page': 1, 'cols': 2}
        t2 = {'markdown': '|A|B|\n|---|---|\n|3|4|', 'rows': [], 'page': 2, 'cols': 2}
        out = self.tp.merge_tables([t1, t2])
        self.assertEqual(out[0]['rows'], [['A', 'B'], ['1', '2'], ['3', '4']])
        self.assertIn('| 3 | 4 |', out[0]['markdown'])

    def test_合并后页码保留起始页(self):
        t1 = make_table(['A'], [['1']], page=1)
        t2 = make_table(['A'], [['2']], page=2)
        out = self.tp.merge_tables([t1, t2])
        self.assertEqual(out[0]['page'], 1)   # 起始页保持不变

    def test_不修改调用方的原始数据(self):
        t1 = make_table(['A', 'B'], [['1', '2']], page=1)
        t2 = make_table(['A', 'B'], [['3', '4']], page=2)
        self.tp.merge_tables([t1, t2])
        # 合并过程只操作内部拷贝，入参的 rows 不会被追加
        self.assertEqual(t1['rows'], [['A', 'B'], ['1', '2']])


# ======================================================================
# 二、其余方法：markdown 字符串进 / 出
# ======================================================================
class TestOtherMethods(unittest.TestCase):
    """table_to_dataframe / dataframe_to_text / table_to_text / extract_key_value。"""

    def setUp(self):
        self.tp = TableProcessor()
        self.md = '| 姓名 | 年龄 |\n|---|---|\n| 张三 | 28 |\n| 李四 | 35 |'

    def test_table_to_dataframe(self):
        df = self.tp.table_to_dataframe(self.md)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(list(df.columns), ['姓名', '年龄'])
        self.assertEqual(df.iloc[0, 0], '张三')

    def test_table_to_dataframe_空输入返回空表(self):
        self.assertEqual(self.tp.table_to_dataframe('').shape, (0, 0))

    def test_table_to_dataframe_缺列自动补空串(self):
        df = self.tp.table_to_dataframe('| A | B | C |\n|---|---|---|\n| 1 | 2 |')
        self.assertEqual(df.iloc[0].tolist(), ['1', '2', ''])

    def test_dataframe_to_text(self):
        text = self.tp.dataframe_to_text(self.tp.table_to_dataframe(self.md))
        self.assertEqual(text.splitlines()[0], '| 姓名 | 年龄 |')
        self.assertIn('| 张三 | 28 |', text)

    def test_table_to_text_规范化输出(self):
        text = self.tp.table_to_text(self.md)
        # 输出固定为：表头 + 分隔行 + 2 行数据
        self.assertEqual(len(text.splitlines()), 4)
        self.assertEqual(text.splitlines()[1], '| --- | --- |')

    def test_extract_key_value(self):
        kv = self.tp.extract_key_value(
            '| 键 | 值 |\n|---|---|\n| 名字 | 张三 |\n| 城市 | 北京 |'
        )
        # 表头行也会被当作一行键值对解析（逐行解析的设计）
        self.assertEqual(kv, {'键': '值', '名字': '张三', '城市': '北京'})


if __name__ == "__main__":
    unittest.main(verbosity=2)
