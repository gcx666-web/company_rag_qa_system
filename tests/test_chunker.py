# -*- coding: utf-8 -*-
"""ChunkStrategy / ChunkText 分块模块的单元测试。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_chunker -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_chunker.py

说明：只测分块模块本身（不加载 docling/PdfParser，速度快、无网络依赖），
用标准库 unittest 编写，无需额外安装 pytest。
"""

import os
import sys
import unittest

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.chunker import ChunkStrategy, ChunkText


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
# 一、ChunkStrategy：参数校验
# ======================================================================
class TestChunkStrategy(unittest.TestCase):
    def test_合法参数正常保存(self):
        s = ChunkStrategy(chunk_size=300, chunk_overlap=30, min_chunk_size=10,
                          strip_tables=False, max_table_chars=1000)
        self.assertEqual((s.chunk_size, s.chunk_overlap, s.min_chunk_size), (300, 30, 10))
        self.assertFalse(s.strip_tables)
        self.assertEqual(s.max_table_chars, 1000)

    def test_max_table_chars默认跟随chunk_size(self):
        self.assertEqual(ChunkStrategy(chunk_size=123).max_table_chars, 123)

    def test_重叠不小于块大小时报错(self):
        with self.assertRaises(ValueError):
            ChunkStrategy(chunk_size=50, chunk_overlap=50)
        with self.assertRaises(ValueError):
            ChunkStrategy(chunk_size=50, chunk_overlap=-1)

    def test_非法chunk_size报错(self):
        with self.assertRaises(ValueError):
            ChunkStrategy(chunk_size=0)
        with self.assertRaises(ValueError):
            ChunkStrategy(chunk_size=-10)

    def test_非法max_table_chars报错(self):
        with self.assertRaises(ValueError):
            ChunkStrategy(max_table_chars=0)


# ======================================================================
# 二、ChunkText.chunk_text：文本分块（段落→句子→定长，带重叠）
# ======================================================================
class TestChunkTextMethod(unittest.TestCase):
    """文本分块规则：不超上限、内容不丢、相邻块有重叠。"""

    def setUp(self):
        self.ck = ChunkText(ChunkStrategy(chunk_size=50, chunk_overlap=10, min_chunk_size=0))

    def test_短文本只生成一块(self):
        chunks = self.ck.chunk_text('只有一句话。')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['type'], 'text')
        self.assertEqual(chunks[0]['content'], '只有一句话。')
        self.assertIsNone(chunks[0]['page'])

    def test_空输入返回空列表(self):
        self.assertEqual(self.ck.chunk_text(''), [])
        self.assertEqual(self.ck.chunk_text(None), [])
        self.assertEqual(self.ck.chunk_text('   \n\n  '), [])

    def test_长文本按chunk_size分块且内容不丢(self):
        text = '。'.join(f'这是第{i}句话内容' for i in range(40)) + '。'
        chunks = self.ck.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            # 允许带重叠前缀略微超出（最多 chunk_overlap 加段落分隔符 2 字符）
            self.assertLessEqual(len(c['content']), 50 + 10 + 2)
        # 所有句子都能在某个块里找到，说明切分没有丢内容
        joined = '\n'.join(c['content'] for c in chunks)
        for i in (0, 10, 20, 39):
            self.assertIn(f'第{i}句话', joined)

    def test_相邻块有重叠(self):
        text = ''.join(f'段落{i}内容占若干字。' for i in range(20))
        chunks = self.ck.chunk_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        # 第二块的开头（重叠前缀）应出现在第一块的内容里
        head = chunks[1]['content'][:5]
        self.assertIn(head, chunks[0]['content'])

    def test_文本中的markdown表格默认被剔除(self):
        text = ('前面一段说明文字。\n\n'
                '| A | B |\n|---|---|\n| 1 | 2 |\n\n'
                '后面一段总结文字。')
        chunks = self.ck.chunk_text(text)
        joined = '\n'.join(c['content'] for c in chunks)
        # 表格已单独成块，文本块里不应再出现竖线表格
        self.assertNotIn('|', joined)
        self.assertIn('说明文字', joined)
        self.assertIn('总结文字', joined)

    def test_strip_tables为False时表格保留在文本块里(self):
        ck = ChunkText(ChunkStrategy(chunk_size=200, chunk_overlap=10, strip_tables=False))
        text = '说明文字。\n\n| A | B |\n|---|---|\n| 1 | 2 |'
        chunks = ck.chunk_text(text)
        self.assertIn('|', chunks[0]['content'])

    def test_尾部碎块并入前一块(self):
        # min_chunk_size=60：最后不足 60 字的块并入前一块
        ck = ChunkText(ChunkStrategy(chunk_size=50, chunk_overlap=0, min_chunk_size=60))
        text = '。'.join(f'句子{i}' for i in range(20)) + '。'
        chunks = ck.chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertGreaterEqual(len(chunks[0]['content']), 60)


# ======================================================================
# 三、ChunkText.chunk_tables：表格单独分块
# ======================================================================
class TestChunkTablesMethod(unittest.TestCase):
    """表格分块规则：每表一块、保留页码、超长按行拆分且每块带表头。"""

    def setUp(self):
        self.ck = ChunkText(ChunkStrategy(chunk_size=200, chunk_overlap=20))

    def test_每张表单独成块且保留页码(self):
        tables = [
            make_table(['A', 'B'], [['1', '2'], ['3', '4']], page=1),
            make_table(['C'], [['x']], page=5),
        ]
        chunks = self.ck.chunk_tables(tables, file_name='demo.pdf')
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(c['type'] == 'table' for c in chunks))
        self.assertEqual([c['page'] for c in chunks], [1, 5])
        self.assertIn('| 1 |', chunks[0]['content'])
        self.assertIn('| 3 |', chunks[0]['content'])
        self.assertNotIn('| x |', chunks[0]['content'])
        self.assertEqual(chunks[0]['metadata']['table_index'], 0)
        self.assertEqual(chunks[1]['metadata']['table_index'], 1)
        self.assertEqual(chunks[0]['metadata']['file_name'], 'demo.pdf')
        self.assertEqual(chunks[0]['metadata']['row_count'], 2)
        self.assertEqual(chunks[0]['metadata']['cols'], 2)

    def test_只有rows没有markdown也能成块(self):
        tables = [{'rows': [['A', 'B'], ['1', '2']], 'page': 2, 'cols': 2}]
        chunks = self.ck.chunk_tables(tables)
        self.assertEqual(len(chunks), 1)
        self.assertIn('| A | B |', chunks[0]['content'])
        self.assertIn('| 1 | 2 |', chunks[0]['content'])

    def test_空表格列表返回空(self):
        self.assertEqual(self.ck.chunk_tables([]), [])
        self.assertEqual(self.ck.chunk_tables(None), [])

    def test_超长表格按行拆分且每块带表头(self):
        rows = [[f'员工{i}', str(i)] for i in range(60)]
        tables = [make_table(['姓名', '年龄'], rows, page=3)]
        ck = ChunkText(ChunkStrategy(chunk_size=100, chunk_overlap=0))  # max_table_chars 默认 = chunk_size
        chunks = ck.chunk_tables(tables)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn('| 姓名 | 年龄 |', c['content'])   # 每块都带表头
            self.assertLessEqual(len(c['content']), 100)     # 每块不超过上限
        # 数据行不丢
        joined = '\n'.join(c['content'] for c in chunks)
        for i in (0, 29, 59):
            self.assertIn(f'员工{i}', joined)
        # part / total_parts 标记拆分位置
        self.assertEqual(chunks[0]['metadata']['part'], 1)
        self.assertEqual(chunks[-1]['metadata']['part'], len(chunks))
        self.assertEqual(chunks[0]['metadata']['total_parts'], len(chunks))

    def test_max_table_chars可单独放大(self):
        rows = [[f'员工{i}', str(i)] for i in range(60)]
        tables = [make_table(['姓名', '年龄'], rows, page=1)]
        # 文本块很小，但给表格更大的预算 → 整表一块
        ck = ChunkText(ChunkStrategy(chunk_size=30, chunk_overlap=5, max_table_chars=2000))
        text_chunks = ck.chunk_text('。'.join(f'句子{i}' for i in range(50)))
        table_chunks = ck.chunk_tables(tables)
        self.assertGreater(len(text_chunks), 1)
        self.assertEqual(len(table_chunks), 1)


# ======================================================================
# 四、ChunkText.chunk：统一入口（文本、表格分开传入）
# ======================================================================
class TestChunkEntry(unittest.TestCase):
    def test_不传策略时使用默认配置(self):
        ck = ChunkText()
        self.assertEqual(ck.strategy.chunk_size, 500)
        self.assertEqual(ck.strategy.chunk_overlap, 50)

    def test_文本与表格分开分块并统一编号(self):
        ck = ChunkText(ChunkStrategy(chunk_size=50, chunk_overlap=5))
        tables = [make_table(['指标', '数值'], [['营收', '100万']], page=2)]
        text = '。'.join(f'公司第{i}季度业绩说明' for i in range(30)) + '。'
        chunks = ck.chunk(text, tables, file_name='report.pdf')
        # 编号连续
        self.assertEqual([c['chunk_id'] for c in chunks], list(range(len(chunks))))
        # 文本块在前、表格块在后
        self.assertEqual(chunks[0]['type'], 'text')
        self.assertEqual(chunks[-1]['type'], 'table')
        self.assertEqual(chunks[-1]['page'], 2)
        # 同一张表不再出现在文本块里
        self.assertFalse(any('|' in c['content'] for c in chunks if c['type'] == 'text'))
        self.assertTrue(all(c['metadata']['file_name'] == 'report.pdf' for c in chunks))

    def test_空输入返回空列表(self):
        ck = ChunkText()
        self.assertEqual(ck.chunk('', None), [])
        self.assertEqual(ck.chunk(None, []), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
