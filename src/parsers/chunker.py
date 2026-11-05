# -*- coding: utf-8 -*-
"""
chunker.py —— 文档分块模块。

分块时文本和表格分开传入、分开分块：

- ChunkStrategy：分块策略配置类，集中保存分块参数（块大小、重叠、
  表格块上限等）并做合法性校验；
- ChunkText：主分块器，持有一份 ChunkStrategy 配置：
    chunk(text, tables) 主入口
      ├─ 文本 → 文本块（type='text'）：按“段落 → 句子 → 定长”逐级切分，
      │         相邻块之间支持重叠（chunk_overlap）；
      └─ 表格 → 表格块（type='table'）：每张表单独一块，不与文本块混在一起，
                超长表格按数据行拆分，拆分后每块都补上完整表头；
- 默认把文本里的 markdown 表格整块剔除（strip_tables=True）：
  表格已经单独成块，再留在文本块里会造成同一份内容出现两次。

chunk 字典字段（与 base.ParseResult.chunks 的约定一致，可直接回填）::

    {
        'chunk_id': int,         # 全局序号，chunk() 汇总后从 0 连续递增
        'type': 'text'|'table',  # 块类型，检索时可据此区分处理
        'content': str,          # 块正文；表格块为 markdown 表格文本
        'page': int | None,      # 页码：表格块取表格所在页；文本块无法定位，为 None
        'metadata': {...},       # file_name / type_index / char_count + 表格专属信息
    }
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.parsers.table_parser import TableProcessor

logger = logging.getLogger(__name__)

__all__ = ["ChunkStrategy", "ChunkText"]


class ChunkStrategy:
    r"""
    分块策略配置类：集中保存分块参数，构造时统一做合法性校验。

    典型用法::

        strategy = ChunkStrategy(chunk_size=500, chunk_overlap=50)
        chunker = ChunkText(strategy)

        # 参数全部合法时也可以用默认值：chunk_size=500, chunk_overlap=50
        chunker = ChunkText()          # 内部自动使用 ChunkStrategy() 默认配置

    设计约定：
    - chunk_size 按字符数计（中文 1 个字就是 1 个字符，比按词切更直观可控）；
    - 带重叠前缀的文本块允许略微超过 chunk_size（最多超出 chunk_overlap
      加段落分隔符），否则等长片段会被严格上限挤掉重叠，重叠无法保证生效；
    - max_table_chars 是单个表格块的字符上限，默认与 chunk_size 相同；
      表格信息密度高，需要更大预算时显式传大一些。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 20,
        strip_tables: bool = True,
        max_table_chars: Optional[int] = None,
    ):
        """
        构造函数。

        :param chunk_size:      单个文本块的最大字符数，必须为正整数。
        :param chunk_overlap:   相邻文本块之间的重叠字符数，
                                必须满足 0 <= chunk_overlap < chunk_size。
        :param min_chunk_size:  尾部文本块短于该值时并入前一块，避免碎块。
        :param strip_tables:    是否从文本中剔除 markdown 表格（表格单独成块时
                                建议开启，默认开启；设为 False 则表格文本保留在文本块里）。
        :param max_table_chars: 单个表格块的最大字符数，默认与 chunk_size 相同；
                                超过该值的表格按数据行拆分为多个块。
        :raises ValueError:     参数不满足上述约束时抛出。
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size 不能为负数")
        if max_table_chars is not None and max_table_chars <= 0:
            raise ValueError("max_table_chars 必须是正整数")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strip_tables = strip_tables
        self.max_table_chars = max_table_chars if max_table_chars is not None else chunk_size


class ChunkText:
    r"""
    主分块器：文本、表格分开传入，分开分块。

    典型用法::

        parser = PdfParser()
        result = parser.parse('demo.pdf')   # 解析：得到 raw_text 和 tables

        chunker = ChunkText(ChunkStrategy(chunk_size=500, chunk_overlap=50))

        # 文本和表格分开传入（表格单独成块），文本块在前、表格块在后
        chunks = chunker.chunk(result.raw_text, result.tables,
                               file_name=result.file_name)

        # 文本、表格也可以各自单独分块
        text_chunks = chunker.chunk_text(result.raw_text)
        table_chunks = chunker.chunk_tables(result.tables)

    设计约定：
    - 传入的文本应是已清洗的文本（先用 TextClearn.clearn 处理）；
    - 相邻文本块之间保留 chunk_overlap 字符的重叠，减少句子在块边界被
      切断造成的语义丢失；表格块不参与重叠；
    - 尾部短于 min_chunk_size 的文本块并入前一块，避免产生碎块。
    """

    # 匹配“整块 markdown 表格”：连续若干行，行首可有空白、以 | 开头且以 | 结尾。
    # 仅用于从文本中剔除表格（表格已单独成块）。要求是“整块”的竖线行，
    # 正文里以 | 开头且以 | 结尾的整行几乎只会出现在表格中，宁可多剔不漏剔。
    # 注意：只识别 markdown 竖线表格（docling 默认导出格式），不含 HTML 表格。
    _TABLE_BLOCK_RE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*$\n?)+", re.MULTILINE)

    # 句子边界：在句末标点（中英文句号、问叹号、分号）和换行之后断开。
    # 用 lookbehind 零宽断言，标点保留在前一句末尾、不消耗字符。
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;\n])")

    def __init__(self, strategy: Optional[ChunkStrategy] = None):
        """
        构造函数。

        :param strategy: 分块策略配置；不传时使用 ChunkStrategy() 默认参数。
        """
        # 配置对象保存在公开属性上，方便外部运行时查看/调整参数
        self.strategy = strategy if strategy is not None else ChunkStrategy()
        # 复用 TableProcessor 做表格 markdown 的规范化与行列解析
        self._table_processor = TableProcessor()

    # ==================================================================
    # 一、分块入口
    # ==================================================================
    def chunk(
        self,
        text: Optional[str],
        tables: Optional[List[Dict[str, Any]]] = None,
        file_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        统一分块入口：文本和表格分开传入，分别分块后合并返回。

        :param text:      文本内容（建议已经 TextClearn 清洗）。
        :param tables:    表格列表，元素为解析器输出的表格字典
                          （{'markdown','rows','page','cols'}，markdown 与 rows 至少有其一）。
        :param file_name: 文件名，写入每个块的 metadata 便于溯源。
        :return:          文本块在前、表格块在后的块列表，
                          chunk_id 为从 0 开始的连续全局序号。
        """
        chunks = self.chunk_text(text, file_name=file_name)
        chunks += self.chunk_tables(tables or [], file_name=file_name)
        # 汇总后统一重排全局序号，保证下游拿到的 chunk_id 连续且唯一
        for i, c in enumerate(chunks):
            c["chunk_id"] = i
        return chunks

    # ==================================================================
    # 二、文本分块
    # ==================================================================
    def chunk_text(self, text: Optional[str], file_name: str = "") -> List[Dict[str, Any]]:
        """
        文本分块：按“段落 → 句子 → 定长”逐级切分，相邻块保留重叠。

        :param text:      文本内容；传 None 或空白串返回空列表。
        :param file_name: 文件名，写入 metadata。
        :return:          文本块列表（type='text'，page 为 None——原始 markdown
                          不带页码信息，文本块无法定位页码）。
        """
        text = "" if text is None else str(text)
        # 统一换行符，后面按 \n 处理（与 cleaner.normalize_whitespace 的做法一致）
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self.strategy.strip_tables:
            # 表格已单独成块，从文本里整块剔除，避免同一张表出现两份内容
            text = self._TABLE_BLOCK_RE.sub("", text)
        text = text.strip()
        if not text:
            return []

        # 切分单元记为 (单元文本, 与前一单元的连接符)：
        # - 超长段落切出的连续片段之间用 '' 连接（它们本来就是连续的文字）；
        # - 其他情况（段落边界）用 '\n\n' 连接。
        units: List[Tuple[str, str]] = []
        for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
            if not para:
                continue
            for i, piece in enumerate(self._split_long_text(para)):
                units.append((piece, "" if i > 0 else "\n\n"))

        # 贪心装填成块，块间生成重叠
        packed = self._pack_units(units)

        # 尾部碎块并入前一块（最后一块才可能 < min_chunk_size，前块允许超一点）
        merged: List[str] = []
        for c in packed:
            if merged and len(c) < self.strategy.min_chunk_size:
                merged[-1] = merged[-1] + "\n\n" + c
            else:
                merged.append(c)

        return [
            {
                "chunk_id": i,
                "type": "text",
                "content": c,
                "page": None,
                "metadata": {
                    "file_name": file_name,
                    "type_index": i,
                    "char_count": len(c),
                },
            }
            for i, c in enumerate(merged)
        ]

    def _split_long_text(self, text: str) -> List[str]:
        """
        内部工具：把超过 chunk_size 的段落切成若干 ≤ chunk_size 的片段。

        先按句子边界聚合；单句仍超长时再按“定长 + 重叠”硬切。
        不超过 chunk_size 的段落原样返回单元素列表。
        """
        if len(text) <= self.strategy.chunk_size:
            return [text]
        pieces: List[str] = []
        buf = ""
        for sent in self._SENTENCE_SPLIT_RE.split(text):
            if not sent:
                continue
            if len(sent) > self.strategy.chunk_size:
                # 单句超长：先落袋 buf，再对整句定长硬切（相邻片段重叠 chunk_overlap）。
                # step = chunk_size - chunk_overlap >= 1（配置校验已保证），不会死循环
                if buf:
                    pieces.append(buf)
                    buf = ""
                step = self.strategy.chunk_size - self.strategy.chunk_overlap
                start = 0
                while start < len(sent):
                    pieces.append(sent[start:start + self.strategy.chunk_size])
                    if start + self.strategy.chunk_size >= len(sent):
                        break
                    start += step
                continue
            if len(buf) + len(sent) <= self.strategy.chunk_size:
                buf += sent
            else:
                pieces.append(buf)
                buf = sent
        if buf:
            pieces.append(buf)
        return pieces

    def _pack_units(self, units: List[Tuple[str, str]]) -> List[str]:
        """
        内部工具：把切分单元贪心装填进块；块满时封块，并用上一块尾部内容
        生成下一块的重叠前缀。

        前缀 + 新单元允许略微超过 chunk_size（最多 chunk_size + chunk_overlap
        加段落分隔符），保证 chunk_overlap 一定生效；否则等长片段会被
        严格上限挤掉重叠。
        """
        packed: List[List[Tuple[str, str]]] = []   # 每个块 = 单元列表
        cur: List[Tuple[str, str]] = []
        cur_len = 0
        for unit in units:
            text, jb = unit
            add = len(text) + (len(jb) if cur else 0)
            if cur and cur_len + add > self.strategy.chunk_size:
                # 当前块已满：封块，用尾部内容开启带重叠前缀的新块
                packed.append(cur)
                cur = self._overlap_prefix(cur)
                cur_len = self._units_len(cur)
                add = len(text) + (len(jb) if cur else 0)
            cur_len += add
            cur.append(unit)
        if cur:
            packed.append(cur)
        return [self._units_to_text(u) for u in packed]

    def _overlap_prefix(self, units: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        内部工具：取上一块末尾内容作为下一块的重叠前缀。

        优先按整单元回退（不切断句子）；若最后一个单元本身超过 chunk_overlap，
        则对该单元取尾部 chunk_overlap 个字符硬截断。
        """
        if self.strategy.chunk_overlap <= 0:
            return []
        prefix: List[Tuple[str, str]] = []
        total = 0
        for text, jb in reversed(units):
            if total + len(text) > self.strategy.chunk_overlap:
                break
            prefix.insert(0, (text, jb))
            total += len(text)
        if not prefix:
            # 最后一个单元比 overlap 还长：取其尾部若干字符硬截断
            last_text = units[-1][0]
            prefix = [(last_text[-self.strategy.chunk_overlap:], "")]
        return prefix

    @staticmethod
    def _units_len(units: List[Tuple[str, str]]) -> int:
        """内部工具：按 _units_to_text 的拼接规则计算单元列表的总字符数。"""
        if not units:
            return 0
        return sum(len(t) for t, _ in units) + sum(len(jb) for _, jb in units[1:])

    @staticmethod
    def _units_to_text(units: List[Tuple[str, str]]) -> str:
        """内部工具：把单元列表拼成块文本。首单元的连接符丢弃，其余按各自连接符拼接。"""
        parts = []
        for i, (text, jb) in enumerate(units):
            parts.append(text if i == 0 else jb + text)
        return "".join(parts)

    # ==================================================================
    # 三、表格分块
    # ==================================================================
    def chunk_tables(
        self,
        tables: Optional[List[Dict[str, Any]]],
        file_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        表格分块：每张表单独成块（type='table'），不与文本块混在一起。

        - 表格 markdown 先用 TableProcessor 规范化格式（统一分隔行、列对齐）；
        - 不超过 max_table_chars 的表格整表一块；
        - 超长的表格按数据行拆成多块，每块都带完整表头，
          metadata 里用 part / total_parts 标记拆分位置。

        :param tables:    表格列表，元素为解析器输出的表格字典
                          （{'markdown','rows','page','cols'}，markdown 与 rows 至少有其一）。
        :param file_name: 文件名，写入 metadata。
        :return:          表格块列表（chunk_id 为本列表内的序号，由 chunk() 汇总时统一重排）。
        """
        chunks: List[Dict[str, Any]] = []
        for t_idx, table in enumerate(tables or []):
            if not isinstance(table, dict):
                logger.warning("第 %d 个表格不是 dict，已跳过：%r", t_idx, table)
                continue

            markdown = str(table.get("markdown") or "").strip()
            rows = table.get("rows")
            # 有 markdown 先规范化（顺带统一格式）；失败则保留原文，不影响分块流程
            if markdown:
                try:
                    normalized = self._table_processor.table_to_text(markdown)
                    if normalized.strip():
                        markdown = normalized
                except Exception as exc:
                    logger.warning("表格 %d 规范化失败，保留原始 markdown：%s", t_idx, exc)
            # markdown 缺失/为空时用 rows 重建
            if not markdown.strip() and rows:
                markdown = self._rows_to_markdown(rows)
            if not markdown.strip():
                logger.warning("第 %d 个表格内容为空，已跳过", t_idx)
                continue

            page = table.get("page")
            if len(markdown) <= self.strategy.max_table_chars:
                # 整表一块
                chunks.append(
                    self._make_table_chunk(
                        markdown, page=page, file_name=file_name,
                        type_index=len(chunks), table_index=t_idx,
                        rows=rows, part=None, total_parts=None,
                    )
                )
            else:
                # 超长表格按数据行拆分，每块带表头
                pieces = self._split_table_rows(markdown, rows)
                total = len(pieces)
                for part, piece in enumerate(pieces, start=1):
                    chunks.append(
                        self._make_table_chunk(
                            piece, page=page, file_name=file_name,
                            type_index=len(chunks), table_index=t_idx,
                            rows=rows, part=part, total_parts=total,
                        )
                    )
        return chunks

    def _split_table_rows(self, markdown: str, rows: Any) -> List[str]:
        """
        内部工具：把超长表格按数据行切成若干 ≤ max_table_chars 的 markdown 片段，
        每片都带完整表头（保证单独一块也能看懂各列含义）。

        单个数据行本身超预算时无法再细拆，仍独占一片（该片会超上限，属预期）。
        """
        # rows 缺失时退回从 markdown 解析（与 TableProcessor.merge_tables 的兜底一致）
        if not rows:
            rows = self._table_processor._split_markdown_table(markdown)
        rows = [[str(c) for c in r] for r in (rows or [])]
        if len(rows) < 2:
            # 只有表头（或为空）时无法按数据行拆分，整表返回
            return [markdown]

        header = rows[0]
        data_rows = rows[1:]
        header_line = "| " + " | ".join(header) + " |"
        sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
        header_block = header_line + "\n" + sep_line
        # 每片的行内容预算 = 上限 - 表头占用 - 与表头间的换行符；
        # 表头本身就超预算时至少给每行留 1 字符空间，避免死循环
        budget = max(self.strategy.max_table_chars - len(header_block) - 1, 1)

        pieces: List[str] = []
        cur_lines: List[str] = []
        cur_len = 0   # 已装行内容的长度（每行额外 +1 计入行间换行符）
        for row in data_rows:
            line = "| " + " | ".join(str(c) for c in row) + " |"
            # 已有内容时放不下就先封片；首行即使超预算也照样装入（无法再细拆）
            if cur_lines and cur_len + len(line) > budget:
                pieces.append(header_block + "\n" + "\n".join(cur_lines))
                cur_lines, cur_len = [], 0
            cur_lines.append(line)
            cur_len += len(line) + 1
        if cur_lines:
            pieces.append(header_block + "\n" + "\n".join(cur_lines))
        return pieces

    @staticmethod
    def _rows_to_markdown(rows: Any) -> str:
        """
        内部工具：把行列二维列表拼成 markdown 表格
        （格式与 TableProcessor.dataframe_to_text 一致：第一行当表头）。
        """
        rows = [[str(c) for c in r] for r in (rows or [])]
        if not rows:
            return ""
        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * len(rows[0])) + " |",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)

    def _make_table_chunk(
        self,
        content: str,
        *,
        page: Any,
        file_name: str,
        type_index: int,
        table_index: int,
        rows: Any,
        part: Optional[int],
        total_parts: Optional[int],
    ) -> Dict[str, Any]:
        """内部工具：组装单个表格块字典。rows 含表头行，row_count 统计的是数据行数。"""
        metadata: Dict[str, Any] = {
            "file_name": file_name,
            "type_index": type_index,
            "char_count": len(content),
            "table_index": table_index,
            "cols": len(rows[0]) if rows else None,
            "row_count": max(len(rows) - 1, 0) if rows else None,
        }
        if part is not None:
            metadata["part"] = part
            metadata["total_parts"] = total_parts
        return {
            "chunk_id": type_index,
            "type": "table",
            "content": content,
            "page": page,
            "metadata": metadata,
        }
