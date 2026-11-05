import logging
from typing import Any, Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from io import BytesIO
from pathlib import Path

from src.parsers.base import BaseParser, ParseResult
from src.parsers.chunker import ChunkText
from src.parsers.cleaner import TextClearn
from src.parsers.table_parser import TableProcessor

logger = logging.getLogger(__name__)


class WordParser(BaseParser):
    '''Word文档解析器'''


    def __init__(
        self,
        chunker: Optional[ChunkText] = None,
        cleaner: Optional[TextClearn] = None,
        table_processor: Optional[TableProcessor] = None,
    ):
        # 初始化组件
        self.chunker = chunker if chunker else ChunkText()
        self.cleaner = cleaner if cleaner else TextClearn()
        self.table_processor = (
            table_processor if table_processor else TableProcessor()
        )

        # 初始化Docling 转换器选项
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        self.converter = DocumentConverter()
    def support_file_types(self) -> list:
        #docling 只支持 .docx，不支持老版二进制 .doc 格式
        return ['.docx','.DOCX']

    def parse(self, file_path: str) -> ParseResult:
        '''
        从文件路径解析Word文档
        '''
        #使用docling 解析文档
        doc=self.converter.convert(file_path)

        #提取内容
        raw_text=doc.document.export_to_markdown()

        #提取表格
        tables=self._extract_tables(doc)

        #docx 没有真实分页概念，docling 的 pages 常为空，兜底为 1
        total_pages=len(doc.document.pages) or 1

        return ParseResult(
            file_name=Path(file_path).name,
            total_pages=total_pages,
            total_chunks=len(raw_text),
            table_count=len(tables),
            chunks=[],
            raw_text=raw_text,
            tables=tables,
            metadata={
                'parser':'docling','file_path':file_path
            }
        )




    def parse_from_bytes(self, file_bytes: bytes, file_name: str) -> ParseResult:
        # 从字节流解析Word文档

        #使用docling 转换成document对象，name 必须带真实扩展名以便 docling 识别格式
        bio = BytesIO(file_bytes)
        source = DocumentStream(name=file_name, stream=bio)
        doc = self.converter.convert(source)
        #提取内容
        raw_text=doc.document.export_to_markdown()

         #提取表格
        tables=self._extract_tables(doc)

        #docx 没有真实分页概念，docling 的 pages 常为空，兜底为 1
        total_pages=len(doc.document.pages) or 1

        return ParseResult(
                    file_name=file_name,
                    total_pages=total_pages,
                    total_chunks=len(raw_text),
                    table_count=len(tables),
                    chunks=[],
                    raw_text=raw_text,
                    tables=tables,
                    metadata={
                        'parser':'docling','file_path':file_name
                    }
            )


    def _extract_tables(self, doc) -> List[Dict[str, Any]]:
        '''
        从docling文档对象中提取表格

        :param doc: docling 转换后的文档对象
        :return: 表格列表
        '''
        # 初始化用于存放所有提取成功表格的空列表
        tables: List[Dict[str, Any]] = []

        # 开始异常处理块，避免在获取 document/tables 时因对象缺失导致整个解析中断
        try:
            # 安全获取 doc 对象的 document 属性，若不存在则返回 None 而不抛异常
            document = getattr(doc, 'document', None)
            # 判断 document 是否为空，为空则无法继续提取表格
            if document is None:
                # 记录警告日志，提示 document 属性缺失
                logger.warning('doc 对象不存在 document 属性，无法提取表格')
                # 返回空列表，提前结束方法执行
                return tables
            # 安全获取文档的表格集合，若属性缺失或为 None 则默认空列表
            raw_tables = getattr(document, 'tables', None) or []
        # 捕获获取表格列表时发生的任何异常
        except Exception as exc:
            # 记录获取表格列表失败的异常信息
            logger.warning('获取文档表格列表时发生异常: %s', exc)
            # 异常时返回已初始化的空列表
            return tables

        # 遍历文档中提取到的每个表格对象
        for table in raw_tables:
            # 为每个表格单独开启异常处理，单个表格失败不影响其余表格提取
            try:
                # docling 的 TableItem 没有 .text / .rows 属性（实测 hasattr 均为 False），
                # 表格文本要用 export_to_markdown 导出；传 doc 参数避免弃用警告
                table_text = str(table.export_to_markdown(doc=document) or '')

                # 结构化行列数据用 export_to_dataframe 导出。注意：docling 会把
                # markdown 表格的首行放入 DataFrame 列名（而非数据行），因此先取
                # 列名作为表头行，再拼接数据行，保证 rows 首行是表头（与 chunker
                # 约定一致：rows[0] 为表头）
                df = table.export_to_dataframe(doc=document)
                table_rows = [[str(c) for c in df.columns]] + [
                    [str(v) for v in row] for row in df.values.tolist()
                ]

                # 表格所在页码：从 provenance（溯源信息）里取，取不到则给 None
                # （docx 没有真实分页概念，页码通常为 None，不影响分块流程）
                prov = getattr(table, 'prov', None) or []
                page_no = getattr(prov[0], 'page_no', None) if prov else None

                # 将提取完成的完整表格字典追加到结果列表
                tables.append({
                    # 表格的 markdown 文本
                    'markdown': table_text,
                    # 表格的逐行单元格数据（含表头行）
                    'rows': table_rows,
                    # 表格所在页码（跨页合并判断用）
                    'page': page_no,
                    # 表格列数（跨页合并判断用）
                    'cols': len(table_rows[0]) if table_rows else 0,
                })
            # 捕获单个表格提取过程中的任何异常
            except Exception as exc:
                # 记录提取失败并跳过当前表格的警告日志
                logger.warning('提取单个表格失败，已跳过: %s', exc)
                # 跳过当前出错表格，继续处理下一个表格
                continue

        # 返回所有成功提取的表格列表
        return tables
