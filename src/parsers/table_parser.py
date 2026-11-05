import pandas as pd
from typing import Dict, List

class TableProcessor:
    '''
    表格处理类
    '''


    # ---------------- 内部工具 ----------------
    @staticmethod
    def _split_markdown_table(table_markdown: str) -> List[List[str]]:
        '''
        把 markdown 表格文本解析成二维列表（内部工具方法）
        - 自动去掉行首、行尾的竖线；
        - 跳过 |---|---| 这类表头分隔行；
        - 空行忽略。
        '''
        rows: List[List[str]] = []
        for line in table_markdown.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('|'):        # 去掉行首竖线
                line = line[1:]
            if line.endswith('|'):          # 去掉行尾竖线
                line = line[:-1]
            cells = [c.strip() for c in line.split('|')]
            # 分隔行（|---|---|）的特征：每个单元格只含 -、: 和空格
            if cells and all(set(c) <= set('-: ') for c in cells):
                continue
            rows.append(cells)
        return rows


    def table_to_dataframe(self,table_markdown:str) -> pd.DataFrame:
        '''
        将markdown表格转换为DataFrame
        参数:
        table_markdown:str,markdown表格字符串
        返回:
        pd.DataFrame,DataFrame对象
        '''
        rows = self._split_markdown_table(table_markdown)
        if not rows:
            # 空表格直接返回空 DataFrame，避免下面取 rows[0] 越界
            return pd.DataFrame()
        header, data = rows[0], rows[1:]
        # 某行单元格数量和表头不一致时，截断/补空串对齐，防止构造 DataFrame 报错
        width = len(header)
        data = [r[:width] + [''] * (width - len(r)) for r in data]
        return pd.DataFrame(data, columns=header)


    def dataframe_to_text(self,dataframe:pd.DataFrame) -> str:
        '''
        将DataFrame转换为markdown表格
        参数:
        dataframe:pd.DataFrame,DataFrame对象
        返回:
        str,markdown表格字符串
        '''
        # 不用 DataFrame.to_markdown()：它依赖额外的 tabulate 包
        header = [str(c) for c in dataframe.columns]
        lines = [
            '| ' + ' | '.join(header) + ' |',
            '| ' + ' | '.join(['---'] * len(header)) + ' |',
        ]
        for _, row in dataframe.iterrows():
            lines.append('| ' + ' | '.join(str(v) for v in row.tolist()) + ' |')
        return '\n'.join(lines)


    def table_to_text(self,table_markdown:str) -> str:
        '''
        将markdown表格转换为markdown表格字符串
        参数:
        table_markdown:str,markdown表格字符串
        返回:
        str,markdown表格字符串
        '''
        # 先解析成 DataFrame 再统一输出，顺带完成格式规范化
        return self.dataframe_to_text(self.table_to_dataframe(table_markdown))


    def merge_tables(self, tables: List[Dict]) -> List[Dict]:
       
        """
        合并跨页的拆分表格
    
        Args:
        tables: 表格列表
    
        Returns:
        合并后的表格列表
        """
        if len(tables) <= 1:
            return tables

        def _rows_of(table: Dict) -> List[List[str]]:
            """取一张表的行列数据；rows 缺失时退回用 markdown 解析。"""
            return table.get('rows') or self._split_markdown_table(str(table.get('markdown') or ''))

        # 拷贝第一张表（连同 rows 列表），避免合并时改动调用方的原始数据
        merged = [{**tables[0], 'rows': list(_rows_of(tables[0]))}]
        span_end = [tables[0].get('page')]   # 每个合并区间当前延伸到的最后一页（不写回输出，保留起始页）

        for i in range(1, len(tables)):
            current = tables[i]
            last = merged[-1]
            page_cur = current.get('page')
            page_last = span_end[-1]
            # 简单判断：如果列数相同且页码连续，可能是同一张表格
            # （页码缺失或不是整数时视为不连续，不做合并，避免 None+1 报错）
            if (current.get('cols') == last.get('cols')
                    and isinstance(page_cur, int) and isinstance(page_last, int)
                    and page_cur == page_last + 1):
                # 合并表格内容：追加数据行；若拆分页重复了表头行，跳过重复表头
                add_rows = _rows_of(current)
                if add_rows and last['rows'] and add_rows[0] == last['rows'][0]:
                    add_rows = add_rows[1:]
                last['rows'] += add_rows
                # 合并区间延伸到当前页，连续三页及以上的表格也能一直合并下去
                span_end[-1] = page_cur
                # 用合并后的完整行列数据重建 markdown，
                # 保证 markdown 和 rows 一致（而不是两段表格文本上下堆叠）
                if last['rows']:
                    last['markdown'] = self.dataframe_to_text(
                        pd.DataFrame(last['rows'][1:], columns=last['rows'][0])
                    )
            else:
                merged.append(current)
                span_end.append(current.get('page'))
        return merged


    def extract_key_value(self, table_markdown: str) -> Dict[str, str]:
        '''
        从markdown表格中提取键值对
        参数:
        table_markdown:str,markdown表格字符串
        返回:
        dict,键值对字典
        '''
        # 每一行按“第1列=键、第2列=值”解析；少于 2 列或键为空的行跳过
        result: Dict[str, str] = {}
        for row in self._split_markdown_table(table_markdown):
            if len(row) >= 2 and row[0]:
                result[row[0]] = row[1]
        return result
