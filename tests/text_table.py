import os
import sys

# 放在最最前面，先设置环境变量，再import（docling 需要镜像源下载模型）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 加入项目根目录
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.parsers.pdf_parser import PdfParser
from src.parsers.table_parser import TableProcessor

root_path = os.path.dirname(os.path.dirname(__file__))
pdf_full_path = os.path.join(root_path, "data", "个人生涯规划报告.pdf")

# 1) PdfParser 解析 PDF —— result.tables 就是 merge_tables 的 tables 参数来源
#    每张表形如 {'markdown': 文本, 'rows': 逐行数据, 'page': 页码, 'cols': 列数}
print("开始解析pdf：", pdf_full_path)
result = PdfParser().parse(pdf_full_path)
print("====解析完成==== 表格数量：", result.table_count)

# 2) result.tables 直接交给 merge_tables（跨页拆分的表格按页码自动合并）
processor = TableProcessor()
merged = processor.merge_tables(result.tables)
print("合并后表格数量：", len(merged))
for i, table in enumerate(merged):
    print(f"--- 合并后的表格{i}（第{table.get('page')}页起） ---")
    print(table["markdown"])

# 3) 样例 PDF 没有表格时，改用内置示例演示跨页合并效果
#    （merge_tables 以 markdown 为准重新解析，rows 传空也不影响）
if not merged:
    print("\n(样例PDF无表格，改用内置示例演示 merge_tables)")
    demo_tables = [
        # 第1页的表格
        {"markdown": "| 姓名 | 年龄 |\n|---|---|\n| 张三 | 28 |\n| 李四 | 35 |",
         "rows": [], "page": 1, "cols": 2},
        # 第2页：同一张表被拆分的下半部分（列数相同、页码连续 → 自动合并）
        {"markdown": "|姓名|年龄|\n|---|---|\n|王五|40|", "rows": [], "page": 2, "cols": 2},
        # 另一张独立的表（列数不同 → 不会被并进去）
        {"markdown": "| 商品 | 价格 | 备注 |\n|---|---|---|\n| 苹果 | 5 | 新鲜 |",
         "rows": [], "page": 3, "cols": 3},
    ]
    for i, table in enumerate(processor.merge_tables(demo_tables)):
        print(f"--- 合并后的表格{i}（第{table.get('page')}页起） ---")
        print(table["markdown"])
        print("rows:", table["rows"])
