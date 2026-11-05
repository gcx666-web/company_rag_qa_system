import os
import sys

# 放在最最前面，先设置环境变量，再import
# hf-mirror.com Xet 协议返回 401，官方源连接超时；禁用 Xet 协议后用镜像源常规下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 加入项目根目录
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.parsers.pdf_parser import PdfParser

parser = PdfParser()
root_path = os.path.dirname(os.path.dirname(__file__))
pdf_full_path = os.path.join(root_path, "data", "个人生涯规划报告.pdf")

print("开始解析pdf：", pdf_full_path)
result = parser.parse(pdf_full_path)
print("====解析完成====")
print("文件名：", result.file_name)
print("页数：", result.total_pages)
print("表格数量：", len(result.tables))
print("文本预览：", result.raw_text[:500])
