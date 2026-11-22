import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

#项目根目录（src/api/config.py 往上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    '''
    API配置类
    '''
    MAX_FILE_SIZE = 50*1024*1024#最大为50MB
    ALLOWED_FILE_EXTENSIONS = ['.pdf', '.docx', '.htm', '.html', '.md']#允许的文件类型
    #上传文件存储目录（绝对路径，避免依赖启动时的工作目录）
    UPLOAD_DIR = str(PROJECT_ROOT / 'data' / 'uploads')
    #向量库持久化目录
    CHROMA_PERSIST_DIR = str(PROJECT_ROOT / 'data' / 'chroma_db')
    #LLM供应商
    LLM_PROVIDER = {
        'zhipu':{
            'API_KEY' : os.getenv('ZHIPU_API_KEY'),
            'BASE_URL': os.getenv('ZHIPU_BASE_URL'),
            'MODEL': os.getenv('ZHIPU_MODEL'),
        },
        'qwen':{
            'API_KEY' : os.getenv('QWEN_API_KEY'),
            'BASE_URL': os.getenv('QWEN_BASE_URL'),
            'MODEL': os.getenv('QWEN_MODEL'),
        },
        'siliconflow':{
            'API_KEY' : os.getenv('SILICONFLOW_API_KEY'),
            'BASE_URL': os.getenv('SILICONFLOW_BASE_URL'),
            'MODEL': os.getenv('SILICONFLOW_MODEL'),
        },
    }
    #LLM参数
    LLM_TEMPERATURE=0.7
    LLM_MAX_TOKENS=2000
    TOP_K=5
    Default_LLM_MODEL=os.getenv('DEFAULT_LLM_MODEL')
    Default_LLM_BASE_URL=os.getenv('DEFAULT_LLM_BASE_URL')
    Default_LLM_API_KEY=os.getenv('DEFAULT_LLM_API_KEY')


settings = Settings()