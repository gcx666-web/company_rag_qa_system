class Settings:
    '''
    API配置类
    '''
    MAX_FILE_SIZE = 50*1024*1024#最大为50MB
    ALLOWED_FILE_EXTENSIONS = ['.pdf', '.docx', '.md', '.html']#允许的文件类型
    #上传文件存储目录
    UPLOAD_DIR = 'data/uploads'
settings = Settings()
    