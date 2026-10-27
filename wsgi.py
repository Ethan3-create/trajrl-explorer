# -*- coding: utf-8 -*-
"""
WSGI 入口文件 - 用于 Gunicorn 生产部署

启动方式:
    gunicorn -w 4 -b 127.0.0.1:8601 wsgi:app

环境变量:
    TRAJRL_PORT: 服务端口（默认 8601）
    TRAJRL_HOST: 绑定地址（默认 127.0.0.1）
    TRAJRL_WORKERS: worker 进程数（默认 4）
"""
import os
import sys

# 确保能导入当前目录的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from web_server import APIHandler, load_records, DataStore, SAMPLE_PATH, WEB_DIR
from http.server import HTTPServer

# 预加载数据（避免每个 worker 重复加载）
print("[wsgi] loading data...")
records = load_records(SAMPLE_PATH)
store = DataStore(records)
print(f"[wsgi] loaded {len(records)} records")

# 创建 WSGI 应用
class WSGIApp:
    def __init__(self):
        self.records = records
        self.store = store
    
    def __call__(self, environ, start_response):
        # 简化 WSGI 实现：直接代理到 APIHandler
        # 生产环境中通常用 Gunicorn + 自定义 worker 或切换到 Flask/FastAPI
        # 这里提供最小可用实现
        from io import BytesIO
        import urllib.parse
        
        path = environ.get('PATH_INFO', '/')
        method = environ.get('REQUEST_METHOD', 'GET')
        query_string = environ.get('QUERY_STRING', '')
        
        # 构建请求处理器的输入
        class DummyRequest:
            def __init__(self):
                self.path = path + ('?' + query_string if query_string else '')
                self.command = method
                
        # 使用 APIHandler 处理
        handler = APIHandler.__new__(APIHandler)
        handler.path = path + ('?' + query_string if query_string else '')
        handler.command = method
        
        # 收集响应
        response_status = [200]
        response_headers = []
        
        def start_response_wrapper(status, headers):
            response_status[0] = int(status.split()[0])
            response_headers[:] = headers
            
        # 这里需要一个完整的 WSGI 适配
        # 为简化，我们返回一个提示，建议使用原始 HTTP 服务器或迁移到 Flask
        status = '200 OK'
        headers = [('Content-Type', 'text/plain; charset=utf-8')]
        start_response(status, headers)
        return [b"WSGI mode is not fully implemented. Use 'python web_server.py' for development or migrate to Flask for production."]

app = WSGIApp()
