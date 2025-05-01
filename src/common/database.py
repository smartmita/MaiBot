import os
from pymongo import MongoClient
from pymongo.database import Database

_client = None
_db = None


def __create_database_instance():
    uri = os.getenv("MONGODB_URI")
    host = os.getenv("MONGODB_HOST", "127.0.0.1")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    # db_name 变量在创建连接时不需要，在获取数据库实例时才使用
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    auth_source = os.getenv("MONGODB_AUTH_SOURCE")

    if uri:
        # 支持标准mongodb://和mongodb+srv://连接字符串
        if uri.startswith(("mongodb://", "mongodb+srv://")):
            return MongoClient(uri)
        else:
            raise ValueError(
                "Invalid MongoDB URI format. URI must start with 'mongodb://' or 'mongodb+srv://'. "
                "For MongoDB Atlas, use 'mongodb+srv://' format. "
                "See: https://www.mongodb.com/docs/manual/reference/connection-string/"
            )

    if username and password:
        # 如果有用户名和密码，使用认证连接
        return MongoClient(host, port, username=username, password=password, authSource=auth_source)

    # 否则使用无认证连接
    return MongoClient(host, port)


def get_db():
    """获取数据库连接实例，延迟初始化。"""
    global _client, _db
    if _client is None:
        _client = __create_database_instance()
        _db = _client[os.getenv("DATABASE_NAME", "MegBot")]
    return _db


class DBWrapper:
    """数据库代理类，保持接口兼容性同时实现懒加载。"""

    def __getattr__(self, name):
        return getattr(get_db(), name)

    def __getitem__(self, key):
        return get_db()[key]

def close_db():
    """关闭全局 MongoDB 客户端连接。"""
    global _client, _db
    if _client:
        try:
            _client.close()
            # print(f"数据库连接已由进程 {os.getpid()} 关闭。") # 可选：添加日志
        except Exception as e:
            # print(f"关闭数据库连接时出错: {e}") # 可选：记录关闭错误
            pass # 关闭期间避免程序崩溃
        finally:
            # 重置全局变量，以便下次 get_db 能重新连接（如果需要）
            _client = None
            _db = None

# 全局数据库访问点
db: Database = DBWrapper()
