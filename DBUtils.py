import uuid
import pymysql

# class DB(object):
#     def __init__(self,host="127.0.0.1",port=3306,db="warmdb",user="root",password="mm123456",charset="utf-8"):
#
#         # 创建数据库连接
#         self.dbconn = pymysql.connect(host=host,port=port,db=db,user=user,passwd=password,charset=charset)
#
#         #创建字典型游标(返回的数据是字典类型)
#         self.dbcur = self.dbconn.cursor(cursor = pymysql.cursors.DictCursor)
#
#     # __enter__() 和 __exit__() 是with关键字调用的必须方法
#     # with本质上就是调用对象的enter和exit方法
#     def __enter__(self):
#         # 返回游标
#         return self.dbcur
#
#     def __exit__(self, exc_type, exc_value, exc_trace):
#         # 提交事务
#         self.dbconn.commit()
#
#         # 关闭游标
#         self.dbcur.close()
#
#         # 关闭数据库连接
#         self.dbconn.close()
#
# if __name__ == "__main__":
#     with DB(db="test") as db:
# #        db.execute("show databases")
#         db.execute("select count(*) from test1")
#         print(db)
#         for i in db:
#             print(i)
#

#数据库增删改查操作
def insert(sql):
    # 数据库连接
    db = pymysql.connect("127.0.0.1", "root", "123456", "warmdb", cursorclass=pymysql.cursors.DictCursor)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    result = cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()
    return result

def delete(sql):
    db = pymysql.connect("127.0.0.1", "root", "123456", "warmdb", cursorclass=pymysql.cursors.DictCursor)
    cursor = db.cursor()
    result = cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()
    return result

def update(sql):
    # 数据库连接
    db = pymysql.connect("127.0.0.1", "root", "123456", "warmdb", cursorclass=pymysql.cursors.DictCursor)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    result = cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()
    return result

def query(sql):
    # 数据库连接
    db = pymysql.connect("127.0.0.1", "root", "123456", "warmdb", cursorclass=pymysql.cursors.DictCursor)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    results = cursor.fetchone()
    cursor.close()
    db.close()
    return results

def select(sql):
    db = pymysql.connect("127.0.0.1", "root", "123456", "warmdb", cursorclass=pymysql.cursors.DictCursor)
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    results = cursor.fetchall()
    cursor.close()
    db.close()
    return results

#自动生成8位uuid
def get_uid():
    array = ["a", "b", "c", "d", "e", "f",
             "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
             "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5",
             "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
             "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
             "W", "X", "Y", "Z"]
    id = str(uuid.uuid4()).replace("-", '')
    buffer = []
    for i in range(0, 8):
        start = i * 4
        end = i * 4 + 4
        val = int(id[start:end], 16)
        buffer.append(array[val % 62])
    return "".join(buffer)

