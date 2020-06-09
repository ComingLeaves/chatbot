from DBUtils import select,insert,delete
from SentimentAnalysis import sa_info
import datetime

def score(uid):
    sql = "select chat from chat where uid='%s'" % uid
    chats = select(sql)
    if chats==None:
        return None
    str = ""
    for chat in chats:
        str = chat['chat'] + " " + str
    score_num = round(sa_info(str), 3)
    sql = "delete from chat where uid='%s'" % uid
    delete(sql)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(score_num)
    sql = "insert into t_score(uid,c_score,c_date) values('%s',%f,'%s')" % (uid,score_num,date)
    insert(sql)

#模块测试
#score("xdMoUtYn")