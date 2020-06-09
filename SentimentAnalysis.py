from aip import AipNlp


APP_ID = '18372708'
API_KEY = 'pNCqdD84NmPRoCEYk20WiE6E'
SECRET_KEY = 'PG2Wxl2lqlgMb7aZlwcVXOvayGGwnAdR'

#百度接口的情感分析
client = AipNlp(APP_ID,API_KEY,SECRET_KEY)
def sa_info(text):
    dic = client.sentimentClassify(text)
    list = dic.get('items')
    dic = list[0]
    result = dic.get('positive_prob')
    return result

