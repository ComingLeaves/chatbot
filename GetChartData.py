from DBUtils import select,insert,delete

def chartdata(uid):
    sql = "select c_score,c_date from t_score where uid='%s' order by c_date ASC" % uid
    dic = select(sql)
    xAxis = []
    yAxis = []

    for i in dic:
        yAxis.append(i['c_score'])
        xAxis.append(i['c_date'])

    list = [yAxis, xAxis]
    return list

#单元测试
# list = chartdata('xdMoUtYn')
# print(list[0])
# print(list[1])