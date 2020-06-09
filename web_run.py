import os,base64
import time,datetime
from flask_cors import *
from datetime import timedelta
from flask import Flask, request, render_template, session, Markup, send_from_directory
from flask import jsonify,send_file,redirect,url_for

from seq2seq import Seq2seq
from DBUtils import insert,delete,update,query,get_uid,select
from document_tool import document_creator,document_product,face
from wc_func import wc_creator
from CalcScore import score
from GetChartData import chartdata
from werkzeug.utils import secure_filename
from os import path
from FaceRecognizer import face_train,match

seq = Seq2seq()
app = Flask(__name__)
app.config['SECRET_KEY']=os.urandom(24)    #设置为24位的字符，每次运行服务器都是不同的，所以服务器启动一次上次的session就清除
app.config['DEBUG']=True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
#app.config['PERMANENT_SESSION_LIFETIME']=timedelta(seconds=1)  #设置session的保存时间
#app.jinja_env.auto_reload = True
#CORS(app, supports_credentials=True) #跨域

#应用启动默认响应页面、创建session
@app.route('/')
def index():
    session.permanent=False  #默认session持续时间31天
    return render_template('Login&Register.html')

#注册页面
@app.route('/registerPost',methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    sql = "select * from t_user where email='" + email + "'"
    result = query(sql)
    if result != None:
        return render_template("information.html", info="该邮箱已被注册！请继续登录！")
    UID = get_uid()
    chatpath = 'static/chatinfo/'+UID+'.txt'
    wordpicpath = 'static/wordpic/'+UID+'.txt'
    keywordpath = 'static/keyword/'+UID+'.txt'
    chat = 'static/chat/'+UID+'.txt'

    rtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    sql = "insert into t_user(uid,statuscode,email,username,pw,chatpath,wordpicpath,keywordpath,rtime) values('%s',1,'%s','%s','%s','%s','%s','%s','%s')" % (UID,email,username,password,chatpath,wordpicpath,keywordpath,rtime)
    result = insert(sql)
    if result == 1:
        document_creator(chatpath)
        document_creator(wordpicpath)
        document_creator(keywordpath)
        document_creator(chat)

        return render_template("information.html", info="注册成功！")
    return render_template("information.html", info="注册失败！请更换邮箱稍后再试！")


#登录、判断是否首次登录
@app.route('/loginPost',methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    sql = "select * from t_user where email='" + email + "'"
    result = query(sql)
    if result==None:
        return render_template("information.html", info="该邮箱尚未注册！")
    else:
        if result.get('pw')==password:
            session['uid'] = result.get('uid')
            session['username'] = result.get('username')
            session['statuscode'] = result.get('statuscode')   #初始值为1，表示首次登录
            session['head_path'] = result.get('head_path')
            session['signature'] = result.get('signature') #个性签名
            session['rtime'] = result.get('rtime') #注册时间
            session['birth'] = result.get('birth')
            session['gender'] = result.get('gender')
            session['fid'] = result.get('fid')
            session['ltime'] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if session['statuscode'] == 1:
                #改变状态码
                sql = "update t_user set statuscode = %s where uid = '%s'" % (0,session['uid'])
                query(sql)
                facepath = 'static/Face_img/' + str(int(session['fid'])-1)
                face(facepath)
                return render_template('XiaoWarmFrame.html',p_url='userinfopage') #应该是进入弹窗请求，后面写到了再改
            return xframe() #进入home页面，后面再改
            #return redirect("http://127.0.0.1:5000/static/XiaoWarmFrame.html")
            #return render_template('ChatPage.html')
        else:
            return render_template("information.html", info="用户名或密码错误！")


#首次登录弹出完善信息窗体
@app.route('/details',methods=['POST'])
def details():
    statuscode = session.get('statuscode')
    uid = session.get('uid')
    username = request.form.get('username')
    email = request.form.get('email')
    sql = "update t_user set email = '%s'，username = '%s',head_path = '%s' where uid = %d" % (email, username,uid)
    result = update(sql)
    if result != None:
        return render_template("chat.html")
    return render_template("information.html",info="数据库出现错误，稍后再试！")


@app.route('/detailsPost',methods=['POST'])
def detailsPost():
    uid = session.get('uid')
    username = request.form.get('username')
    email = request.form.get('email')
    gender = request.form.get('gender')
    birth = request.form.get('birth')
    sql = "update t_user set email = '%s',username = '%s',gender = '%s',birth = '%s' where uid = '%s'" % (email, username,gender,birth,uid)
    result = update(sql)
    if result != None:
        return userinfopage()
    return render_template("information.html", info="数据库出现错误，稍后再试！")

#聊天机器人应答
@app.route('/testPost',methods=['POST','GET'])
def ajaxPost():
    qtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    charinfo = 'static/chatinfo/'+session.get('uid')+'.txt'
    chat = 'static/chat/'+session.get('uid')+'.txt' #拿来做分析用的文本
    qstr = request.values['qstr']
    sql = "insert into chat values('%s','%s','%s')" % (session.get('uid'),qstr,qtime)
    insert(sql)
    # print('========'+qstr+'+++++++++++++++')
    # print(type(qstr))
    # print('===================')
    #astr = (seq.predict(qstr))[0:-7]
    astr = seq.predict(qstr)
    print(astr)
    if astr[-7:0] == "__UNK__" or astr[0:-7] == '痛苦之村列瑟芬':
        astr = "小暖不知道哎，能教教我吗？"
    else :
        astr = astr[0:-7]
    atime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lis = ['['+qtime+']我  > '+qstr,'['+atime+']小暖> '+astr]
    document_product(charinfo,lis)
    document_product(chat,qstr)
    return jsonify({'astr':astr})

@app.route('/facepage',methods=['POST','GET'])
def facepage():
    return render_template("Face_Recognizer.html")

@app.route('/facePost',methods=['POST','GET'])
def facePost():
    #canvas = request.form.get('data')
    img_str = request.values['cs']
    img_data = base64.b64decode(img_str)
    with open('static/temp_img/001.png', 'wb') as f:
        f.write(img_data)
        f.close()
    i,j = match('static/temp_img/001.png')
    print(i)
    print(j)
    if i=="未识别":
        return jsonify({'data':'未识别'})
    if float(j)  < 40:
        session['uid'] = i
        return jsonify({'data': "facelogin"})
    return jsonify({'data': '未识别'})

@app.route('/facelogin',methods=['POST','GET'])
def facelogin():
    sql = "select * from t_user where uid='" + session.get('uid') + "'"
    result = query(sql)
    session['uid'] = result.get('uid')
    session['email'] = result.get('email')
    session['username'] = result.get('username')
    session['statuscode'] = result.get('statuscode')  # 初始值为1，表示首次登录
    session['head_path'] = result.get('head_path')
    session['signature'] = result.get('signature')  # 个性签名
    session['rtime'] = result.get('rtime')  # 注册时间
    session['birth'] = result.get('birth')
    session['gender'] = result.get('gender')
    session['fid'] = result.get('fid')
    session['ltime'] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print("*************************************")
    return xframe()  # 进入home页面，后面再改


#小暖框架
@app.route('/Xframe',methods=['GET','POST'])
def xframe():
    return render_template('XiaoWarmFrame.html',p_url='homePagePost')

#跳转页面
@app.route('/directPost',methods=['GET','POST'])
def homePagePost():
    uinfo = userinfoPost()
    return render_template("home.html",uinfo=uinfo)

@app.route('/adirectPost',methods=['GET','POST'])
def infoPagePost():
    return render_template("information.html")

#frame下的page
@app.route('/homepage',methods=['GET','POST'])
def homepage():
    return render_template("home.html",uinfo=userinfoPost())

@app.route('/chatpage',methods=['GET','POST'])
def chatpage():
    return render_template("chat.html",uinfo=userinfoPost())

@app.route('/introspectionpage',methods=['GET','POST'])
def introspectionpage():
    # 刷新聊天情感分数
    list=[]
    try:
        score(session.get('uid'))
    except TypeError:
        print("Type Error")
    #读取情感评分并导入可视化图表
    try:
        list = chartdata(session.get('uid'))
    except:
        print("have a problem")
    # 刷新词云图
    wc_creator(session['uid'])
    return render_template("Introspection.html",uinfo=userinfoPost(),yAxis=list[0],xAxis=list[1])

@app.route('/treeholePage',methods=['GET','POST'])
def treeholepage():
    sql = "select cd_date,d_title,d_content from diary where uid='%s' order by cd_date DESC" % session.get('uid')
    dic = select(sql)
    cd_dates = []
    d_titles = []
    d_contents = []
    for i in dic:
        cd_dates.append(i['cd_date'])
        d_titles.append(i['d_title'])
        d_contents.append(i['d_content'])
    print(cd_dates)
    print(d_titles)
    print(d_contents)
    return render_template("Treehole.html",cd_dates=cd_dates,d_titles=d_titles,d_contents=d_contents,rtime=session.get('rtime')[0:10])
    #return render_template("Treehole.html",dic=dic)

@app.route('/writediary',methods=['GET','POST'])
def writediary():
    return render_template("writediary.html")

@app.route('/newdiary',methods=['GET','POST'])
def newdiary():
    uid = session.get('uid')
    d_title = request.form.get('d_title')
    d_content = request.form.get('d_content')
    cd_date = time.strftime("%b %d %Y", time.localtime())
    sql = "insert into diary(uid,cd_date,d_title,d_content) values('%s','%s','%s','%s')" % (uid,cd_date,d_title,d_content)
    insert(sql)
    return treeholepage()


@app.route('/userinfopage',methods=['GET','POST'])
def userinfopage():
    return render_template("UserInfo.html",uinfo=userinfoPost())

@app.route('/download',methods=['GET','POST'])
def download():
    print("///*******---+++++")
    return send_from_directory(r"E:\python\编程存放处\xiaowarm\static\chatinfo", filename=session.get('uid')+'.txt', as_attachment=True)

@app.route('/faceidpage',methods=['GET','POST'])
def faceidpage():
    return render_template("FaceID.html")

@app.route('/facetrain',methods=['GET','POST'])
def facetrain():
    face_train()
    return jsonify({'data':'训练完成！'})

@app.route('/aboutpage',methods=['GET','POST'])
def aboutpage():
    return render_template("About.html")

@app.route('/contactpage',methods=['GET','POST'])
def contactpage():
    return render_template("Contact.html")

@app.route('/logout',methods=['GET','POST'])
def logout():
    session.clear
    print("logout!!!!!")
    return redirect(url_for('index'))




#前端ajax获取用户数据
@app.route('/getuserinfo',methods=['GET','POST'])
def userinfoPost():
    getSession()
    uinfo = {'hello':'你好'}
    uinfo['uid'] = session.get('uid')
    uinfo['email'] = session.get('email')
    uinfo['username'] = session.get('username')
    uinfo['statuscode'] = session.get('statuscode')
    uinfo['signature'] = session.get('signature')
    uinfo['head_path'] = session.get('head_path')
    uinfo['rtime'] = session.get('rtime')
    uinfo['birth'] = session.get('birth')
    uinfo['gender'] = session.get('gender')
    uinfo['fid'] = session.get('fid')
    uinfo['ltime'] = session.get('ltime')
    return uinfo

@app.route('/signaturePost',methods=['POST'])
def signPost():
    username = session.get('username')
    sign = request.form.get('moodtext')
    sql = "UPDATE t_user SET signature='%s' WHERE username='%s' " % (sign,username)
    result = update(sql)
    return homePagePost()

#头像上传
@app.route('/headerupload',methods=['GET','POST'])
def headerupload():
    if request.method=='POST':
        f = request.files['headerpic']
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path,'static/uh_img/')
        head_name = session.get('uid')+'.jpg'
        file_name = upload_path + secure_filename(head_name)
        f.save(file_name)
        sql = "UPDATE t_user SET head_path='%s' WHERE uid='%s'" % (head_name,session.get('uid'))
        update(sql)
        return userinfopage()

#FaceID照片上传
@app.route('/faceupload',methods=['GET','POST'])
def faceupload():
    if request.method=='POST':
        f = request.files['face']
        base_path = path.abspath(path.dirname(__file__))
        fid = int(session.get('fid'))-1
        upload_path = path.join(base_path,'static/Face_img/'+str(fid)+'/')
        file_name = upload_path + secure_filename(f.filename)
        f.save(file_name)
        return render_template("information.html", info="上传成功！")


def getSession():
    uid = session.get('uid')
    sql = "select * from t_user where uid='" + uid + "'"
    result = query(sql)
    if result == None:
        return render_template("information.html", info="数据库出现错误，稍后再试！")
    else:
        session['uid'] = result.get('uid')
        session['email'] = result.get('email')
        session['username'] = result.get('username')
        session['statuscode'] = result.get('statuscode')  # 初始值为1，表示首次登录
        session['head_path'] = result.get('head_path')
        session['signature'] = result.get('signature')  # 个性签名
        session['rtime'] = result.get('rtime')  # 注册时间
        session['birth'] = result.get('birth')



app.run()
