import wordcloud
import jieba
import imageio

#词云图生成代码
#生产模式下的方法
def wc_creator(uid):
    fin = open("static/chatinfo/"+uid+".txt", encoding='UTF-8')
    #fin = open("dialog/xiaohuangji50w_nofenci.conv", encoding='UTF-8')
    txt = fin.read()
    wordlist = jieba.lcut(txt)
    type(wordlist)
    string = " ".join(wordlist)

    image = imageio.imread("static/img/wc-1.jpg")

    wc = wordcloud.WordCloud(width=530, height=530, background_color='white', font_path='static/font/msyh.ttf',mask=image, scale=1)
    wc.generate(string)
    wc.to_file("static/wordcloudpic/"+uid+".png")

    return None



#wc_creator("xdMoUtYn")
# #开发模式下的代码
# fin = open("dialog/xiaonuan.conv",encoding='UTF-8')
# txt = fin.read()
# # txt[:100]
# #使用jieba分词，精确模式（还用全模式和搜索引擎模式）
# wordlist = jieba.lcut(txt)
# type(wordlist)
# # wordlist[:10]
# string = " ".join(wordlist)
#
# image = imageio.imread("static/img/wcpic.jpg")
# #图片的shape形状(行像素，列像素，像素颜色种类组成的个数)
# # image.shape
# # image[0][0]
# # image[100][100]
#
# wc = wordcloud.WordCloud(width=1080,height=810,background_color='white',font_path='static/font/msyh.ttf',mask=image,scale=15)
# wc.generate(string)
# wc.to_file('static/img/result_wordcloud1.png')

