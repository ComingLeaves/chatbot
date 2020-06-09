import os

#创建文件
def document_creator(doc_path):
    fp = open(doc_path,'w',encoding='UTF-8')
    fp.write("来自小暖的文件：")
    fp.close()

# #方法一：
# for line in lists:
#     f.write(line + '\n')
#
# #方法二：
# lists = [line + "\n" for line in lists]
# f.writelines(lists)
#
# #方法三：
# f.write('\n'.join(lists))

#相对应文件添加记录
def document_product(doc_path,lis):
    with open(doc_path,'a+',encoding='UTF-8') as fp:
        fp.write('\n'+'\n'.join(lis))
        fp.close()


def face(facepath):
    os.mkdir(facepath)



