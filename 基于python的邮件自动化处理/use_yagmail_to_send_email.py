import yagmail #处理SMTP协议
import keyring #从python访问系统密钥环服务，方便安全储存密码
import schedule #定时任务执行器
import imbox #简易的python IMAP包，进行IMAP相关的操作

"""
邮件的相关概念
pop3:Post Office Protocol3的简称，即邮局协议的第三个版本，它规定怎样将个人计算机连接到internet的邮件服务器和下载电子邮件的电子协议
STMP：Simple Mail Transfer Protocol，即简单邮件传输协议
IMAP：Internet Mail Access Protocol，即交互式邮件存取协议，它是跟POP3类似邮件访问标准呢协议之一
#注意：写代码发邮件时候一定要注意不能频繁发送，容易被当作垃圾邮件被屏蔽。
"""
# 1. 注册一个邮箱，
# 开通POP3/SMTP/IMAP，设置第三方邮件客户端专用密码（不同于普通的邮箱登录密码），找到各服务器地址
# 打开python交互式解释器，存入你的邮箱地址和米姆，如下所示：
"""
$python
import yagmail
yagmail.register("用户名","密码")
"""

# 2.发送邮件
import yagmail
print("准备发送邮件")
yag = yagmail.SMTP(user = "1366359374@qq.com",host="smtp.qq.com")
contents = ["老叶，这是一个python自动化办公的测试，请忽略"
            ,"你真帅:you are so handsome"
            ,'<a href="https://www.baidu.com">百度</a>' #发送带有HTML样式的邮件
            ,"G:\FileRecv\自动化办公\自动化处理word文件\样式修改文档.docx" #发送附件（将文件的路径添加即可）
            ,yagmail.inline(r"C:\Users\Administrator\Pictures\Feedback\{A82FF3D6-CFB2-4FFA-B7D7-93ADE2BC6E9A}") #发送嵌入的图片
            ]
yag.send("240014070@qq.com","老叶，这是一个python自动化办公测试，请忽略",contents)
# 群发邮件
# yag.send(["邮箱名1","邮箱名2","邮件说明",contents])
print("邮件发送成功")

# 3.定时邮件任务
import schedule
import time
def job():
    print("定时发送邮件，干活呢")
schedule.every(10).minutes.do(job) #每10分钟运行一次Job
schedule.every().hour.do(job) #每一个小时运行一次job
schedule.every().day.at("10:30").do(job) #每一天的10：30运行一次job
schedule.every(5).to(10).minute(job) #每5~10分钟处理一次job
schedule.every().monday.do(job) #每月处理一次
schedule.every().wednesday.at("13:15").do(job) #每周三的13：15处理一次job
schedule.every().minute.at(":17").do(job) #每分钟第17秒的时候处理一次

while True:
    schedule.run_pending() #运行设定程序
    time.sleep(1)
