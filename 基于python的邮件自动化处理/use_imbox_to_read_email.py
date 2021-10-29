#使用python读取邮件。

import yagmail
import keyring
#import imbox
import schedule
"""
对于163，126邮箱，需要提前配置一下
输入网址：http://config.mail.163.com/settings/imap/index.jsp?uid=自己的邮箱
设置允许第三方客户端读取内容
"""
#从keyring中读取密码（前提是已经在python交互端中设置了）
password = keyring.get_password("yagmail","1366359374@qq.com")

#读取所有邮件
from imbox import Imbox
"""
Imbox(IMAP服务器地址，邮箱地址，密码，是否开启SSL加密）
"""
with Imbox("imap.qq.com","1366359374@qq.com",password,ssl=True) as imbox:
    all_inbox_message = imbox.messages() #默认读取所有邮件
    print(len(all_inbox_message))
    # print([*all_inbox_message])
    for uid,messages in all_inbox_message: #uid邮件编号，messages邮件内容
        print(messages.subject) #邮件标题
        print(messages.body["plain"]) #邮件正文
        """
        每个邮件中都可以读取的参数
        message.sent_from：发件人
        message.sent_to：收件人
        message.subject：主题
        message.date：时间
        message.body['plain']：文本格式内容
        messaga.body['html']：HTML格式内容
        message.attachments：附件
        """

    #查看不同类型的邮件
    """
    未读邮件：imbox.messages(unread=True)
    红旗邮件：imbox.messages(flagged=True)
    某发件人邮件：imbox_messages_from = imbox.messages(sent_from = "邮箱地址")
    某收件人邮件：inbox_messages_from = imbox.messages(sent_to = "邮箱地址")
    按照日期筛选邮件：
    import datetime
    date__lt：某天前
    date__gt：某天后
    date__on：指定某一天
    """
    unread_inbox_messages = imbox.messages(unread=True) #查看未读邮件
    inbox_flagged_messages = imbox.messages(flagged = True) #查看标记文件
    inbox_messages_from = imbox.messages(sent_from = "congress@service.sae-china.org") #查看某人发送的邮件
    import datetime
    inbox_messages_received_before = imbox.messages(date__lt=datetime.date(2018,10,1)) #某天前的邮件
print("读取完成")

    #标记已读和删除邮件
"""
        imbox.mark_seen(uid)：标记已读
        imbox.delete(uid)：删除邮件
"""
    # for uid,message in all_inbox_message:
    #     if 满足某种条件的邮件:
    #         imbox.delete(uid)