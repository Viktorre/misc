# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:34:14 2021

@author: user
"""
#1. weg, standard modul, aber benutzt outlook
import win32com.client
outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
#msg = outlook.OpenSharedItem(r"C:\test_msg.msg")
mail = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/Keras/sample2.msg'
msg = outlook.OpenSharedItem(mail)


print (msg.Body)

#count_attachments = msg.Attachments.Count
#if count_attachments > 0:
#    for item in range(count_attachments):
#        print msg.Attachments.Item(item + 1).Filename

del outlook, msg




#2. weg, aber mit pip install
import extract_msg

f = mail
msg = extract_msg.Message(f)
msg_sender = msg.sender
msg_date = msg.date
msg_subj = msg.subject
msg_message = msg.body

print('Sender: {}'.format(msg_sender))
print('Sent On: {}'.format(msg_date))
print('Subject: {}'.format(msg_subj))
print('Body: {}'.format(msg_message))





# 3. ohne pip, decode error in pos 1348

start_text = "<html>"
end_text = "</html>"
msg_file= mail
def parse_msg(msg_file,start_text,end_text):
  with  open(msg_file) as f:
      print(dir(f))
      b=f.read()
  return b[b.find(start_text):b.find(end_text)+len(end_text)]

print(parse_msg(mail,start_text,end_text))



































