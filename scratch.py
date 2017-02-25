# -*- coding: utf-8 -*-

count = 0
for i in clordids:
    msgs = gp.get_group(i)

    if len(msgs)>2:
        print(msgs)
        count +=1
    if count > 10:
        break