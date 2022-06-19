# encoding: utf-8
import numpy as np
import datetime
import os

def step1(path):
    '''
     calculate dait score and rait score
    :param path:
    :return:
    '''

    last_user_id=0
    last_item_id=0
    time_list=[]
    peroid_list=[]
    rait=[]
    s_rait=[]
    user_rait={}
    k=0
    user=[]
    with open(path,'r')as f:
        for line in f:
            line = line.strip("\n").split('\t')
            user_id=line[0]
            item_id=line[1]
            if last_user_id==0 and last_item_id==0:
                last_user_id=user_id
                last_item_id=item_id
                peroid_list.append(line[2])
                time_list.append(line[3])
                user.append(last_user_id)
                continue
            if last_item_id==item_id and last_user_id==user_id:
                peroid_list.append(line[2])
                time_list.append(line[3])
                continue

            rait1, rait2, rait3, rait4= rait_calculate(peroid_list, time_list)
            rait.append(rait1)
            rait.append(rait2)
            rait.append(rait3)
            rait.append(rait4)
            s_rait.append(rait)
            rait=[]
            last_item_id = item_id
            if last_user_id!=user_id:
                user_rait[last_user_id]=s_rait
                s_rait=[]
                last_user_id = user_id
                user.append(last_user_id)
            time_list = []
            peroid_list = []
            peroid_list.append(line[2])
            time_list.append(line[3])
        rait1, rait2, rait3, rait4 = rait_calculate(peroid_list, time_list)
        rait.append(rait1)
        rait.append(rait2)
        rait.append(rait3)
        rait.append(rait4)
        s_rait.append(rait)
        user_rait[last_user_id] = s_rait
    print(len(user))
    return  user_rait,user

def step2(user_rait,user):
    '''
    calculate the set of virtual users
    :param user_rait:
    :param user:
    :return:
    '''

    s={}
    for u in range(len(user)):
        user_id=user[u]
        list=[]
        rait_for_user=user_rait[user_id]
        for i in range(4):
            for j in range(i+1,4):
                up=0
                down1=0
                down2=0
                for k in range(len(rait_for_user)):
                    rai=rait_for_user[k]
                    up+=rai[i]*rai[j]
                    down1+=rai[i]
                    down2+=rai[j]
                if down1==0 and down2==0:
                    list.append([i,j])
                    continue
                sa=up/(down1*down2)**(1/2)
                if sa>0.5:
                    list.append([i, j])
        set_list=[set([0]),set([1]),set([2]),set([3])]
        for p in range(len(list)):
            pair_1=list[p][0]
            pair_2=list[p][1]
            for ss in range(len(set_list)):
                if pair_1 in set_list[ss]:
                    pair_1=set_list[ss]
                    break

            for ss in range(len(set_list)):
                if pair_2 in set_list[ss]:
                    pair_2=set_list[ss]
                    break

            set_list.remove(pair_1)
            if pair_1-pair_2:
                set_list.remove(pair_2)
            union=pair_1|pair_2
            set_list.append(union)
        s[user_id]=set_list
    return s,user

def step3(s,user,path,path2):
    '''
    merge the data of virtual users
    :param s:
    :param user:
    :param path:
    :param path2:
    :return:
    '''

    l=0
    merge_set=set()
    with open(path,'r') as f,open(path2,'w')as w:
        for line in f:
            line = line.strip("\n").split('\t')
            l+=1
            user_id=line[0]
            peroid_id=int(line[2])-1
            if user_id in user:
                set_list=s[user_id]
                for i in range(len(set_list)):
                    if peroid_id in set_list[i]:
                        new_peroid=min(set_list[i])
                w.write(line[0]+"\t"+line[1]+"\t"+str(new_peroid)+"\t"+str(peroid_id)+"\n")

def step4(path,path2):
    '''
    sort the data
    :param path:
    :param path2:
    :return:
    '''

    i=0
    data=[]
    with open(path,'r')as f,open(path2,'w')as w:
        for line in f:
            line = line.strip("\n").split('\t')
            i+=1
            data.append(line)
        data= sorted(data,key=lambda d: d[2])
        data = sorted(data, key=lambda d: d[0])
        for i in range(len(data)):
            w.write(data[i][0] + "\t" + data[i][1] + "\t" + data[i][2]+"\t"+data[i][3] + "\n")

def rait_calculate(peroid_list,time_list):
    time1=0
    time2=0
    time3=0
    time4=0
    for i in range(len(peroid_list)):
        if peroid_list[i]=='1':
            time1+=(int(time_list[i])+0.0)/3600
            continue
        if peroid_list[i]=='2':
            time2+=(int(time_list[i])+0.0)/3600
            continue
        if peroid_list[i] == '3':
            time3 += (int(time_list[i])+0.0)/3600
            continue
        if peroid_list[i]=='4':
            time4+=(int(time_list[i])+0.0)/3600
            continue
    print('time1:',int(time1))
    print('time2:',int(time2))
    print('time3:',int(time3))
    print('time4:',int(time4))
    if time1 >= 710:
        time1 = 709
    if time2 >= 710:
        time2 = 709
    if time3 >= 710:
        time3 = 709
    if time4 >= 710:
        time4 = 709
    s = np.e ** time1 + np.e ** time2 + np.e ** time3 + np.e ** time4

    rait1 = (np.e ** time1+0.0)/ s
    rait2 = (np.e ** time2+0.0)/ s
    rait3 = (np.e ** time3+0.0)/ s
    rait4 = (np.e ** time4+0.0)/ s
    if time1==0:
        rait1=0
    if time2==0:
        rait2=0
    if time3==0:
        rait3=0
    if time4==0:
        rait4=0
    return rait1,rait2,rait3,rait4

def data_divide(path,path2):
    '''
    Divide the original data into 6 time periods
    :param path:
    :param path2:
    :return:
    '''

    t1 = "00:00:00"
    t2 = "06:00:00"
    t3 = "12:00:00"
    t4 = "18:00:00"
    t5 = "24:00:00"
    number=0
    new_num=0
    with open(path,'r')as f,open(path2,'w')as w:
        for line in f:
            line = line.strip().split('\t')
            number+=1
            begin_time=line[4]
            end_time=line[6]
            if begin_time>=t1 and end_time<t2:
                w.write(line[0]+"\t"+line[1]+"\t"+str(1)+"\t"+line[7]+"\n")
                new_num+=1
                continue
            if begin_time>=t1 and begin_time<t2 and end_time<t3 and end_time>=t2:
                time=(datetime.datetime.strptime(t2,'%H:%M:%S')-datetime.datetime.strptime(line[4],
                                                                                           '%H:%M:%S')).seconds
                w.write(line[0]+"\t"+line[2]+"\t"+str(1)+"\t"+str(time)+"\n")
                time=(datetime.datetime.strptime(line[6],'%H:%M:%S')-datetime.datetime.strptime(t2,
                                                                                                '%H:%M:%S')).seconds
                w.write(line[0]+"\t"+line[2]+"\t"+str(2)+"\t"+str(time)+"\n")
                new_num += 2
                continue
            if begin_time>=t1 and begin_time<t2 and end_time<t4 and end_time>=t3:
                time = (datetime.datetime.strptime(t2, '%H:%M:%S') - datetime.datetime.strptime(line[4],
                                                                                                '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(1) + "\t" + str(time) + "\n")
                w.write(line[0] + "\t" + line[2] + "\t" + str(2) + "\t" + str(21600) + "\n")
                time = (datetime.datetime.strptime(line[6], '%H:%M:%S') - datetime.datetime.strptime(t3,
                                                                                                     '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(3) + "\t" + str(time) + "\n")
                new_num += 3
                continue
            if begin_time>=t1 and begin_time<t2 and end_time<=t5 and end_time>=t4:
                time = (datetime.datetime.strptime(t2, '%H:%M:%S') - datetime.datetime.strptime(line[4],
                                                                                                '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(1) + "\t" + str(time) + "\n")
                w.write(line[0] + "\t" + line[2] + "\t" + str(2) + "\t" + str(21600) + "\n")
                w.write(line[0] + "\t" + line[2] + "\t" + str(3) + "\t" + str(21600) + "\n")
                time = (datetime.datetime.strptime(line[6], '%H:%M:%S') - datetime.datetime.strptime(t4,
                                                                                                     '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(4) + "\t" + str(time) + "\n")
                new_num += 4
                continue
            if begin_time>=t2 and end_time<t3:
                w.write(line[0]+"\t"+line[2]+"\t"+str(2)+"\t"+line[7]+"\n")
                new_num += 1
                continue
            if begin_time>=t2 and begin_time<t3 and end_time<t4 and end_time>=t3:
                time = (datetime.datetime.strptime(t3, '%H:%M:%S') - datetime.datetime.strptime(line[4],
                                                                                                '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(2) + "\t" + str(time) + "\n")
                time = (datetime.datetime.strptime(line[6], '%H:%M:%S') - datetime.datetime.strptime(t3,
                                                                                                     '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(3) + "\t" + str(time) + "\n")
                new_num += 2
                continue
            if begin_time>=t2 and begin_time<t3 and end_time<=t5 and end_time>=t4:
                time = (datetime.datetime.strptime(t3, '%H:%M:%S') - datetime.datetime.strptime(line[4],
                                                                                                '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(2) + "\t" + str(time) + "\n")
                w.write(line[0] + "\t" + line[2] + "\t" + str(3) + "\t" + str(21600) + "\n")
                time = (datetime.datetime.strptime(line[6], '%H:%M:%S') - datetime.datetime.strptime(t4,
                                                                                                     '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(4) + "\t" + str(time) + "\n")
                new_num += 3
                continue
            if begin_time>=t3 and end_time<t4:
                w.write(line[0] + "\t" + line[2] + "\t" + str(3) + "\t" + line[7] + "\n")
                new_num += 1
                continue
            if begin_time>=t3 and begin_time<t4 and end_time<=t5 and end_time>=t4:
                time = (datetime.datetime.strptime(t4, '%H:%M:%S') - datetime.datetime.strptime(line[4],
                                                                                                '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(3) + "\t" + str(time) + "\n")
                time = (datetime.datetime.strptime(line[6], '%H:%M:%S') - datetime.datetime.strptime(t4,
                                                                                                     '%H:%M:%S')).seconds
                w.write(line[0] + "\t" + line[2] + "\t" + str(4) + "\t" + str(time) + "\n")
                new_num += 2
                continue
            if begin_time>=t4 and end_time<=t5:
                w.write(line[0] + "\t" + line[2] + "\t" + str(4) + "\t" + str(line[7]) + "\n")
                new_num += 1
                continue
            # print("begin:"+str(begin_time)+"\t"+"end:"+str(end_time))
        print(number)
        print(new_num)

def data_sort(path,path2):
    data=[]
    with open(path,'r')as f:
        for line in f:
            line=line.strip('\n').split('\t')
            data.append(line)
        data = sorted(data,key=lambda d: d[1])  # sort the item_id
        data=sorted(data,key=lambda d: d[0])  # sort the user_id
    with open(path2,'w')as w:
        for i in range(len(data)):
            w.write(data[i][0]+"\t"+data[i][1]+"\t"+data[i][2]+"\t"+data[i][3]+"\n")


if __name__ == '__main__':

    path = '"V(E) domain train data path"'
    path2 = '"generate V(E) domain train peroid data path"'
    data_divide(path, path2)  # Virtually divide the time of the training set
    path3 = '"V(E)-domain test data path"'
    path4 = '"generate V(E) domain test peroid data path"'
    data_divide(path3, path4) # Virtually divide the time of the test set

    path5 = '"generate V(E) domain train sort data path"'
    data_sort(path2, path5)
    user_rait, user = step1(path5)
    s, user = step2(user_rait, user)
    path6 = '"generate V(E) domain train_period_merge data path"'
    step3(s, user, path2, path6)
    path9 = '"generate V(E) domain train_sorted_merge data path"'
    step4(path6, path9)

    path7 = '"generate V(E) domain test_sort data path"'
    data_sort(path4, path7)
    user_rait, user = step1(path7)
    path8 = '"generate V(E) domain test_period_merge data path'
    s, user = step2(user_rait, user)
    step3(s, user, path4, path8)  
