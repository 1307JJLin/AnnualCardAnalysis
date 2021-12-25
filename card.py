'''
本代码由公众号【杰然不同之GR】原创，仅作个人学习，请勿用于任何商业用途。
转载及其他形式合作请与我联系，未经授权严禁搬运及二次创作，侵权必究!

本代码用于记录每天所做的事情，便于对自己一天情况有所了解，长期使用本代码记录数据，
积累下来的excel文件可以作为本hub上配套数据分析代码的数据源。

本代码适用于需要长期“修行”的人对自己自律情况进行自检，例如考研党，留学党。
代码的全部内容都在此文件中，所以请使用者放心，绝对不会有隐私泄露的问题。

使用方法简介：
0) 修改本文件中第150行代码（path = ...）中的路径，后续生成的数据就会保存在该文件夹下
1) 用python运行本文件，出现“输入任意键进入新的一次打卡记录的输入，输入end结束这一天:”时，代表程序开始运行
2) 做完一件事后，输入任意键再按enter，就进入到该事件的记录中。
需要记录的包括有事件名称，以及事件对应的属性（有效时间，浪费时间，必要时间）
事件名称需要自己全部输入，而事件的属性只需要输入对应的数字代码即可（1-必要, 2-有效, 3-浪费）
3) 一次事件记录完成后，程序会输出已有的结果，使用者需要自己检查是否有误，无误则按enter继续
4) 结束一整天的记录时输入end再按enter，这时程序会提醒使用者输入当日的手机使用时间

一些常见问题的回答
Q1：记录的时候出错了怎么办？
A：每次记录后，程序都会展示目前已有的所有记录，并提示是否需要修改。
如果需要修改，则输入y并按enter，就会进行修改流程。默认只修改事件的名称或是对应的属性。

Q2：记录的中途停电了或是出现了其他意外中断怎么办？
A：每一次记录确认无误之后，程序就会把当前的记录保存在一开始设定的路径下。
所以，中途停止了也没关系，重新运行本代码即可，程序会提示是否从已有的记录继续，选择是即可

Q3：在做了好几件事之后才想起来记录，之前的那些事要怎么记录呢？
A：这时候请退出该程序，在设定用于保存的路径下找到今日的excel，打开之后按照已有记录的格式
手动输入忘记记录的事件。之后保存excel，再打开本代码继续记录。

其他问题以及使用效果可参照公众号【杰然不同之GR】于2021年12月31日发表的名为
《再见，2021(下)》推文中1.1部分的内容。
'''


import time
import pandas as pd
import os
import math

class CardSystem:
    def __init__(self, path, date):
        self.date = date
        self.path = path
        try:
            self.info = pd.read_excel(os.path.join(self.path, f"{self.date}打卡记录.xlsx"), engine='openpyxl').iloc[:, 1:]
        except:
            self.info = pd.DataFrame()
        self.code = {'1':'必要', '2':'有效', '3':'浪费'}
          
    # 开始一天的记录
    def begin(self):
        if self.info.shape[0] == 0:
            start = time.strftime('%H:%M:%S', time.localtime())
            print('新的一天打卡开始了，这一次的开始时间是:', start)
            return time.time(), time.localtime()
        else:
            print(self.info)
            sign = input('检查到今天已经有打卡记录，是否将上一次记录中的最后一次打卡时间作为此次打卡的开始时间，1-是，2-否:')
            while sign not in ['1', '2']:
                print('请确保输入范围正确')
                sign = input('检查到今天已经有打卡记录，是否将上一次记录中的最后一次打卡时间作为此次打卡的开始时间，1-是，2-否:')
            if sign == '2':
                start = time.strftime('%H:%M:%S', time.localtime())
                print('新的一天打卡开始了，这一次的开始时间是:', start)
                return time.time(), time.localtime()
            row_num = self.info.shape[0] - 1
            last = str(self.info.loc[row_num, 'date']) + ' ' + str(self.info.loc[row_num, 'end'])
            timearray = time.strptime(last,"%Y-%m-%d %H:%M:%S")
            print('新的一天打卡开始了，这一次的开始时间是:', last.split(' ')[-1])
            return time.mktime(timearray), time.localtime(time.mktime(timearray))

    # 结束一天的记录
    def end(self):
        summary = self.info.groupby('attr').sum().T
        summary['日期'] = self.date
        phone = eval(input('输入今日的手机时间:'))
        check = input (f'检查手机时间是否输入错误，输入"y"进行修改，你输入的手机时间是{phone}分钟:')
        while check == 'y':
            phone = eval(input('输入今日的手机时间(单位:分钟):'))
            check = input(f'检查手机时间是否输入错误，输入"y"进行修改，你输入的手机时间是{phone}分钟:')
        summary['手机时间'] = phone
        type_list = ['有效时间', '浪费时间', '必要时间']
        for time_type in type_list:
            if time_type not in summary.columns:
                summary[time_type] = 0
        summary[['日期', '有效时间', '浪费时间', '必要时间', '手机时间']].set_index('日期').\
            to_excel(os.path.join(self.path, f"{self.date}打卡总结.xlsx"))

    # 每次事件记录后的检查
    def check(self):
        print('这是目前已有的记录\n', self.info[['date', 'event', 'start', 'end', 'duration', 'attr']])
        check = input('是否需要修改？输入"y"进行修改，其他任意键不修改:')
        while check == 'y':
            print('这是目前已有的记录\n', self.info[['date', 'event', 'start', 'end', 'duration', 'attr']])
            idx = eval(input('请输入想要修改行数:'))
            col = input('请输入想要修改的列代码，1-事项, 2-属性:')
            while col not in ['1', '2']:
                print('请确保输入范围正确')
                col = input('请输入想要修改的列代码，1-事项, 2-属性:')
            if col == '1':
                new_event = input('输入做的新事项:')
                self.info.loc[idx, 'event'] = new_event
            else:
                new_attr = input('输入该事项的性质代码数字:1-必要, 2-有效, 3-浪费,请保证代码数字的范围正确:')
                while new_attr not in ['1', '2','3']:
                    print('请确保输入范围正确')
                    new_attr = input('输入该事项的性质代码数字:1-必要, 2-有效, 3-浪费,请保证代码数字的范围正确:')
                self.info.loc[idx, 'attr'] = self.code.get(new_attr) + '时间'     
            print('这是目前已有的记录\n', self.info[['date', 'event', 'start', 'end', 'duration', 'attr']])
            check = input('是否需要修改？输入"y"进行修改，其他任意键不修改:')
        
    # 每次时间的记录
    def record(self):
        start = self.begin()
        sign = input('输入任意键进入新的一次打卡记录的输入，输入end结束这一天:')
        while sign != 'end':
            end = time.time(), time.localtime()           
            event = input('输入这段时间内做的事:')
            attr = input('输入该事项的性质代码数字:1-必要, 2-有效, 3-浪费,请保证代码数字的范围正确:')
            while attr not in ['1', '2','3']:
                print('请确保输入范围正确')
                attr = input('输入该事项的性质代码数字:1-必要, 2-有效, 3-浪费,请保证代码数字的范围正确:')
            duration = end[0] - start[0]
            info = dict(
                date=self.date, 
                event=event, 
                start=time.strftime('%H:%M:%S', start[1]), 
                end=time.strftime('%H:%M:%S', end[1]),
                duration=math.ceil(duration / 60),
                attr=self.code.get(attr) + '时间'
            )
            self.info = self.info.append(info, ignore_index=True)
            self.check()
            start = end
            self.info[['date', 'event', 'start', 'end', 'duration', 'attr']].\
                to_excel(os.path.join(self.path, f"{self.date}打卡记录.xlsx"))
            sign = input('输入任意键进入新的一次打卡记录的输入，输入end结束这一天:')

    # 运行函数
    def run(self):
        self.record()
        self.end()
            

def main():
    date = time.strftime('%Y-%m-%d', time.localtime())
    path = r'D:\打卡\2021'
    CardSystem(path, date).run()

if __name__ == '__main__':
    main()