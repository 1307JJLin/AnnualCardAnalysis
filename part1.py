def get_mid(start, duration):
    times = str(start).split(':')
    minutes = int(times[0]) * 60 + int(times[1])
    res = minutes + duration // 2
    hour = res // 60
    minute = res - hour * 60
    return f'{int(hour)}:{int(minute)}' if minute >= 10 else f'{int(hour)}:0{int(minute)}'

def get_period(hour):
    ''' 
    一天的时间段划分为:
    8-12点: morning
    12-13点: noon
    13-19点: afternoon
    19点到24点: evening
    0-8点: night
    '''
    hour = int(hour)
    if hour in range(8, 12):
        return 'morning'
    elif hour == 12:
        return 'noon'
    elif hour in range(13, 19):
        return 'afternoon'
    elif hour in range(19, 24):
        return 'evening'
    return 'night'

def get_duration(duration):
    ''' 
    30分钟以下为short
    30-60分钟为middle
    60分钟以上为long
    '''
    if int(duration) < 30:
        return 'short'
    elif int(duration) <= 60:
        return 'middle'
    return 'long'

def correct_events(e):
    alist = {
        '代码':'改代码',
        '拍照':'摄影',
        '简历':'简历',
        '总结':'写报告',
        '报告':'写报告',
        '面试': '面试',
        '博弈论':'博弈论',
        '磨蹭':'磨蹭',
        'PPT':'PPT',
        '快递':'快递',
        'TPO':'托福听力',
        '算法':'算法课程',
        '推文':'写推文',
        '演唱会':'演唱会',
        '人文课':'人文课',
        'AQF':'AQF',
    }
    for k in alist:
        if k in e:
            e = alist[k]
    return e