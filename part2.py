import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pylab as pl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import pandas as pd
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont
import networkx as nx 
from scipy import spatial


# 数值变量直方图
def numeric_hist(df, col, year, part, path):
    temp = df[df['year'] == str(year)].copy()
    if col == 'phone':
        temp = temp[[col, 'date']].drop_duplicates()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.hist(temp[col], bins=20)
    plt.xlabel(col, fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    title = f'{part} Histgram of {col} in year {year}'
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)

# 类别变量饼图
def plot_pie(data, col, n, part, path, suffix='', year=None):
    fig = plt.figure(figsize=(20, 10), dpi=100)
    fig.tight_layout()
    ax = fig.add_subplot(111)
    df = data.copy()
    if year is not None:
        try:
            temp = df[df['year'] == str(year)].copy()
        except:
            temp = df.copy()
    else:
        temp = df.copy()
    data = temp[col].value_counts().sort_values(ascending=False).values[:n] 
    labels = temp[col].value_counts().sort_values(ascending=False).index[:n]
    explodes = [0] * len(data)
    explodes[0] = 0.015
    ax.pie(
        data, 
        labels=labels, 
        radius=0.8, 
        explode=explodes, 
        autopct='%1.1f%%', 
        pctdistance = 0.5,
        labeldistance=0.7,  
        textprops={'fontsize': 25, 'color': 'black'}
    ) 
    plt.axis('equal') 
    title = f'{part} Pie plot of {col} in year {year} {suffix}' if year is not None else \
        f'{part} Pie plot of {col} {suffix}'
    plt.title(title, fontsize=20) 
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)

# 专注力分析
def concentrate(df, year, part, path):
    temp = df[
        (df['year'] == str(year)) & (df['attr'] == '有效时间')
    ].copy()

    # Figure 1
    plt.figure(dpi=100)
    df[df['year'] == str(year)].groupby(['attr', 'duration_attr'])['date'].\
    count().unstack().plot(kind='bar', figsize=(20, 10))
    title = f'{part} Duration attribution of each time in year {year}'
    plt.title(title, fontsize=20)
    plt.xlabel('Effective time duration', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    file = _extracted_from_concentrate_16(path, title)
    temp['duration'].plot(kind='hist', figsize=(20, 10))
    title = _extracted_from_concentrate_22(part, ' Duration of effective time in year ', year)

    plt.ylabel('Frequency', fontsize=20)
    file = _extracted_from_concentrate_16(path, title)
    temp['duration_attr'].value_counts().plot(kind='bar', figsize=(20, 10))
    title = _extracted_from_concentrate_22(part, ' Bar plot of duration attribution of effective time in year ', year)

    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)


# TODO Rename this here and in `concentrate`
def _extracted_from_concentrate_16(path, title):
    result = os.path.join(path, f'{title}.png')
    plt.savefig(result, dpi=100)

    # Figure 2
    plt.figure(dpi=100)
    return result


# TODO Rename this here and in `concentrate`
def _extracted_from_concentrate_22(part, arg1, year):
    result = f'{part}{arg1}{year}'
    plt.title(result, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return result

# num vs cate
def plot_num_vs_cate(df, num, cate, path, part, n=30, year=2022):
    if cate != 'year':
        stats = ['max', 'min', 'mean', 'std', 'median', 'sum']
        for sts in stats:
            temp = df[df['year'] == str(year)].groupby(cate)[num].\
            agg(sts).sort_values(ascending=False)
            plt.figure(dpi=100)
            title = f"{part} bar plot of {sts} of {num} in different {cate} groups in year {year}"
            temp[:n].plot(
                kind='bar', 
                figsize=(20, 10),
            )
            plt.title(title, fontsize=20)
            plt.xlabel(cate, fontsize=20)
            plt.ylabel(f'{sts} of {num}', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            file = os.path.join(path, f'{title}.png')
            plt.savefig(file, dpi=100)

# 箱型图
def plot_box(df, num, cate, path, part, n=30, year=2022):
    temp = df[df['year'] == str(year)].groupby(cate)[num].agg('median').\
        sort_values(ascending=False)[:n]
    plt.figure(figsize=(20, 10), dpi=100)
    sns.boxplot(
        x=cate, 
        y=num, 
        data=df[df[cate].isin(temp.index)], 
        order=temp.index
    )
    title = f'{part} Boxplot of {num} wrt {cate} in year {year}'
    plt.title(title, fontsize=20)
    plt.xlabel(cate, fontsize=20)
    plt.ylabel(num, fontsize=20)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)

# 计数图
def plot_count_plot(df, c1, c2, path, part, n=15, year=2022):
    temp = df[df['year'] == str(year)].groupby(c1)[c2].\
    count().sort_values(ascending=False)[:n]
    plt.figure(figsize=(20, 10), dpi=100)
    sns.countplot(
        x=c1, 
        hue=c2, 
        data=df[df[c1].isin(temp.index)], 
        order=temp.index
    )
    plt.legend() if c2 != 'week_order' else plt.legend([])
    title = f'{part} Countplot of {c1} in different {c2} group'
    plt.title(title, fontsize=20)
    plt.xlabel(c1, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    if c2 not in ['day', 'week_order']:
        plt.legend(fontsize=20, loc=1) 
    else:
        plt.legend([])
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)

# 每年各类时间折线图对比
def line_compare(daily_attr, col, part, path):
    temp = daily_attr[['year', 'day_of_year', col]].melt(
        id_vars=[col, 'day_of_year']
    )
    plt.figure(figsize=(20, 10))
    for y in temp['value'].unique():
        tmp = temp[temp['value'] == y]
        plt.plot(
            tmp['day_of_year'],
            tmp[col],
            label=y,
        )
    plt.legend()
    plt.xlabel('Day of year', fontsize=20)
    plt.ylabel(col, fontsize=20)
    title = f'{part} 不同年份{col}的折线图对比'
    plt.title(title, fontsize=20);
    plt.xticks(fontsize=20);
    plt.yticks(fontsize=20);
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)

# 二维插值
def daily_interpolate(daily_attr_copy, x, y, z, path, part, n=300, step=None):
    xx, yy, zz = x, y, z
    x, y, z = daily_attr_copy[x].values, daily_attr_copy[y].values, daily_attr_copy[z].values
    upper, lower = 75, 25
    xmin, xmax = np.percentile(x, lower), np.percentile(x, upper)
    ymin, ymax = np.percentile(y, lower), np.percentile(y, upper) # 不直接用最大最小避免异常值
    newfunc = interpolate.interp2d(x, y, z, kind='cubic')
    if step is not None:
        X, Y = np.arange(xmin, xmax + step, step), np.arange(ymin, ymax + step, step)
    else:
        X, Y = np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n),
    Z = newfunc(X, Y)
    plt.figure(figsize=(20, 10))
    im2 = pl.imshow(
        Z, 
        extent=[xmin, xmax, ymin, ymax], 
        cmap=cm.jet_r, 
        interpolation='bilinear', 
        origin='lower',
        aspect='auto'
    )
    title = f'{part} Relationship between {zz} VS {xx} and {yy} after projection to a plane'
    pl.xlabel(xx, fontsize=20)
    pl.ylabel(yy, fontsize=20)
    pl.xticks(fontsize=20)
    pl.yticks(fontsize=20)
    pl.title(title, fontsize=20)
    cb = pl.colorbar(im2)
    file = _extracted_from_daily_interpolate_29(cb, path, title)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(20, 10))
    ax = Axes3D(fig)
    surf2 = ax.plot_surface(
        X, 
        Y, 
        Z, 
        rstride=1, 
        cstride=1, 
        cmap=cm.jet_r,
        linewidth=0.01, 
        antialiased=True
    )
    ax.set_xlabel(xx, fontsize=20)
    ax.set_ylabel(yy, fontsize=20)
    ax.set_zlabel(zz, fontsize=20)
    pl.xticks(fontsize=15)
    pl.yticks(fontsize=15)
    ax.tick_params(axis='z',labelsize=15)
    title = f'{part} Relationship between {zz} VS {xx} and {yy}'
    ax.set_title(title, fontsize=20)
    cb = plt.colorbar(surf2, shrink=0.5, aspect=5)
    file = _extracted_from_daily_interpolate_29(cb, path, title)


# TODO Rename this here and in `daily_interpolate`
def _extracted_from_daily_interpolate_29(cb, path, title):
    cb.ax.tick_params(labelsize=20)
    result = os.path.join(path, f'{title}.png')
    plt.savefig(result, dpi=100)

    return result

def flatten(x):
    res = []
    def flatten_helper(res, temp):
        try:
            for t in temp:
                flatten_helper(res, t)
        except:
            res.append(temp)
    for xx in x:
        flatten_helper(res, xx)
    return res

def get_max_sum_year(daily_attr, col, time_type):
    tmp = daily_attr.groupby([col, 'year'])[time_type].sum().reset_index()
    max_idx = tmp.groupby(col)[time_type].agg(lambda x: x[x == x.max()].index)
    return tmp.loc[flatten(max_idx)]

def get_max_mean_year(daily_attr, col, time_type):
    tmp = daily_attr.groupby([col, 'year'])[time_type].mean().reset_index()
    max_idx = tmp.groupby(col)[time_type].agg(lambda x: x[x == x.max()].index)
    return tmp.loc[flatten(max_idx)]

def plot_max_year(daily_attr, col, time_type, func, part, path):
    tmp = func(daily_attr, col, time_type)
    sts = func.__name__.split('_')[-2]
    # Fig 1
    plt.figure(figsize=(20, 10))
    plt.plot(tmp[col], tmp['year'])
    plt.xlabel(col, fontsize=20)
    plt.ylabel('year', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    title = f'{part} Plot of {col} that has the max {sts} value of {time_type} in each year'
    plt.title(title, fontsize=20)
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)
    # Fig 2
    plot_pie(
        tmp, 'year',
        4, part, path,
        f'that has the max {sts} value of {time_type} in each year'
    )
    return tmp


# 相似度热力图
def plot_sim_heatmap(df, col, part, path):
    min_lens = df[col].value_counts().min()
    df_copy = df.copy()
    df_copy['event_code'] = df_copy['event'].map(
        dict(zip(df_copy['event'].unique(), range(df_copy['event'].nunique())))
    )

    cos_sim = pd.DataFrame()
    for y1 in df[col].unique():
        for y2 in df[col].unique():
            t1 = df_copy.loc[df[col] == y1, 'event_code'].head(min_lens).values
            t2 = df_copy.loc[df[col] == y2, 'event_code'].head(min_lens).values
            cos_sim.loc[y1, y2] = 1 - spatial.distance.cosine(t1, t2)
    plt.figure(figsize=(20, 10))
    sns.heatmap(cos_sim, annot=True)
    title = f'{part} Cos similarity of events in each {col}'
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20);
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)



# 透视表
def get_pivot(df, item, layer, attr, ops):
    tmp = df.groupby([item, layer])[attr].agg(ops).to_frame('values').reset_index()
    my_pivot = pd.pivot_table(
        data=tmp, 
        index=item,
        columns=layer,
        values='values',
        fill_value=0
    )
    my_pivot.columns = [f'{my_pivot.columns.names[0]}_{attr}_pivot_{ops}_{str(col)}' for col in my_pivot.columns]

    return my_pivot

def get_keys(dic, threshold=.5):
    _sum = sum(list(dic.values()))
    res, key_sum = [], 0
    for k, v in sorted(dic.items(), key=lambda x:x[1], reverse=True):
        if key_sum / _sum <= threshold:
            res.append(k)
            key_sum += v
    return ' '.join(res)

def get_key_words(info, col, threshold=1):
    info = info.groupby(col)['event'].agg(list).reset_index()
    dct = Dictionary(info['event'].values)
    corpus = [dct.doc2bow(line) for line in info['event'].values] 

    # 用TfidfModel筛选关键词
    model = TfidfModel(corpus)
    movie_profile = {}
    for i, mid in enumerate(info[col]):
        vector = model[corpus[i]]
        movie_tag = sorted(vector, key=lambda x:x[1], reverse=True)
        movie_profile[mid] = dict(map(lambda x:(dct[x[0]], x[1]), movie_tag))

    info['key_words'] = [get_keys(dic, threshold) for _, dic in movie_profile.items()]   
    return info 

# 定义生成背景字样的函数
def WordPic(text, path):
    lens = 2000
    image = Image.new("RGB",(lens,lens),"white")
    draw_table = ImageDraw.Draw(im=image)
    draw_table.text(
        xy=(0, lens // 3), 
        text=str(text), 
        fill='#000000', 
        font=ImageFont.truetype('C:\\windows\\Fonts\\STHUPO.ttf', 450)
    )
    
    # image.show()  # 直接显示图片
    image.save(os.path.join(path, 'back.png'), 'PNG')  # 保存在当前路径下，格式为PNG
    image.close()

# 筛选关键词中的event做词云图
def plot_key_word_cloud(
        info, 
        col, 
        part,
        path='', 
        mask=None, 
        store=True,
        use_key_words=False
    ):
    info['plot_words'] = info.apply(
        lambda x: [e for e in x['event'] if e in x['key_words']],
        axis=1
    )
    temp = info[[col, 'plot_words']].itertuples(index=False) if use_key_words \
        else info[[col, 'event']].itertuples(index=False)
    for c, words in tqdm(temp):
        try:
            dpi = 500
            if mask is not None:
                dpi = 1500
                WordPic(c, path)
                mask = np.array(Image.open(os.path.join(path, 'back.png')))
            
            wordcloud = WordCloud(
                font_path='C:\\windows\\Fonts\\STHUPO.ttf', 
                background_color='white', mask=mask,
                collocations=False,
                scale=2
            ).fit_words(pd.Series(words).value_counts(normalize=True))
            plt.figure(dpi=200)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            if store:
                plt.savefig(
                    os.path.join(path, f'{part} 按{col}分类后{c}的关键词{str(mask)[:2]}.png'), 
                    dpi=dpi
                )
        except:
            continue

def sort_dic(dic:dict, reverse=True):
    temp = sorted(dic.items(), key=lambda x:x[1], reverse=reverse)
    res = '{'
    for k, v in temp:
        res += f'{k}:{v},'
    res += '}'
    return res

def net(df, col_from, col_to, year, event2attr, part, path, weight=None, dis=None, directed=False):
    if weight is None:
        df['weight'] = 1
        df = df.groupby([col_from, col_to])['weight'].sum().reset_index()
        df = df.rename(columns={
            col_from:'from', 
            col_to:'to'
        })
    else:
        df = df.rename(columns={
            col_from:'from', 
            col_to:'to',  
            weight:'weight'
        })
    if (df['weight'] < 1).all():
        df['weight'] *= 500

    if directed:
        GA = nx.from_pandas_edgelist(
            df, 
            source="from", 
            target="to", 
            edge_attr='weight', 
            create_using=nx.DiGraph()
        )
    else:
        GA = nx.from_pandas_edgelist(
            df, 
            source="from", 
            target="to", 
            edge_attr='weight', 
        )
    print(nx.info(GA))
    author_lst = df['from'].to_list() + df['to'].to_list()

    dic = {i: author_lst.count(i) for i in author_lst if author_lst.count(i) > 0}
    nodelst = []
    node_colors = []
    sizes = []
    node_color = {
        '有效时间':'r', 
        '浪费时间':'b',
        '必要时间':'g'
    }
    for m, k in dic.items():
        if node_color.get(event2attr.get(m)) is not None:
            node_colors.append(node_color.get(event2attr.get(m)))
        else:
            node_colors.append('m')
        nodelst.append(m)
        sizes.append(k * 1000)
    print('closeness_centrality:', sort_dic(nx.betweenness_centrality(GA)))
    plt.figure(figsize=(50, 50), dpi=50)
    pos = nx.drawing.kamada_kawai_layout(GA) if directed \
        else nx.drawing.layout.spring_layout(GA, iterations=200)

    labels = {m: m for m, k in dic.items() if k >= 1}
    #set the argument 'with labels' to False so you have unlabeled graph

    edges = GA.edges()
    weights = [GA[u][v]['weight'] / 10 for u,v in edges]

    nx.draw(
        GA,
        pos,
        with_labels=False,
        alpha=0.5,
        node_color=node_colors,
        nodelist=nodelst,
        node_size=sizes,
        font_family='SimHei',
    )

    #Now only add labels to the nodes you require (the hubs in my case)
    nx.draw_networkx_labels(
        GA,
        pos,
        labels,
        font_family='SimHei',
        font_size=40,
        font_color='black'
    )

    nx.draw_networkx_edges(
        GA,
        pos,
        # part,
        width=weights,
        edge_color='c',
        alpha=0.5,
        arrowsize=100,
        arrows=True,
    )
    title = f'{part} {col_from}与{col_to}在{year}的网络图'
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=50)

# 月份与事项的关系图
def month2event(df, n, event2attr, part, path):
    assert n in range(2019, 2023)
    n = str(n)
    temp = df[df['year'] == n].copy() 
    month_event = temp.groupby(['month', 'event'])['duration'].count().reset_index()
    net(month_event, 'month', 'event', n, event2attr, part, path, weight='duration', directed=False, dis=3)

# 前后事项的网络图
def PostEvent(df, n, event2attr, part, path):
    assert n in range(2019, 2023)
    n = str(n)
    temp = df[df['year'] == n].copy() 
    pro_post_ass = temp[['date', 'event']]
    pro_post_ass['previous_event'] = temp.groupby(['date'])['event'].shift()
    pro_post_ass['post_event'] = temp.groupby(['date'])['event'].shift(-1)
    pro_post_ass = pro_post_ass.dropna()    
    net(pro_post_ass, 'event', 'post_event', n, event2attr, part, path, directed=True, dis=5)