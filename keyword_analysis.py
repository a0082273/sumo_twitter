import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import MeCab
import re
from collections import Counter
from wordcloud import WordCloud
import sys



def make_1d_data(ymd):
    print('make_1d_data')
    infilename = f'input/sumou{ymd}.csv'
    one_day = pd.read_csv(open(infilename, 'rU'), encoding='utf-8')
    one_day = one_day.drop('Unnamed: 0', axis=1)
    return one_day



def exclude_inappropriate_data(data, ymd):
    print('exclude_inappropriate_data')
    data = exclude_nan_data(data)
    data = exclude_bot_data(data)
    data.to_csv(f'output/{ymd}_tweets.csv')
    return data



def exclude_nan_data(data):
    improper_col = []
    for col in range(data.shape[0]):
        # if type(data.loc['text', col]) != str:
        if type(data.loc[col, 'text']) != str:
            # print('-'*20)
            # print(col)
            # print(type(data.name[col]), type(data.profile[col]), type(data.text[col]),)
            # print(data.iloc[col])
            improper_col.append(col)
    print('num of improper col:', len(improper_col))
    data = data.drop(improper_col, axis=0)
    data = data.reset_index(drop=True)
    return data


def exclude_bot_data(data):
    improper_names  = ['bot', 'Bot', 'BOT', 'ぼっと', '情報', '案内', '相互', '出会', 'セフレ',
                       'エッチ', '法人', '相撲 バズウォール', '逢華', '大西啓太', 'News', 'NEWS',
                       'news', 'ニュース', '新聞', '報道', 'バカボンのパパ', '久保寺健之',
                       '沖縄タイムス', 'チケット×チケット']
    improper_profiles  = ['improper_words', 'bot', 'Bot', 'BOT', 'ぼっと', '公式ツイッター',
                          '公式アカウント']
    improper_texts  = ['improper_words', "I'm at", '相互', 'imacoconow', '火ノ丸', '大喜利',
                       '手押し相撲' , '一人相撲' , '独り相撲', 'ローション', '名言集', '紙相撲',
                       '火の丸', 'カーネーション', '他人の褌で相撲を取る', '菊とギロチン', '伝令',
                       '格闘技、プロ野球、相撲好き', '女相撲', '氷結相撲', '尻相撲', 'とんとん相撲',
                       'トントン相撲', 'バブル相撲', 'ストッキング相撲', 'ちびっこ相撲']

    for col in range(data.shape[0]):
        if type(data.loc[col, 'name']) == str:
            for word in improper_names:
                if word in data.loc[col, 'name']:
                    # data['profile'][col] = 'improper_words'
                    data.loc[col, 'text'] = 'improper_words'
                    # data.loc[col, 'profile'] = 'improper_words'
                    break
        if type(data.loc[col, 'profile']) == str:
            for word in improper_profiles:
                if word in data.loc[col, 'profile']:
                    # data['text'][col] = 'improper_words'
                    data.loc[col, 'text'] = 'improper_words'
                    break
        for word in improper_texts:
            if word in data.loc[col, 'text']:
                data = data.drop(col, axis=0)
                break

    data = data.reset_index(drop=True)
    return data



def words_with_keyword(data):
    print('words_with_keyword')
    word_list = make_word_list(data)
    # counter = Counter(word_list)
    # for word, cnt in counter.most_common():
    #     print(word, cnt)

    word_list = remove_stop_words(word_list)
    # counter = Counter(word_list)
    # for word, cnt in counter.most_common(10):
    #     print(word, cnt)
    return word_list



def make_word_list(data):
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
    word_list = []
    for i in range(data.shape[0]):
        if type(data.text[i]) == str:
            # texts = m.parse(data.loc['text', i])
            texts = m.parse(data.loc[i, 'text'])
            texts = texts.split('\n')
            for text in texts:
                text = re.split('[\t,]', text)
                if text[0] == 'EOS' or text[0] == '':
                    pass
                elif text[1] == '名詞':
                    word_list.append(text[0])
    return word_list



def remove_stop_words(word_list):
    stop_words = [
        'in', 'ー', 'bot', 'https', 'co', 'ない', '無い', '投稿', 'ツイート', '今日', '明日',
        'さん', 'こと', 'よう', 'それ', 'どこ', 'これ', 'みたい', '名前', '自分', 'ちゃん', 'そう',
        '登録', 'くん', 'あと', 'そこ', 'ため', 'うち', 'ここ', 'ところ', 'なん', '感じ', '情報',
        'もの', 'とき', 'やつ', 'もん', 'しよう', 'わけ', 'たち', 'とこ', 'つもり', 'こちら',
        'しんみ', 'した', 'せい', 'さま', 'さっき', 'こっち', 'かな', 'まま', '最近', '時間',
        '場所', '本日', '付近', 'よろしくお願いします', '昨日', '今週', '来週', '先週', 'みんな',
        '相撲', '名古屋場所', 'sumo', '大相撲', '力士', '中継', '相撲部', '相撲取り',
        '大相撲名古屋場所','相手', '土俵', '？？？', 'あれ', '近く', 't', 'の', 'ん', 'w',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '０', '１', '２','３', '４',
        '５', '６', '７', '８', '９', '方', '何', '中', '笑', '今', 'さ', '時', '関', '事',
        '気', 'ニュース', '話', 'www', 'ω', '的', '目', '様', '後', '俺', 'スーモ', '日',
        '手', '次', '観', '君', '…。', '山', '感', '回', 'ww', 'SD', '差'
    ]

    For word in stop_words:
        remove_specified_values(word_list, word)
    return word_list



def remove_specified_values(arr, value):
    while value in arr:
        arr.remove(value)



def make_word_cloud(ymd, word_list):
    print('make_word_cloud')
    word_list = ' '.join(word_list)

    fpath = "~/Library/Fonts/RictyDiminished-Regular.ttf"
    wordcloud = WordCloud(background_color="white", font_path=fpath, width=900, height=500, max_words=80).generate(word_list)

    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(f'output/{ymd}.png')
    plt.show()



def make_summary(ymd, word_list, summary):
    print('make_summary')
    counter = Counter(word_list)
    words, counts = zip(*counter.most_common(100))
    summary_1d = pd.Series(data=counts, index=words, name=ymd)
    if ymd == '2018-07-08':
        summary = summary_1d.copy()
    else:
        summary = pd.concat([summary, summary_1d], axis=1)
    summary_1d.to_csv(f'output/{ymd}.csv')
    summary.to_csv(f'output/2018-07.csv')
    return summary



if __name__ == '__main__':
    key_word = 'somou' #相撲
    ym = '2018-07-'
    day = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    infile = []
    summary = np.nan
    for d in day:
        ymd = ym+d
        print('-'*40)
        print(ymd)
        one_day = make_1d_data(ymd)
        one_day = exclude_inappropriate_data(one_day, ymd)
        word_list = words_with_keyword(one_day)
        # make_word_cloud(ymd, word_list)
        summary = make_summary(ymd, word_list, summary)
