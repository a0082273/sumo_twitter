import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import MeCab
import re
from collections import Counter
from wordcloud import WordCloud
import sys



def make_basho_tweets(rikishi):
    print('make_rikishi_data')
    ym = '2018-07-'
    day = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    one_day = []

    for d in day:
        ymd = ym+d
        infilename = f'input/{rikishi}{ymd}.csv'
        one_day.append(pd.read_csv(open(infilename, 'rU'), encoding='utf-8'))
    basho = pd.concat(one_day, axis=0)
    basho = basho.reset_index(drop=True)
    return basho



def exclude_inappropriate_data(rikishi, data):
    print('exclude_inappropriate_data')
    data = data.drop('Unnamed: 0', axis=1)
    data = exclude_nan_data(data)
    data = exclude_bot_data(data)
    data.to_csv(f'output/{rikishi}_tweets.csv')
    return data



def exclude_nan_data(data):
    improper_col = []
    for col in range(data.shape[0]):
        if type(data.loc[col, 'text']) != str:
            improper_col.append(col)
    print('num of improper col:', len(improper_col))
    data = data.drop(improper_col, axis=0)
    data = data.reset_index(drop=True)
    return data


def exclude_bot_data(data):
    improper_names  = [
        'bot', 'Bot', 'BOT', 'ぼっと', '情報', '案内', '法人', 'News', 'NEWS', 'news',
        'ニュース', '新聞', '報道', '沖縄タイムス', 'オンリーワンダースポーツカフェ'
        # '相互', '出会', 'セフレ', 'エッチ', '相撲 バズウォール', '逢華', '大西啓太',
        # 'バカボンのパパ', '久保寺健之', 'チケット×チケット'
    ]
    improper_profiles  = ['improper_words', 'bot', 'Bot', 'BOT', 'ぼっと', '公式ツイッター',
                          '公式アカウント']
    improper_texts  = [
        'improper_words', 'snd', '白鵬女子', '孤高の荒鷲', '荒鷲よ翔べ', 'イーグルシャウト',
        'アニカフェ', 'SideM','缶バッジ', 'かのん', 'ニシキヘビ', '盆栽', '雑木'
        # 'ライブ', '糸千代丸', '志倉千代丸'
        # "I'm at", '相互', 'imacoconow', '火ノ丸', '大喜利',
        # '手押し相撲' , '一人相撲' , '独り相撲', 'ローション', '名言集', '紙相撲',
        # '火の丸', 'カーネーション', '他人の褌で相撲を取る', '菊とギロチン', '伝令',
        # '格闘技、プロ野球、相撲好き', '女相撲', '氷結相撲', '尻相撲', 'とんとん相撲',
        # 'トントン相撲', 'バブル相撲', 'ストッキング相撲', 'ちびっこ相撲', 'モンゴル相撲'
    ]

    for col in range(data.shape[0]):
        if type(data.loc[col, 'name']) == str:
            for word in improper_names:
                if word in data.loc[col, 'name']:
                    data.loc[col, 'text'] = 'improper_words'
                    break
        if type(data.loc[col, 'profile']) == str:
            for word in improper_profiles:
                if word in data.loc[col, 'profile']:
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
    word_list = remove_stop_words(word_list)
    return word_list


def make_word_list(data):
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
    word_list = []
    for i in range(data.shape[0]):
        if type(data.text[i]) == str:
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

    for word in stop_words:
        remove_specified_values(word_list, word)
    return word_list


def remove_specified_values(arr, value):
    while value in arr:
        arr.remove(value)



def make_word_cloud(rikishi, word_list):
    print('make_word_cloud')
    word_list = ' '.join(word_list)

    fpath = "~/Library/Fonts/RictyDiminished-Regular.ttf"
    wordcloud = WordCloud(background_color="white", font_path=fpath, width=900, height=500, max_words=80).generate(word_list)

    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(f'output/{rikishi}_wordcloud.png')
    plt.show()



# def make_words_summary(rikishi, ymd, word_list, summary):
#     print('make_summary')
#     counter = Counter(word_list)
#     words, counts = zip(*counter.most_common(100))
#     summary_1d = pd.Series(data=counts, index=words, name=ymd)
#     if ymd == '2018-07-08':
#         summary = summary_1d.copy()
#     else:
#         summary = pd.concat([summary, summary_1d], axis=1)
#     # summary_1d.to_csv(f'output/{rikishi}{ymd}_words.csv')
#     if ymd == '2018-07-22':
#         summary.to_csv(f'output/{rikishi}_words.csv')
#     return summary




def make_rikishi_data(rikishi):
    print('make_rikishi_data')

    day = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    infilename = f'output/{rikishi}_tweets.csv'
    rikishi_tweets = pd.read_csv(open(infilename, 'rU'), encoding='utf-8')

    ntweets = rikishi_ntweets(rikishi_tweets, day) #rikishiを含むツイートの総数
    following = rikishi_following(rikishi_tweets)
    followed = rikishi_followed(rikishi_tweets)
    n_tweets = rikishi_n_tweets(rikishi_tweets) #rikishiを含むツイートをしたアカウントのこれまでのツイート総数
    favorited = rikishi_favorited(rikishi_tweets)
    rikishi_data = pd.concat([ntweets, following, followed, n_tweets, favorited], axis=0)
    return rikishi_data


def rikishi_ntweets(rikishi_tweets, day):
    ntweets = []
    for d in day:
        cnt = list(rikishi_tweets['time'].str.contains(f' {d} ').values).count(True)
        ntweets.append(cnt)
    ntweets.append(rikishi_tweets.shape[0])
    index = ['ntweets_day'+str(i) for i in range(1, 16)]
    index.append('ntweets_sum')
    rikishi_data = pd.Series(data=ntweets, index=index, name=rikishi)
    return rikishi_data


def rikishi_following(rikishi_tweets):
    index = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    following = []
    for ind in index:
        following.append(rikishi_tweets['n_following'].describe()[ind])
    index = ['following_'+ind for ind in index]
    rikishi_data = pd.Series(data=following, index=index, name=rikishi)
    return rikishi_data


def rikishi_followed(rikishi_tweets):
    index = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    followed = []
    for ind in index:
        followed.append(rikishi_tweets['n_followed'].describe()[ind])
    index = ['followed_'+ind for ind in index]
    rikishi_data = pd.Series(data=followed, index=index, name=rikishi)
    return rikishi_data


def rikishi_n_tweets(rikishi_tweets):
    index = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    n_tweets = []
    for ind in index:
        n_tweets.append(rikishi_tweets['n_tweets'].describe()[ind])
    index = ['n_tweets_'+ind for ind in index]
    rikishi_data = pd.Series(data=n_tweets, index=index, name=rikishi)
    return rikishi_data


def rikishi_favorited(rikishi_tweets):
    index = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    favorited = []
    for ind in index:
        favorited.append(rikishi_tweets['n_favorited'].describe()[ind])
    index = ['favorited_'+ind for ind in index]
    rikishi_data = pd.Series(data=favorited, index=index, name=rikishi)
    return rikishi_data



def rikishi_ability():
    ability = pd.read_csv('input/201807_bt_ability.csv', index_col=0)
    ability = ability.drop('s.e.', axis=1)
    ability = ability.T
    return ability





if __name__ == '__main__':
    rikishi_list = [
        'tsururyuu', 'shiroootori', 'mareseinosato', 'kusamafuji', 'goueidou', 'takayasu',
        'tochinokokoro', 'ontakeumi', 'tamawashi', 'matsuootoriyama', 'shoudai', 'konshoukiku',
        'chiyonokuni', 'ahonoo', 'takashikeishou', 'kaihijiri', 'daishoumaru', 'kakaze',
        'chiyodairyuu', 'takarafuji', 'daieishou', 'chiyoshouuma', 'asahidaihoshi', 'myougiryuu',
        'yutakayama', 'nishikiki', 'hekiyama', 'abusaki', 'sadanoumi', 'kouwashi', 'tochikouyama',
        'asanoyama', 'konmegumihikari', 'okinoumi', 'ryuuden', 'kitakachifuji', 'hattorisakura'
        ### 'endo', 'ikioi', 'chiyomaru', 'akiumi', 'ishiura'
    ]
    rikishi_kanji = [
        '鶴竜', '白鵬', '稀勢の里', '日馬富士', '豪栄道', '高安',
        '栃ノ心', '御嶽海', '玉鷲', '松鳳山', '正代', '琴奨菊',
        '千代の国', '阿炎', '貴景勝', '魁聖', '大翔丸', '嘉風',
        '千代大龍', '宝富士', '大栄翔', '千代翔馬', '旭大星', '妙義龍',
        '豊山', '錦木', '碧山', '阿武咲', '佐田の海', '荒鷲', '栃煌山',
        '朝乃山', '琴恵光', '隠岐の海', '竜電', '北勝富士', '服部桜'
    ]
    rikishi_dict = dict(zip(rikishi_list, rikishi_kanji))
    all_rikishi_data = pd.DataFrame(columns=rikishi_list)

    for rikishi in rikishi_list:
        print('-'*40)
        print(rikishi)
        # basho = make_basho_tweets(rikishi)
        # basho = exclude_inappropriate_data(rikishi, basho)
        # word_list = words_with_keyword(basho)
        # make_word_cloud(rikishi, word_list)
        ## summary = make_words_summary(rikishi, ymd, word_list, summary)
        rikishi_data = make_rikishi_data(rikishi)
        all_rikishi_data[rikishi] = rikishi_data

    all_rikishi_data = all_rikishi_data.rename(columns=rikishi_dict)
    ability = rikishi_ability()
    all_rikishi_data = pd.concat([all_rikishi_data, ability], axis=0)
    all_rikishi_data.to_csv('output/201807_all_data.csv')
