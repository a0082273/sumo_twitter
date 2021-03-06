# -*- coding: utf-8 -*-

from requests_oauthlib import OAuth1Session
import json
import datetime, time, sys
from abc import ABCMeta, abstractmethod
import csv
import pandas as pd
from pykakasi import kakasi
import setting

CK = setting.CK
CS = setting.CS
AT = setting.AT
# AS = setting.AS
AS = 'b9rC3W8LlNQrvrNKkSdlVNHrYcfFsQ1q47AhhD9A63h5T'


class TweetsGetter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.session = OAuth1Session(CK, CS, AT, AS)

    @abstractmethod
    def specifyUrlAndParams(self, keyword):
        '''
        呼出し先 URL、パラメータを返す
        '''

    @abstractmethod
    def pickupTweet(self, res_text, includeRetweet):
        '''
        res_text からツイートを取り出し、配列にセットして返却
        '''

    @abstractmethod
    def getLimitContext(self, res_text):
        '''
        回数制限の情報を取得 （起動時）
        '''

    def collect(self, total = -1, onlyText = False, includeRetweet = False):
        '''
        ツイート取得を開始する
        '''

        #----------------
        # 回数制限を確認
        #----------------
        self.checkLimit()

        #----------------
        # URL、パラメータ
        #----------------
        url, params = self.specifyUrlAndParams()
        params['include_rts'] = str(includeRetweet).lower()
        # include_rts は statuses/user_timeline のパラメータ。search/tweets には無効

        #----------------
        # ツイート取得
        #----------------
        cnt = 0
        unavailableCnt = 0
        while True:
            res = self.session.get(url, params = params)
            if res.status_code == 503:
                # 503 : Service Unavailable
                if unavailableCnt > 10:
                    raise Exception('Twitter API error %d' % res.status_code)

                unavailableCnt += 1
                print ('Service Unavailable 503')
                self.waitUntilReset(time.mktime(datetime.datetime.now().timetuple()) + 30)
                continue

            unavailableCnt = 0

            if res.status_code != 200:
                raise Exception('Twitter API error %d' % res.status_code)

            tweets = self.pickupTweet(json.loads(res.text))
            if len(tweets) == 0:
                # len(tweets) != params['count'] としたいが
                # count は最大値らしいので判定に使えない。
                # ⇒  "== 0" にする
                # https://dev.twitter.com/discussions/7513
                break

            for tweet in tweets:
                if (('retweeted_status' in tweet) and (includeRetweet is False)):
                    pass
                else:
                    if onlyText is True:
                        yield tweet['text']
                    else:
                        yield tweet

                    cnt += 1
                    if cnt % 100 == 0:
                        print ('%d件 ' % cnt)

                    if total > 0 and cnt >= total:
                        return

            params['max_id'] = tweet['id'] - 1

            # ヘッダ確認 （回数制限）
            # X-Rate-Limit-Remaining が入ってないことが稀にあるのでチェック
            if ('X-Rate-Limit-Remaining' in res.headers and 'X-Rate-Limit-Reset' in res.headers):
                if (int(res.headers['X-Rate-Limit-Remaining']) == 0):
                    self.waitUntilReset(int(res.headers['X-Rate-Limit-Reset']))
                    self.checkLimit()
            else:
                print ('not found  -  X-Rate-Limit-Remaining or X-Rate-Limit-Reset')
                self.checkLimit()

    def checkLimit(self):
        '''
        回数制限を問合せ、アクセス可能になるまで wait する
        '''
        unavailableCnt = 0
        while True:
            url = "https://api.twitter.com/1.1/application/rate_limit_status.json"
            res = self.session.get(url)

            if res.status_code == 503:
                # 503 : Service Unavailable
                if unavailableCnt > 10:
                    raise Exception('Twitter API error %d' % res.status_code)

                unavailableCnt += 1
                print ('Service Unavailable 503')
                self.waitUntilReset(time.mktime(datetime.datetime.now().timetuple()) + 30)
                continue

            unavailableCnt = 0

            if res.status_code != 200:
                raise Exception('Twitter API error %d' % res.status_code)

            remaining, reset = self.getLimitContext(json.loads(res.text))
            if (remaining == 0):
                self.waitUntilReset(reset)
            else:
                break

    def waitUntilReset(self, reset):
        '''
        reset 時刻まで sleep
        '''
        seconds = reset - time.mktime(datetime.datetime.now().timetuple())
        seconds = max(seconds, 0)
        print ('\n     =====================')
        print ('     == waiting %d sec ==' % seconds)
        print ('     =====================')
        sys.stdout.flush()
        time.sleep(seconds + 10)  # 念のため + 10 秒

    @staticmethod
    def bySearch(keyword):
        return TweetsGetterBySearch(keyword)


class TweetsGetterBySearch(TweetsGetter):
    '''
    キーワードでツイートを検索
    '''
    def __init__(self, keyword):
        super(TweetsGetterBySearch, self).__init__()
        self.keyword = keyword

    def specifyUrlAndParams(self):
        '''
        呼出し先 URL、パラメータを返す
        '''
        url = 'https://api.twitter.com/1.1/search/tweets.json'
        params = {'q':self.keyword, 'count':100}
        return url, params

    def pickupTweet(self, res_text):
        '''
        res_text からツイートを取り出し、配列にセットして返却
        '''
        results = []
        for tweet in res_text['statuses']:
            results.append(tweet)

        return results

    def getLimitContext(self, res_text):
        '''
        回数制限の情報を取得 （起動時）
        '''
        remaining = res_text['resources']['search']['/search/tweets']['remaining']
        reset     = res_text['resources']['search']['/search/tweets']['reset']

        return int(remaining), int(reset)



def makeNewCol(tweet):
    date_time = (tweet["created_at"])
    id = (tweet["user"]["screen_name"])
    name = (tweet["user"]["name"])
    profile = (tweet["user"]["description"])
    n_following = (tweet["user"]["friends_count"])
    n_followed = (tweet["user"]["followers_count"])
    n_tweets = (tweet["user"]["statuses_count"])
    addres = (tweet["user"]["location"])
    n_favorited = (tweet["favorite_count"])
    text = (tweet["text"])

    new_col = pd.Series([date_time, id, name, profile, n_following,
                         n_followed, n_tweets, addres, n_favorited, text],
                        index=["time", "id", "name", "profile", "n_following",
                               "n_followed", "n_tweets", "adress", "n_favorited","text"])





if __name__ == '__main__':
    kakasi = kakasi()
    kakasi.setMode('H', 'a')
    kakasi.setMode('K', 'a')
    kakasi.setMode('J', 'a')
    conv = kakasi.getConverter()

    keyword_list = [
                    # '鶴竜', '白鵬', '稀勢の里', '日馬富士',
                    # '豪栄道', '高安', '栃ノ心',
                    # '御嶽海', '玉鷲', '松鳳山',
                    # '正代', '琴奨菊', '千代の国', '阿炎', '貴景勝', '魁聖',
                    # '大翔丸', '嘉風', '千代大龍', '宝富士', '大栄翔',
                    # '千代翔馬', '旭大星', '妙義龍', '豊山', '千代丸',
                    # '錦木', '碧山', '阿武咲', '佐田の海', '荒鷲', '栃煌山',
                    # '朝乃山', '琴恵光', '隠岐の海', '石浦', '竜電',
                    # '北勝富士', '明生', '服部桜',
                    # '相撲',
                    # '#sumo', '#相撲', '#大相撲', '#名古屋場所'
        '妙義龍'
    ]

    for keyword in keyword_list:
        print(keyword)
        keyword_romaji = conv.do(keyword)
        # since = '2018-07-15'
        # until = '2018-07-16'
        since = '2018-08-15'
        until = '2018-08-16'
        getter = TweetsGetter.bySearch(keyword+' since:'+since+' until:'+until)

        df = pd.DataFrame(columns=["time", "id", "name", "profile", "n_following",
                                   "n_followed", "n_tweets", "adress", "n_favorited","text"])
        for tweet in getter.collect(total = 20000):
            new_col = makeNewCol(tweet)
            df.append(new_col, ignore_index=True)

        df.to_csv('output/'+keyword_romaji+since+'.csv')
