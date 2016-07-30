import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from scipy import sparse
import pickle


class Morp:
    '''
    点予測を利用して、日本語の単語分割を行う。　
    学習, 推定をこれで行う.
    '''
    def __init__(self, word_dict=None, estimator = LinearSVC(C=1.0)):  # 多分データ量が多くなってきたらSVGとかにする?
        self.name = ""
        self.word_dict = word_dict
        self.estimator = estimator
        self.unigram_dict = {}
        self.bigram_dict = {}
        self.trigram_dict = {}
        self.type_dict = self.get_type_dict()  # 6**3=216

    def word_segment(self, text):
        '''
        入力されたtextを単語分割して返す.
        '''
        text = text.strip()
        if len(text) < 2:  # 分割する必要なし
            return text
        chars = list(text)
        output_chars = chars[0]
        for pointer in range(1, len(chars)):  # その文字の右側に空白があるかどうかを判定
            feature = self.get_feature(pointer, chars)
            binary_boundary = self.decision_boundary(feature)
            if binary_boundary == 0:  # 単語分割なし
                output_chars = output_chars + chars[pointer]
            else:  # 単語分割を行う
                output_chars = output_chars + ' ' + chars[pointer]
        return output_chars

    def train(self, text_path_list): 
        '''
        引数にpath_listを与えると、学習を行う。(path単体でも一応動作する)
        '''
        if type(text_path_list) == str:  # pathが一つの時
            text_path_list = [text_path_list]  # 要素数1のlistに
        if type(text_path_list) == list:  # pathが複数
            text = ""
            for text_path in text_path_list:
                for line in open(text_path, 'r'):  # 全ての文字を繋げたtextを生成
                    line = line.strip().replace(" ", "")
                    line = '^^^'+line+'$$$'  # 3-gram用
                    text = text + line
            
            char_number = 0
            for n, dict in zip(range(1, 4), [self.unigram_dict , self.bigram_dict, self.trigram_dict]):  # 1gram~3gram
                dict['UNK'] = char_number
                chars = self.ngram(text, n)
                for char in chars:  # 辞書を作る
                    if char in dict:
                        continue
                    else:
                        dict[char] = char_number
                    char_number += 1
                cahr_number = 0  # reset

            first_flag = 1
            for text_path in text_path_list:
                if first_flag == 1: # 初回用
                    total_feature, total_teacher = self.train_file(text_path)
                    first_flag = 0
                else:
                    feature, teacher = self.train_file(text_path)
                    total_feature = sparse.vstack((total_feature, feature))
                    total_teacher = np.hstack((total_teacher, teacher))
        total_feature = total_feature.todense()
        print(total_feature.shape, total_teacher)
        self.estimator.fit(total_feature, total_teacher)
        return 0

    def ngram(self, text, n):  # n-gramを取ってくるコード(http://blog.livedoor.jp/yawamen/archives/51513695.htmlから引用)
        results = []
        if len(text) >= n:
            for i in range(len(text)-n+1):
                results.append(text[i:i+n])
        return results

    def train_file(self, text_path):  # ファイル一つから学習を行う.
        first_flag = 1  # 学習用のarrayの初期用
        line_number = 0
        for line in open(text_path, 'r'):  # 学習データ生成
            line = line.strip()
            if len(line) < 2:  # 学習するところがないなら
                continue
            print(line)
            features, teacher = self.train_text(line)
            feature_array = sparse.csr_matrix(self.make_feature_array(features, len(teacher)))  # data_sizeは教師データのsizeから求めてる
            if first_flag == 1:  # 初回用
                total_feature = feature_array
                total_teacher = teacher
                first_flag = 0
            else:  # ここが計算量のneckになっている...??  # つーかメモリの量の問題な気もする....(500文で,10000次元ぐらいのベクトル)  # scipyを入れて多少はマシに?
                total_feature = sparse.vstack((total_feature, feature_array))
                total_teacher = np.hstack((total_teacher, teacher))
            line_number += 1
        return total_feature, total_teacher

    def get_feature(self, pointer, chars):  # featureを取ってくる関数
        # 文字
        l1 = chars[pointer-1]  # 左1文字目
        if pointer-2 < 0:
            l2 = '^'
        else:
            l2 = chars[pointer-2]
        if pointer-3 < 0:
            l3 = '^'
        else:
            l3 = chars[pointer-3]
        r1 = chars[pointer]  # 右1文字目
        if pointer+1 > len(chars)-1:
            r2 = '$'
        else:
            r2 = chars[pointer+1]
        if pointer+2 > len(chars)-1:
            r3 = '$'
        else:
            r3 = chars[pointer+2]
        # bigram
        l3l2 = l3 + l2
        l2l1 = l2 + l1
        l1r1 = l1 + r1
        r1r2 = r1 + r2
        r2r3 = r2 + r3
        # trigram
        l3l2l1 = l3 + l2 + l1
        l2l1r1 = l2 + l1 + r1
        l1r1r2 = l1 + r1 + r2
        r1r2r3 = r1 + r2 + r3
        # type_unigram
        tl1 = self.get_types(l1)
        tl2 = self.get_types(l2)
        tl3 = self.get_types(l3)
        tr1 = self.get_types(r1)
        tr2 = self.get_types(r2)
        tr3 = self.get_types(r3)
        # type_bigram
        tl3l2 = self.get_types(l3 + l2)
        tl2l1 = self.get_types(l2 + l1)
        tl1r1 = self.get_types(l1 + r1)
        tr1r2 = self.get_types(r1 + r2)
        tr2r3 = self.get_types(r2 + r3)
        # type_trigram
        tl3l2l1 = self.get_types(l3 + l2 + l1)
        tl2l1r1 = self.get_types(l2 + l1 + r1)
        tl1r1r2 = self.get_types(l1 + r1 + r2)
        tr1r2r3 = self.get_types(r1 + r2 + r3)
        # 辞書  # dictを読んで、現在のpoint直前で終わる(f) / point直後から単語が始める(s) / pointの上を辞書がまたぐ(o).
        f, s, o = 0, 0, 0
        if not self.word_dict is None:  # 辞書を持っている時
            f, s, o = self.get_dict_feature(pointer, chars)
        feature = [l1, l2, l3, r1, r2, r3, l3l2, l2l1, l1r1, r1r2, r2r3, l3l2l1, l2l1r1, l1r1r2, r1r2r3, tl1, tl2, tl3, tr1, tr2, tr3, tl3l2, tl2l1, tl1r1, tr1r2, tr2r3, tl3l2l1, tl2l1r1, tl1r1r2, tr1r2r3, f, s, o]  # ここ長すぎるから望ましくないよね
        return feature

    def decision_boundary(self, feature):  # 単語分割を行うかどうか(word_segmentに併合?)
        feature_array = self.make_feature_array([feature], 1)
        prediction = self.estimator.predict(feature_array)
        return prediction[0]

#    Feature説明
#    [0,1,2,3,4,5||6,7,8,9,10,||11,12,13,14||15,16,17,18,19,20,||21,22,23,24,25||26,27,28,29||30,31,32]
#         unigram    bigram       trigram     type-unigram       type_bigram     type_trigram   dict
#    変換後の次元数: n-gram: 6*unigram_dict_length +5*bigram_dict_length +4*trigram_dict_length | type_unigram, type_bigram, type_trigram: (6+5+4)*type_dict_length | dict: 3

    def make_feature_array(self, features, data_size):  # featuresを投げるとarrayを返す  # sparseで作ってCSRに変換の方が早いかも...
        feature_array = np.zeros([data_size, 6*len(self.unigram_dict) + 5*len(self.bigram_dict) + 4*len(self.trigram_dict) + 15*len(self.type_dict) + 3])  # 行はデータサイズ、列は素性の次元
        line_number = 0
        for feature in features:
            pre_size = 0  # 左橋から埋めていく
            for char, number in zip(feature[:5], range(1, 7)):  #  unigram
                try:
                    feature_array[line_number][number*self.unigram_dict[char]] = 1  # 既知
                except:
                    feature_array[line_number][number*self.unigram_dict['UNK']] = 1  # UNK
            pre_size = 6*len(self.unigram_dict)
            for char, number in zip(feature[6:10], range(1, 6)):  #  bigram
                try:
                    feature_array[line_number][pre_size + number*self.bigram_dict[char]] = 1
                except:
                    feature_array[line_number][pre_size + number*self.bigram_dict['UNK']] = 1
            pre_size = 6*len(self.unigram_dict) + 5*len(self.bigram_dict)
            for char, number in zip(feature[11:14], range(1, 5)):  #  trigram
                try:
                    feature_array[line_number][pre_size + number*self.trigram_dict[char]] = 1
                except:
                    feature_array[line_number][pre_size + number*self.trigram_dict['UNK']] = 1
            pre_size = 6*len(self.unigram_dict) + 5*len(self.bigram_dict) + 4*len(self.trigram_dict)

            for type, number in zip(feature[15:29], range(1, 17)):  # type_unigram, type_bigram, type_trigram
                feature_array[line_number][pre_size + self.type_dict[type]] = 1  # typeはunigram, bigram, trigram がそれぞれ216次元持っておりUNKは存在しない
            pre_size = 6*len(self.unigram_dict) + 5*len(self.bigram_dict) + 4*len(self.trigram_dict) + 15*len(self.type_dict)

            for number in range(30, 33):  # dict
                feature_array[line_number][pre_size + number -30] = feature[number]
            line_number += 1
        return feature_array

    def get_type_dict(self):  # インスタンス作成時にtype_dictを作成する
        type_dict = {}
        dict_number = 0
        for number1 in range (0, 6):
            type_dict[str(number1)] = dict_number
            dict_number += 1
            for number2 in range(0, 6):
                type_dict[str(number1)+str(number2)] = dict_number
                dict_number += 1
                for number3 in range(0, 6):
                    type_dict[str(number1)+str(number2)+str(number3)] = dict_number
                    dict_number += 1
        return type_dict

    def get_types(self, chars):  # 複数の文字列をself.type_dictに照らし合わせて、番号をもらう
        type = ''
        for char in chars:
            type = type+str(self.get_type(char))
            return type

    def get_type(self, char):  # 文字種判別. ひらがな=0, カタカナ=1, 漢字=2, Alphabet=3, 数字=4, その他=5
        if 'ぁ' <= char <= 'ん':
            return 0
        if 'ァ' <= char <= 'ﾝ' and not '亜' <= char <= '話':
            return 1
        if '亜' <= char <= '話':  # "雨"とかだと失敗するなぜだ...?
            return 2
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            return 3
        if char.isdigit():
            return 4
        return 5

    def get_dict_feature(self, pointer, chars): # dictを読んで、現在のpoint直前で終わる(f) / point直後から単語が始める(s) / pointの上を辞書がまたぐ(o).
        # ウルトラ汚いので整備必要(一応動くはず)
        f, s, o = 0, 0, 0
        cand = chars[pointer-1]  # 左1文字目
        for number in range(0, pointer):  # f
            if cand in self.word_dict:
                f = 1
                break
            if pointer-number-2 < 0:
                break
            cand = chars[pointer-number-2] + cand  # 左に文字を追加 
        cand = chars[pointer]  # 右1文字目
        for char in chars[pointer+1:]:  # s
            if cand in self.word_dict:
                s = 1
                break
            cand = cand + char
        if cand in self.word_dict:
            s = 1
        cand = chars[pointer-1] + chars[pointer]
        cand_l = cand  # 左側に増やす
        for number in range(0, pointer):  # o
            for char in chars[pointer+1:]:
                if cand in self.word_dict:
                    o = 1
                    break
                cand = cand + char
            if cand in self.word_dict:
                o = 1
            if pointer-number-2 < 0:
                break
            cand = chars[pointer-number-2] + cand_l
            cand_l = cand
        return f, s, o

    def train_text(self, text):  # textを投げ込むと素性を学習データを作る
        text = text.strip()
        if '|' in list(text) or '-' in list(text):  # ここ部分的annotationを自動で判定するけどやめた方がいいかもね
            teacher = self.get_teacher_part(text)
            position_list = self.get_position_part(text)  # 学習速度の問題で、ここで学習するpointerを把握
            chars = list(text.replace("-", "").replace("|", ""))
            features = []  
            for pointer in position_list:  # 学習
                feature = self.get_feature(pointer, chars)
                features.append(feature)
        else:  # フルアノテーション時
            teacher = self.get_teacher(text)
            chars = list(text.replace(" ", ""))
            features = []
            for pointer in range(1, len(chars)):  # その文字の右側に空白があるかどうかを判定
                feature = self.get_feature(pointer, chars)
                features.append(feature)
        return features, teacher

    def get_position_part(self, text):  # 部分的アノテーションの学習するpointerの場所をもらう.
        chars = list(text)
        position_list = []
        count = 0  # 学習pointの個数
        for number in range(1, len(chars)):
            if chars[number] == '|' or chars[number] == '-':
                position_list.append(number - count)  # number-countで、原文の学習pointを示せる
                count+= 1
        return position_list

    def get_teacher(self, text):  # 空白付きの文字列を送ると、boundary_list(teacher)を返す
        chars = list(text)
        teacher = []
        flag = 0
        for char in chars:
            if char == ' ':
                teacher.append(1)
                flag = 1
            elif flag == 0:  # 直前がboundaryでないときのみ
                teacher.append(0)
                flag = 0
            else:
                flag = 0  # 直前処理
        teacher = np.array(teacher[1:])
        return teacher

    def get_teacher_part(self, text):  # 部分的アノテーションの文字列を送ると、boudary_list(teacher)を返す
        chars = list(text)
        teacher = []
        flag = 0
        for char in chars:
            if char == '|':
                teacher.append(1)
                flag = 1
            elif char == '-':
                teacher.append(0)
                flag = 0
            elif flag == 0:  # 直前が文字である時のみ  # not_segmented
                #teacher.append(2)
                pass
            else:
                flag = 0  # 直前処理
        return teacher

#Analyser = Morp(estimator = SGDClassifier(loss='hinge'))
Analyser = Morp()
Analyser.train(['../experiment/corpus/OY-test.word'])
print('正解:決算 発表 まで 、 じっと 我慢 の 子 で い られ る か な 。')
print(Analyser.word_segment('決算発表まで、じっと我慢の子でいられるかな。'))
print(Analyser.word_segment('インフレは欧州市民にとって最大の懸念事項'))
print(Analyser.word_segment('人の命の大事さを実感できる施設で働いていたにも関わらず、'))

#f = open('model', 'wb')
#pickle.dump(Analyser, f)
#f.close()
#print(Analyser.word_segment('市民'))
#print(Analyser2.word_segment('人の命の大事さを実感できる施設で働いていたにも関わらず、'))
#print(Analyser2.word_segment('インフレは欧州市民にとって最大の懸念事項'))
#print(Analyser2.word_segment('市民'))
