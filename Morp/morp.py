import numpy as np
from sklearn.svm import LinearSVC

class Morp:
    '''
    点推定を利用して、日本語の単語分割を行う。　
    学習, 推定をこれで行う.
    '''
    def __init__(self, word_dict=None):                  # コンストラクタ
        self.name = ""
        self.word_dict = word_dict
        self.estimator = LinearSVC(C=1.0)
        self.char_dict = {}

    def word_segment(self, text):
        '''
        入力されたtextを単語分割して返す.
        '''
        text = text.strip()
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

    def get_feature(self, pointer, chars):
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

        # 文字種
        tl1 = self.get_types(l1)
        tl2 = self.get_types(l2)
        tl3 = self.get_types(l3)
        tr1 = self.get_types(r1)
        tr2 = self.get_types(r2)
        tr3 = self.get_types(r3)

        # 辞書  # dictを読んで、現在のpoint直前で終わる(f) / point直後から単語が始める(s) / pointの上を辞書がまたぐ(o).
        f, s, o = 0, 0, 0
        if not self.word_dict == None:  # 辞書を持っているpattern
            f, s, o = self.get_dict_feature(pointer, chars)
        feature = [l1, l2, l3, r1, r2, r3, tl1, tl2, tl3, tr1, tr2, tr3, f, s, o]
        return feature

    def decision_boundary(self, feature):  # 
        feature_array = np.zeros([1, 6*(len(self.char_dict) + 1) + 3])
        for char, number in zip(feature[:5], range(1,7)):  # r1~l3まで
            if not char in self.char_dict:
                feature_array[0][number*self.char_dict['UNK']] = 1 # 存在しない時 
            else:
                feature_array[0][number*self.char_dict[char]] = 1
        for number in range(6, 15):  # tr1~o
            feature_array[0][6*(len(self.char_dict)) + number-6] = feature[number]
        prediction = self.estimator.predict(feature_array)
        return prediction[0]

    def get_types(self, char):
        '''
        文字種判別. ひらがな=0, カタカナ=1, 漢字=2, Alphabet=3, 数字=4, その他=5
        '''
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

    def train_model(self, text_path): # ファイル一つから学習を行う。
        text = ""
        for line in open(text_path, 'r'):  # 全ての文字を繋げたtextを生成
            line = line.strip().replace(" ", "")
            text = text + line
        chars = list(text)  # 1文字づつのリスト
        self.char_dict = {'UNK':0, '^':1, '$':2} # 0はUNK
        char_number = 3
        for char in chars: # 辞書を作る
            if char in self.char_dict:
                continue
            else:
                self.char_dict[char] = char_number
                char_number += 1
        line_number = 0
        for line in open(text_path, 'r'):  # 学習データ生成
            line = line.strip()
            features, teacher = self.train_text(line)
            feature_array = self.make_feature_array(features, len(teacher))  # data_sizeは教師データのsizeから求めてる
#            print(feature_array, teacher)
            self.estimator.fit(feature_array, teacher)  # 学習部分 # まとめて学習するようにしよう
        return 0

    def make_feature_array(self, features, data_size):  # featuresを投げるとarrayを返す
        feature_array = np.zeros([data_size, 6*(len(self.char_dict) + 1) + 3])  # 行はデータサイズ、列は素性の次元(1文字につき、文字次元+文字種)
        line_number = 0
        for feature in features:
            for char, number in zip(feature[:5], range(1,7)):  # r1~l3まで
                feature_array[line_number][number*self.char_dict[char]] = 1
            for number in range(6, 15):  # tr1~o
                feature_array[line_number][6*(len(self.char_dict)) + number-6] = feature[number]
            line_number += 1
        return feature_array

    def train_text(self, text):  # textを投げ込むと素性を学習データを作る
        text = text.strip()
        teacher = self.get_teacher(text)
        chars = list(text.replace(" ", ""))
        features = []
        for pointer in range(1, len(chars)):  # その文字の右側に空白があるかどうかを判定
            feature = self.get_feature(pointer, chars)
            features.append(feature)  
        return features, teacher

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
        teacher = teacher[1:]
        return teacher
# sample_code
word_dict = {"ドイツ人":1, "ドイツ":1}  # word_dict
Analyser = Morp(word_dict)  # word_dictを指定(省略可能)
Analyser.train_model('sample.txt')  # テキストファイルから学習
print(Analyser.word_segment('私は元気です'))  # 単語分割結果を返す
