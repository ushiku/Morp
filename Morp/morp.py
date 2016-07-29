import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from scipy import sparse


class Morp:
    '''
    点予測を利用して、日本語の単語分割を行う。　
    学習, 推定をこれで行う.
    '''
    def __init__(self, word_dict=None, estimator = LinearSVC(C=1.0)):  # 多分データ量が多くなってきたらSVGとかにする?
        self.name = ""
        self.word_dict = word_dict
        self.estimator = estimator
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

    def train(self, text_path_list):  # 引数はpath, あるいはpath_list
        if type(text_path_list) == str:  # pathが一つの時
            text = ""
            for line in open(text_path_list, 'r'):  # 全ての文字を繋げたtextを生成
                line = line.strip().replace(" ", "")
                text = text + line
            chars = list(text)  # 1文字づつのリスト
            self.char_dict = {'UNK': 0, '^': 1, '$': 2}  # 0はUNK
            char_number = 3
            for char in chars:  # 辞書を作る
                if char in self.char_dict:
                    continue
                else:
                    self.char_dict[char] = char_number
            total_feature, total_teacher = self.train_file(text_path_list)

        if type(text_path_list) == list:  # pathが複数
            text = ""
            for text_path in text_path_list:
                for line in open(text_path, 'r'):  # 全ての文字を繋げたtextを生成
                    line = line.strip().replace(" ", "")
                    text = text + line
            chars = list(text)  # 1文字づつのリスト
            self.char_dict = {'UNK': 0, '^': 1, '$': 2}  # 0はUNK
            char_number = 3
            for char in chars:  # 辞書を作る
                if char in self.char_dict:
                    continue
                else:
                    self.char_dict[char] = char_number
            first_flag = 1
            for text_path in text_path_list:
                if first_flag == 1: # 初回用
                    total_feature, total_teacher = self.train_file(text_path)
                    first_flag = 0
                else:
                    feature, teacher = self.train_file(text_path)
                    total_feature = sparse.vstack((total_feature, feature))
                    total_teacher = np.hstack((total_teacher, teacher))
        self.estimator.fit(total_feature.todense(), total_teacher)
        return 0

    def train_file(self, text_path):
        '''
        ファイル一つから学習を行う。学習部分について要確認. text_pathを複数いれた方がいいのかね? まあcatくらいは自前でやってほしいかも
        '''
        first_flag = 1  # 学習用のarrayの初期用
        line_number = 0
        for line in open(text_path, 'r'):  # 学習データ生成
            line = line.strip()
            features, teacher = self.train_text(line)
            feature_array = self.make_feature_array(features, len(teacher))  # data_sizeは教師データのsizeから求めてる
            if first_flag == 1:  # 初回用
                total_feature = feature_array
                total_teacher = teacher
                first_flag = 0
            else:  # ここが計算量のneckになっている...??  # つーかメモリの量の問題な気もする....(500文で,10000次元ぐらいのベクトル)  # scipyを入れて多少はマシに?
                total_feature = sparse.vstack((total_feature, feature_array))
                total_teacher = np.hstack((total_teacher, teacher))
#            print(line_number, len(total_teacher))
            line_number += 1
        return total_feature, total_teacher

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
        if not self.word_dict is None:  # 辞書を持っているpattern
            f, s, o = self.get_dict_feature(pointer, chars)
        feature = [l1, l2, l3, r1, r2, r3, tl1, tl2, tl3, tr1, tr2, tr3, f, s, o]
        return feature

    def decision_boundary(self, feature):  # 
        feature_array = np.zeros([1, 6*(len(self.char_dict) + 1) + 3])
        for char, number in zip(feature[:5], range(1, 7)):  # 素性r1~l3まで
            if char not in self.char_dict:
                feature_array[0][number*self.char_dict['UNK']] = 1  # 辞書に存在しない時 
            else:
                feature_array[0][number*self.char_dict[char]] = 1
        for number in range(6, 15):  # 素性tr1~o
            feature_array[0][6*(len(self.char_dict)) + number-6] = feature[number]
        prediction = self.estimator.predict(feature_array)
        return prediction[0]

    def get_types(self, char):  # 文字種判別. ひらがな=0, カタカナ=1, 漢字=2, Alphabet=3, 数字=4, その他=5
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

    def make_feature_array(self, features, data_size):  # featuresを投げるとarrayを返す
        feature_array = np.zeros([data_size, 6*(len(self.char_dict) + 1) + 3])  # 行はデータサイズ、列は素性の次元(1文字につき、文字次元+文字種)
        line_number = 0
        for feature in features:
            for char, number in zip(feature[:5], range(1, 7)):  # 素性r1~l3まで
                feature_array[line_number][number*self.char_dict[char]] = 1
            for number in range(6, 15):  # 素性tr1~o
                feature_array[line_number][6*(len(self.char_dict)) + number-6] = feature[number]
            line_number += 1
        return sparse.csr_matrix(feature_array)  # csrは足し算が早い

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

Analyser = Morp()
Analyser.train(['../experiment/corpus/train100.word'])
for number in range(0, 10000):
    Analyser.word_segment('人の命の大事さを実感できる施設で働いていたにも関わらず、')

#print(Analyser.word_segment('市民'))
#print(Analyser2.word_segment('人の命の大事さを実感できる施設で働いていたにも関わらず、'))
#print(Analyser2.word_segment('インフレは欧州市民にとって最大の懸念事項'))
#print(Analyser2.word_segment('市民'))
