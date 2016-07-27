
s
class Morp:
    '''
    単語分割のみを点推定でするやつ.
    '''
    def __init__(self):                  # コンストラクタ
        self.name = ""

    def word_segment(self, text):
        '''
        入力されたtextを単語分割して返す
        '''
        text = text.strip()
        chars = list(text)
        output_chars = chars[0]
        for pointer in range(1, len(chars)):  # その文字の右側に空白があるかどうかを判定
            features = self.get_features(pointer, chars)
            binary_boundary = self.decision_boundary(features)
            if binary_boundary == 0:
                output_chars = output_chars + chars[pointer]  
            else:
                output_chars = output_chars + ' ' + chars[pointer] 
        return output_chars

    def get_features(self, pointer, chars):
        return 0
    
    def decision_boundary(self, features):
        return 1



Analyser = Morp()
print(Analyser.word_segment('私は元気です'))
