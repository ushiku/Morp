from Morp.morp import Morp

Analyser = Morp()
Analyser.train(['sample.txt', 'sample2.txt'])
print(Analyser.word_segment('私は元気です'))
print(Analyser.word_segment('ドイツ人は元気です'))
print(Analyser.word_segment('君の名前は'))
print(Analyser.word_segment('私信を書く'))
