# Morp

## 概要
Pythonだけで動く単語分割器です。
点予測とよばれる、文字境界を単語境界として分割できるかを独立に推定しています。
そのため、部分的にアノテーションされたコーパスを学習することができます。
これは、分野適応において有効であると考えられています。
点予測による形態素解析としては、[KyTea](http://www.phontron.com/kytea/index-ja.html)がありますが、Pythonで自然言語処理をやる人向けにもっと手軽に利用できたらということで作りました。KyTeaと異なり、品詞の推定は行いません。

## インストール

Python3.4.3で動作確認をしています。
```
pip install git+https://github.com/ushiku/morp.git
```
ライブラリとして、numpyとscikit-learn, scipyが必要です.


## 使い方
```
word_dict = {'ドイツ人':1}
Analyser = Morp(word_dict)
```
インスタンス生成時に、word_dictを指定できます。 word_dictはなくても動作します。

```
Analyser.train(['sample.txt', 'sample2.txt'])  # テキストファイルから学習
```
trainで、単語境界を学習できます。
テキストファイルの形式は
```
私 は お金　が 好き です
私|は|お-金|が|好-き|で-す　　
```
のいづれかの書き方です。
前者はword_boundaryをスペースで、後者はword_boundaryがあるところを|、ないところを-で表示しています。後者は、後述の部分的アノテーションに対応しています。
行ごとには混在していても大丈夫ですが、同一行に二つの表記が混じるとエラーを吐きます。

```
print(Analyser.word_segment('私は元気です'))
>>> 私 は 元気 です
```
学習後に、word_segmentの引数に文字列を渡すと、単語分割された文字列を出力します。

## 素性について
KyTeaと同じ素性を使っています。
すなわち、とある文字境界について、左側1, 2, 3文字と右側1, 2, 3文字.それぞれの文字種(ひらがな、カタカナ、漢字、数字、その他).単語辞書に乗っている単語が、境界の直前で終わるか(f),境界の直後から始まるか(s),
境界をまたいでいるか(o)のbinary。また、上記の文字、文字種にたいしてbigram, trigramを見ています。

##部分的アノテーションについて
特定の分野の解析制度をあげたい時は、部分的アノテーションをすることで効率的に精度があげられると考えられます。 
```
私は|金-色-夜-叉|が好きです
```
このように部分的に、金色夜叉のみをアノテーションします。
