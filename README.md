# Sub-Character Embedding for Chinese Language Models and Transfer Learning for Gender By Chinese Name

[Ted Yuan](https://www.linkedin.com/in/tedyuan/)
August, 2026

## 1. Introduction to Chinese Characters 

### 1.1 汉字的组成部分, 偏旁和部首 (Chinese Character Components and Radicals)

汉字由笔画、部件（偏旁）和整字三个层次组成。偏旁是合体字的构字部件，部首是字典中为了分类和检索而设立的特定表义偏旁，而更小的层级还包括笔画和声旁/形旁

### 1.2 汉字的主要组成层级

* 笔画 (Strokes)
   * 汉字最小的构成单位。
   * 例子：一（横）、丨（竖）、丿（撇）、㇏（捺）。 [2]
* 部件 / 偏旁 (Components / Pianpang)
   * 由笔画组成的、能组配汉字的构字单位。古代指左右结构的左偏右旁，现代泛指上下、左右、内外等所有构字部分。
   * 例子：“明”字可拆分为“日”和“月”两个部件。 [2, 4, 5, 6]
* 部首 (Radicals)
   * 具有字形归类和查检字典功能的特殊偏旁，通常处于字的首位或表义核心。
   * 例子：“妈”字的部首是“女”，“你”字的部首是“亻”。 [3, 4, 5, 7]
* 形旁与声旁 (Meaning and Sound components)
   * 形旁：表示汉字的意义范畴（如“氵”表示与水相关）。
   * 声旁：表示汉字的读音线索（如“清”中的“青”提示读音）。 [5]

### 1.3 偏旁、部首与部件的联系和区别

* 范围不同：部首一定是偏旁，但偏旁不一定能成为部首。所有的部首都是部件，但部件比偏旁和部首的范围更大。
* 用途不同：偏旁侧重于分析字形结构与组字规律；部首侧重于辞书编纂与汉字检索。

### 1.4 演示一个具体汉字的完整拆解过程

这里以汉字“湖” (hú) 为例，展示它从整字到笔画的完整拆解过程。
#### 汉字“湖”的结构拆解
```
       【 湖 】 (整字)
        /    \
     【氵】  【胡】 (第一层：偏旁/部件)
              /   \
           【古】 【月】 (第二层：切分部件)
            /   \
         【十】【口】 (第三层：独体字)
```
#### 1. 整字层 (Whole Character)

* 字：湖
* 结构：左中右结构（也可以看作左边的“氵”和右边的“胡”组成的左右结构）。
* 类型：形声字（形旁表意，声旁表音）。

#### 2. 偏旁与部件层 (Components / Pianpang)

* 左侧偏旁（形旁 / 部首）：氵 (三点水)
* 作用：表示意义。提示这个字的含义与水有关（江河湖海）。
   * 身份：它既是偏旁，也是检索这个字时的部首。
* 右侧部件（声旁）：胡
* 作用：提示读音。“胡” (hú) 与“湖” (hú) 读音相同。

#### 3. 更深层的部件拆解 (Sub-components)
右边的“胡”字是一个合体字，可以进一步拆解：

* 古：左上方的部件。
* 可再细分为独体字：十（shí）和 口（kǒu）。
* 月：右下方的部件（肉字旁）。

#### 4. 最小单位：笔画层 (Strokes)
“湖”字共有 12 画。按正确的笔顺拆解为：

* 氵（3画）：点、点、提
* 古（5画）：横、竖、竖、横折、横
* 月（4画）：撇、横折钩、横、横


## 2. Goal 

We propose a compact sub-character structural encoder that jointly represents radical identity and the ordered structural decomposition of Chinese characters, compressing them into a 64-dimensional latent representation through multi-target reconstruction. The resulting encoder can be frozen and reused as a lightweight character representation for downstream tasks.

To prove the following hypothesis.

### Hypothesis
A Chinese character representation learned solely from Chinese sub-character structure can transfer to a downstream semantic/social prediction task without task-specific modification of the representation.

In other words,
How much useful Chinese character knowledge can be compressed into a small, reusable representation when the model is given both radical identity and ordered structural decomposition?

### Architecture

Sub-Character Transfer Learning for Chinese Name Gender Prediction

#### Design
```
        AUTOENCODER

learn Chinese character structure
             ↓
          freeze
             ↓
       GENDER MODEL

learn how structural representations
   relate to name-level gender
```

The architecture effectively implements:

$$z(c)=E(\text{radical}(c),\text{components}(c))$$

Then for a two-character name:

$$
z(name)=[z(c_1),z(c_2)]
$$

and:

$$
P(\text{male}\mid name)=G([z(c_1),z(c_2)])
$$


This means the gender classifier doesn't directly learn from the glyph identity.

It learns from a representation of the character's sub-character structure.

That's exactly where the phrase Sub-Character Transfer Learning becomes meaningful.

The latent representation captures reusable structural information about Chinese characters, and this structural information can be transferred to another task.

## 3. Dataset and  Code Availability
The source code and trained model weights for this project are available here.

### Programming Environment
```bash
$python --version
Python 3.12.13
$ sqlite3 --version
3.51.0 2025-06-12 13:14:41 f0ca7bba1c5e232e5d279fad6338121ab55af0c8c68c84cdfb18ba5114dcaapl (64-bit)
```

### Datasets
  - Vocabularies
    - **data/radicals.txt**: contains 294 unique radicals 部首
    - **data/components.txt**: contains 1823 unique components 偏旁部件

  - Chinese Character (汉字) SQLite3 Database
    - **data/hanzi.db**: contains detail of 9574 characters. Example usage:
```bash
# show tables
sqlite3 hanzi.db ".tables"
# display schema
sqlite3 hanzi.db ".schema characters"

sqlite3 hanzi.db "select * from characters limit 10"
```
output:
```
⺀|？|ice|||||None,None||⺀
⺈|？||||||None,None||⺈
⺊|⿰丨？||A crack on an oracle bone; compare 卜|||ideographic|[0],None||⺊
⺌|？||||||None,None,None||⺌
⺍|？||||||None,None,None||⺍
⺗|？|heart; mind; soul|||||None,None,None,None||⺗
⺮|？|bamboo; flute|Two stalks of bamboo; see 竹|||pictographic|None,None,None,None,None,None|zhú|⺮
⺳|⿱冖八|net, network|||||[0],[0],[1],[1]||⺳
⺼|？|meat, flesh; organic compound|Meat on the ribs of an animal; compare 肉|||pictographic|None,None,None,None||⺼
㐆|⿻尸？|old form of 隱|||||None,[0],[0],None,[0],None|yǐn|尸
```

```bash
sqlite3 hanzi.db "select * from ideographic_description_characters limit 10"
```
output:
```
1|⿰|U+2FF0|Ideographic description character left to right
2|⿱|U+2FF1|Ideographic description character above to below
3|⿲|U+2FF2|Ideographic description character left to middle and right
4|⿳|U+2FF3|Ideographic description character above to middle and below
5|⿴|U+2FF4|Ideographic description character full surround
6|⿵|U+2FF5|Ideographic description character surround from above
7|⿶|U+2FF6|Ideographic description character surround from below
8|⿷|U+2FF7|Ideographic description character surround from left
9|⿼|U+2FFC|Ideographic description character surround from right
10|⿸|U+2FF8|Ideographic description character surround from upper left

```

  - **data/training_dataset.txt**: contains 9754 entries of character, radical, decomposition fields separated by '|'.
    - The decomposition field contains component sequence, phonetic and semantic components
    - These serve as features in the sub-character embedding and unsupervised learning, e.g., an autoencoder.

  - Two external datasets and dependency
    - **[dictionary.txt](https://github.com/skishore/makemeahanzi/blob/master/dictionary.txt)** contains 9574 characters and the sub-character information.
    - **[gender/dataverse_files/CnGender.txt](https://www.nature.com/articles/s41597-025-06276-y)** contains ~1 million Chinese names with Male probability.


### Data Processing
  - **python/scan_schema.py**
    - Input: **[dictionary.txt](https://github.com/skishore/makemeahanzi/blob/master/dictionary.txt)** contains 9574 characters and the sub-character information.
    - Output: hanzi_dictionary.csv. It is used to create hanzi.db. The CSV file has the schema:
```
{
    "character": ["str"],
    "definition": ["str"],
    "pinyin": ["list"],
    "decomposition": ["str"],
    "radical": ["str"],
    "matches": ["list"],
    "etymology_hint": ["null", "str"],
    "etymology_phonetic": ["null", "str"],
    "etymology_semantic": ["null", "str"],
    "etymology_type": ["null", "str"]
}
```

## 4. Sub-Character Embedding for Chinese Language Models

We will need to install necessary python packages to develop the embedding and autoencoder.

### Tensorflow/Keras Installation
```bash
# 1. Create a new environment named 'keras3_env' with Python 3.12
conda create -n keras3_env python=3.12 -y

# 2. Activate the new environment
conda activate keras3_env

# 3. Upgrade pip to ensure clean build wheels
pip install --upgrade pip

# 4. Install the required deep learning framework versions
pip install tensorflow==2.21.0 keras==3.15.1 numpy

# verify
python -c "import os; os.environ['KERAS_BACKEND']='tensorflow'; import keras; print(keras.__version__)"

```


### Embedding and Un-supervised Learning


  - python/chinese_char_autoencoder.py
    - input: radicals.txt, components.txt, training_dataset.txt
    - output: 
      - Encoder latent embedding, model and vocabs: hanzi_encoder_weights.weights.h5, saved_radical_vocab.npy, saved_component_vocab.npy
      - Latent embedding cache of the characters in training_dataset: hanzi_embeddings_64d.npy, hanzi_index_lookup.txt

## 5. Transfer Learning for Gender By Chinese Name

  - python/chinese_name_gender_predictor.py
    - input: 
      - Encoder embedding model: saved_radical_vocab.npy, saved_component_vocab.npy, hanzi_encoder_weights.weights.h5, 
      - training_dataset.txt, used as the character → sub-character decomposition dictionary
      - [gender/dataverse_files/CnGender.txt](https://www.nature.com/articles/s41597-025-06276-y), 1 million names and male probability.
    - output: hanzi_name_gender_predictor.keras



## 6. References

* [Chinese character components - wikipedia.org](https://en.wikipedia.org/wiki/Chinese_character_components)
* [Make Me a Hanzi - Free, open-source Chinese character data](https://github.com/skishore/makemeahanzi)
* The Chinese Name-to-Gender Dataset ([Nature](https://www.nature.com/articles/s41597-025-06276-y)/[PubMed](https://pubmed.ncbi.nlm.nih.gov/41413051/))
* Inspired by [Chinese Character Decomposition Data](https://github.com/JustinMi/Chinese-Character-Decomposition-Data) for data processing in Sqlite, but this repository does not use nor depend on it.
* Paper is published at [https://doi.org/10.17605/OSF.IO/H2UJA](https://doi.org/10.17605/OSF.IO/H2UJA)
