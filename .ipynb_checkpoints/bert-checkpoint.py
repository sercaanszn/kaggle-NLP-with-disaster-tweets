{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "acac39ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk.stem.porter import PorterStemmer as ps\n",
    "from nltk.corpus import stopwords\n",
    "import nltk\n",
    "#nltk.download('wordnet')\n",
    "wordLemm = WordNetLemmatizer()\n",
    "ps = ps()\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from tqdm.notebook import tqdm\n",
    "import tensorflow as tf\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2064589",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df = pd.read_csv(r\"C:\\Users\\Sercan\\Desktop\\MLProject\\train.csv\")\n",
    "test_df =  pd.read_csv(r\"C:\\Users\\Sercan\\Desktop\\MLProject\\test.csv\")\n",
    "\n",
    "print(len(train_df))\n",
    "train_df = train_df.drop_duplicates('text', keep='last')\n",
    "print(len(train_df))\n",
    "\n",
    "print(train_df['target'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "1e47b8a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "EMOJIS = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', \n",
    "          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',\n",
    "          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\\\': 'annoyed', \n",
    "          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',\n",
    "          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',\n",
    "          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', \":'-)\": 'sadsmile', ';)': 'wink', \n",
    "          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}\n",
    "URLPATTERN        = r\"((http://)[^ ]*|(https://)[^ ]*|( www\\.)[^ ]*)\"\n",
    "USERPATTERN       = '@[^\\s]+'\n",
    "SEQPATTERN   = r\"(.)\\1\\1+\"\n",
    "SEQREPLACE = r\"\\1\\1\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d2029ff6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "\n",
    "def preprocess_text(text):\n",
    "    text = re.sub(r\"http\\S+\", \"\", text)\n",
    "    ### Replacing URL\n",
    "    text = re.sub(URLPATTERN,' URL',text)\n",
    "    ### Replacing EMOJI\n",
    "    for emoji in EMOJIS.keys():\n",
    "        text = text.replace(emoji, \"EMOJI\" + EMOJIS[emoji])  \n",
    "    ### Replacing USER pattern\n",
    "    text = re.sub(USERPATTERN,' URL',text)\n",
    "    ### Removing non-alphabets\n",
    "    text = re.sub('[^a-zA-z]',\" \",text)\n",
    "    ### Removing consecutive letters\n",
    "    text = re.sub(SEQPATTERN,SEQREPLACE,text)\n",
    "    text = text.split()\n",
    "    text = [wordLemm.lemmatize(word) for word in text if not word in stopwords.words('english') and len(word) > 1]\n",
    "    text = ' '.join(text)\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "b34551f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df['text'] = train_df['text'].apply(preprocess_text) \n",
    "test_df ['text'] = test_df['text'].apply(preprocess_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "86542498",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>keyword</th>\n",
       "      <th>location</th>\n",
       "      <th>text</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Our Deeds Reason earthquake May ALLAH Forgive</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Forest fire near La Ronge Sask Canada</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>All resident asked shelter place notified offi...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>people receive wildfire evacuation order Calif...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Just got sent photo Ruby Alaska smoke wildfire...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>RockyFire Update California Hwy closed directi...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>10</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>flood disaster Heavy rain cause flash flooding...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>13</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>top hill see fire wood</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>14</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>There emergency evacuation happening building ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>15</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>afraid tornado coming area</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>16</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Three people died heat wave far</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>17</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Haha South Tampa getting flooded hah WAIT SECO...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>18</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>raining flooding Florida TampaBay Tampa day lo...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>19</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Flood Bago Myanmar We arrived Bago</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Damage school bus multi car crash BREAKING</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    id keyword location                                               text  \\\n",
       "0    1     NaN      NaN      Our Deeds Reason earthquake May ALLAH Forgive   \n",
       "1    4     NaN      NaN              Forest fire near La Ronge Sask Canada   \n",
       "2    5     NaN      NaN  All resident asked shelter place notified offi...   \n",
       "3    6     NaN      NaN  people receive wildfire evacuation order Calif...   \n",
       "4    7     NaN      NaN  Just got sent photo Ruby Alaska smoke wildfire...   \n",
       "5    8     NaN      NaN  RockyFire Update California Hwy closed directi...   \n",
       "6   10     NaN      NaN  flood disaster Heavy rain cause flash flooding...   \n",
       "7   13     NaN      NaN                             top hill see fire wood   \n",
       "8   14     NaN      NaN  There emergency evacuation happening building ...   \n",
       "9   15     NaN      NaN                         afraid tornado coming area   \n",
       "10  16     NaN      NaN                    Three people died heat wave far   \n",
       "11  17     NaN      NaN  Haha South Tampa getting flooded hah WAIT SECO...   \n",
       "12  18     NaN      NaN  raining flooding Florida TampaBay Tampa day lo...   \n",
       "13  19     NaN      NaN                 Flood Bago Myanmar We arrived Bago   \n",
       "14  20     NaN      NaN         Damage school bus multi car crash BREAKING   \n",
       "\n",
       "    target  \n",
       "0        1  \n",
       "1        1  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  \n",
       "5        1  \n",
       "6        1  \n",
       "7        1  \n",
       "8        1  \n",
       "9        1  \n",
       "10       1  \n",
       "11       1  \n",
       "12       1  \n",
       "13       1  \n",
       "14       1  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head(15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "e242393a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import tensorflow_hub as hub\n",
    "import tensorflow_text as text\n",
    "from official import nlp\n",
    "from official.nlp import bert"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "0dccaf38",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_valid, y_train, y_valid = train_test_split(train_df['text'].tolist(),\\\n",
    "                                                      train_df['target'].tolist(),\\\n",
    "                                                      test_size=0.05,\\\n",
    "                                                      stratify = train_df['target'].tolist(),\\\n",
    "                                                      random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "dc7cc40c",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 32\n",
    "seed = 42\n",
    "train_ds = tf.data.Dataset.from_tensor_slices((train_df['text'].tolist(),train_df['target'].tolist())).batch(batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "47c481bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "BERT model selected           : https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3\n",
      "Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2\n"
     ]
    }
   ],
   "source": [
    "bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'  #@param [\"bert_en_uncased_L-12_H-768_A-12\", \"bert_en_cased_L-12_H-768_A-12\",\n",
    "#\"bert_multi_cased_L-12_H-768_A-12\", \"small_bert/bert_en_uncased_L-2_H-128_A-2\", \"small_bert/bert_en_uncased_L-2_H-256_A-4\", \n",
    "#\"small_bert/bert_en_uncased_L-2_H-512_A-8\", \"small_bert/bert_en_uncased_L-2_H-768_A-12\", \"small_bert/bert_en_uncased_L-4_H-128_A-2\", \"small_bert/bert_en_uncased_L-4_H-256_A-4\", \"small_bert/bert_en_uncased_L-4_H-512_A-8\", \"small_bert/bert_en_uncased_L-4_H-768_A-12\", \"small_bert/bert_en_uncased_L-6_H-128_A-2\", \"small_bert/bert_en_uncased_L-6_H-256_A-4\", \"small_bert/bert_en_uncased_L-6_H-512_A-8\", \"small_bert/bert_en_uncased_L-6_H-768_A-12\", \"small_bert/bert_en_uncased_L-8_H-128_A-2\", \"small_bert/bert_en_uncased_L-8_H-256_A-4\", \"small_bert/bert_en_uncased_L-8_H-512_A-8\", \"small_bert/bert_en_uncased_L-8_H-768_A-12\", \"small_bert/bert_en_uncased_L-10_H-128_A-2\", \"small_bert/bert_en_uncased_L-10_H-256_A-4\", \"small_bert/bert_en_uncased_L-10_H-512_A-8\", \"small_bert/bert_en_uncased_L-10_H-768_A-12\", \"small_bert/bert_en_uncased_L-12_H-128_A-2\", \"small_bert/bert_en_uncased_L-12_H-256_A-4\", \"small_bert/bert_en_uncased_L-12_H-512_A-8\", \"small_bert/bert_en_uncased_L-12_H-768_A-12\", \"albert_en_base\",\n",
    "#\"electra_small\", \"electra_base\", \"experts_pubmed\", \"experts_wiki_books\", \"talking-heads_base\"]\n",
    "\n",
    "map_name_to_handle = {\n",
    "    'bert_en_uncased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',\n",
    "    'bert_en_cased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',\n",
    "    'bert_multi_cased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',\n",
    "    'small_bert/bert_en_uncased_L-2_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',\n",
    "    'small_bert/bert_en_uncased_L-2_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',\n",
    "    'small_bert/bert_en_uncased_L-2_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',\n",
    "    'small_bert/bert_en_uncased_L-2_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',\n",
    "    'small_bert/bert_en_uncased_L-4_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',\n",
    "    'small_bert/bert_en_uncased_L-4_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',\n",
    "    'small_bert/bert_en_uncased_L-4_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',\n",
    "    'small_bert/bert_en_uncased_L-4_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',\n",
    "    'small_bert/bert_en_uncased_L-6_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',\n",
    "    'small_bert/bert_en_uncased_L-6_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',\n",
    "    'small_bert/bert_en_uncased_L-6_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',\n",
    "    'small_bert/bert_en_uncased_L-6_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',\n",
    "    'small_bert/bert_en_uncased_L-8_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',\n",
    "    'small_bert/bert_en_uncased_L-8_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',\n",
    "    'small_bert/bert_en_uncased_L-8_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',\n",
    "    'small_bert/bert_en_uncased_L-8_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',\n",
    "    'small_bert/bert_en_uncased_L-10_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',\n",
    "    'small_bert/bert_en_uncased_L-10_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',\n",
    "    'small_bert/bert_en_uncased_L-10_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',\n",
    "    'small_bert/bert_en_uncased_L-10_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',\n",
    "    'small_bert/bert_en_uncased_L-12_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',\n",
    "    'small_bert/bert_en_uncased_L-12_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',\n",
    "    'small_bert/bert_en_uncased_L-12_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',\n",
    "    'small_bert/bert_en_uncased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',\n",
    "    'albert_en_base':\n",
    "        'https://tfhub.dev/tensorflow/albert_en_base/2',\n",
    "    'electra_small':\n",
    "        'https://tfhub.dev/google/electra_small/2',\n",
    "    'electra_base':\n",
    "        'https://tfhub.dev/google/electra_base/2',\n",
    "    'experts_pubmed':\n",
    "        'https://tfhub.dev/google/experts/bert/pubmed/2',\n",
    "    'experts_wiki_books':\n",
    "        'https://tfhub.dev/google/experts/bert/wiki_books/2',\n",
    "    'talking-heads_base':\n",
    "        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',\n",
    "}\n",
    "\n",
    "map_model_to_preprocess = {\n",
    "    'bert_en_uncased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'bert_en_cased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-2_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-2_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-2_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-2_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-4_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-4_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-4_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-4_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-6_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-6_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-6_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-6_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-8_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-8_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-8_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-8_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-10_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-10_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-10_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-10_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-12_H-128_A-2':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-12_H-256_A-4':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-12_H-512_A-8':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'small_bert/bert_en_uncased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'bert_multi_cased_L-12_H-768_A-12':\n",
    "        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2',\n",
    "    'albert_en_base':\n",
    "        'https://tfhub.dev/tensorflow/albert_en_preprocess/2',\n",
    "    'electra_small':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'electra_base':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'experts_pubmed':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'experts_wiki_books':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "    'talking-heads_base':\n",
    "        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',\n",
    "}\n",
    "\n",
    "tfhub_handle_encoder = map_name_to_handle[bert_model_name]\n",
    "tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]\n",
    "\n",
    "print(f'BERT model selected           : {tfhub_handle_encoder}')\n",
    "print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "e7b977d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_classifier_model():\n",
    "  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')\n",
    "  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')\n",
    "  encoder_inputs = preprocessing_layer(text_input)\n",
    "  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')\n",
    "  outputs = encoder(encoder_inputs)\n",
    "  net = outputs['pooled_output']\n",
    "  net = tf.keras.layers.Dropout(0.1)(net)\n",
    "  net = tf.keras.layers.Dense(1, activation= \"sigmoid\" , name='classifier')(net)\n",
    "  return tf.keras.Model(text_input, net)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "edca2179",
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier_model = build_classifier_model()\n",
    "tf.keras.utils.plot_model(classifier_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd135133",
   "metadata": {},
   "outputs": [],
   "source": [
    "loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)\n",
    "metrics = tf.metrics.BinaryAccuracy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e8c4bd9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs = 20\n",
    "steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()\n",
    "num_train_steps = steps_per_epoch * epochs\n",
    "num_warmup_steps = int(0.1*num_train_steps)\n",
    "\n",
    "init_lr = 3e-5\n",
    "optimizer = optimization.create_optimizer(init_lr=init_lr,\n",
    "                                          num_train_steps=num_train_steps,\n",
    "                                          num_warmup_steps=num_warmup_steps,\n",
    "                                          optimizer_type='adamw')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4eb8c597",
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier_model.compile(optimizer=optimizer,\n",
    "                         loss=loss,\n",
    "                         metrics=metrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bcf00d52",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Training model with {tfhub_handle_encoder}')\n",
    "history = classifier_model.fit(x=train_ds, epochs=epochs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dab890c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "probs = classifier_model.predict(test_df[\"text\"]) \n",
    "threshold = 0.4\n",
    "preds = np.where(probs[:,] > threshold, 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca48fd1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(preds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3486a0f7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
