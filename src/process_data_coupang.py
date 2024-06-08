# encoding=utf-8
import pickle
import random
from random import *
import numpy as np
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import math
from types import *
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
import torchvision.transforms as transforms
from konlpy.tag import Okt
import gensim
from gensim.models import KeyedVectors
import pickle

# Google Drive에서 작업할 기본 경로를 설정
base_path = '/content/drive/MyDrive/Colab Notebooks/EANN - (src에 코드)'

# 기본 경로를 변경
os.chdir(base_path)

def stopwordslist(filepath='Data/coupang/stop_words.txt'):
    stopwords = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            stopwords[line] = 1
    return stopwords

def clean_str_sst(string):
    def cleansing(text):  
        pattern = '(\[a-zA-Z0-9\_.+-\]+@\[a-zA-Z0-9-\]+.\[a-zA-Z0-9-.\]+)' # e-mail 주소 제거  
        text = re.sub(pattern=pattern,repl=' ',string=text)

        pattern = '(http|ftp|https)://(?:[-\w.]|(?:\da-fA-F]{2}))+'  # url 제거
        text = re.sub(pattern=pattern,repl=' ',string=text)

        pattern = '&lt;[^&gt;]*&gt;'                                 # html tag 제거
        text = re.sub(pattern=pattern,repl=' ',string=text)

        pattern = '[\r|\n|\t]'                                          # \r, \n 제거
        text = re.sub(pattern=pattern,repl=' ',string=text)

        pattern = '[^\w\s]'                                          # 특수기호 제거
        text = re.sub(pattern=pattern,repl=' ',string=text)

        pattern = re.compile(r'\s+')                                 # 이중 space 제거
        text = re.sub(pattern=pattern,repl=' ',string=text)

        return text

    def remove_stopwords(text, stop_words):
            words = text.split()
            words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(words)

    stop_words = set(stopwordslist())
    string = cleansing(string)
    string = remove_stopwords(string, stop_words)
    string = cleansing(string)
    
    return string.strip()

def read_image():
    image_list = {}
    path = '/content/sample_data/image1vs1'
    
    data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    for filename in os.listdir(path):
        try:
            with Image.open(os.path.join(path, filename)).convert('RGB') as im:
                im = data_transforms(im)
                image_list[filename.rsplit('/')[-1].split(".")[0].lower()] = im
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    print(f"Image length {len(image_list)}")

    return image_list
    
    
def write_data(flag, image, text_only):
    def read_post():
        stop_words = stopwordslist()
        key = -1
        
        file_path = "/content/drive/MyDrive/Colab Notebooks/EANN - (src에 코드)/Data/coupang_with_flags_1vs1.csv"
        
        df = pd.read_csv(file_path)
        df = df[df['flag'] == flag]
        
        map_id = df['Category'].tolist()

        data_df = pd.DataFrame(columns=['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label'])
        data_df['post_id'] = df['index']
        data_df['image_id'] = df['index']
        data_df['original_post'] = df['review_clean']
        data_df['post_text'] = df['review_clean'].apply(clean_str_sst)
        data_df['label'] = df['label'] 
        data_df['event_label'] = df['Category']
        data_df['img_num'] = df['img_num']
        data_df['title1'] = df['title1']
        data_df['helpfulness'] = df['helpfulness']
        data_df['rate'] = df['rate']
        data_df['top_reviewer'] = df['top_reviewer']
        data_df['realname_reviewer'] = df['realname_reviewer']
        data_df['review_num'] = df['review_num']
        data_df['line_breaks'] = df['line_breaks']
        
        return data_df['post_text'], data_df

    post_content, post = read_post()
    print("Original post length is " + str(len(post_content)))
    print("Original data frame is " + str(post.shape))

    def select(train, selec_indices):
        temp = []
        for i in range(len(train)):
            ele = list(train[i])
            temp.append([ele[i] for i in selec_indices])
            #   temp.append(np.array(train[i])[selec_indices])
        return temp

    def paired(text_only = False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event= []
        label = []
        post_id = []
        image_id_list = []
        img_num = []
        img_num_list = []
        title1_list = []
        helpfulness_list = []
        rate_list = []
        top_reviewer_list = []
        realname_reviewer_list = []
        review_num_list = []
        line_breaks_list = []

        image_id = ""
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split(':'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post'])
                ordered_post.append(post.iloc[i]['post_text'])
                ordered_event.append(post.iloc[i]['event_label'])
                title1_list.append(post.iloc[i]['title1'])
                helpfulness_list.append(post.iloc[i]['helpfulness'])
                rate_list.append(post.iloc[i]['rate'])
                img_num_list.append(post.iloc[i]['img_num'])
                top_reviewer_list.append(post.iloc[i]['top_reviewer'])
                realname_reviewer_list.append(post.iloc[i]['realname_reviewer'])
                review_num_list.append(post.iloc[i]['review_num'])
                line_breaks_list.append(post.iloc[i]['line_breaks'])
                post_id.append(id)

                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int_)
        ordered_event = np.array(ordered_event, dtype=np.int_)

        print("Sponsored :" + str(sum(label)))
        print("Non Sponsored :" + str(len(label) - sum(label)))
        print("-------------------------------------")

        data = {"post_text": np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "event_label": ordered_event, "post_id":np.array(post_id),
                "image_id":image_id_list, "img_num":img_num_list,
                "title1":title1_list, "helpfulness": helpfulness_list,
                "rate":rate_list, "top_reviewer":top_reviewer_list,
                "realname_reviewer":realname_reviewer_list, 
                "review_num":review_num_list, "line_breaks":line_breaks_list
                }
        #print(data['image'][0])

        print("data size is " + str(len(data["post_text"])))
        
        return data

    paired_data = paired(text_only)
    
    print("paired post length is " + str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data

def load_data(train, validate, test):
    okt = Okt()
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    
    for sentence in all_text:
        # 형태소 분석을 위해 각 문장을 처리
        words = set(okt.morphs(sentence))
        for word in words:
            vocab[word] += 1
    return vocab, all_text

def get_W(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_data(text_only):
    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()

    train_data = write_data("train", image_list, text_only)
    validate_data = write_data("validate", image_list, text_only)
    test_data = write_data("test", image_list, text_only)

    print("loading data...")
    vocab, all_text = load_data(train_data, validate_data, test_data)
    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(vocab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))

    word_embedding_path = "/content/drive/MyDrive/Colab Notebooks/EANN - (src에 코드)/Data/coupang/w2v.pickle"
    with open(word_embedding_path, 'rb') as file:
        w2v = pickle.load(file, encoding='latin1')

    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))

    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    W2 = rand_vecs = {}
    
    with open("/content/drive/MyDrive/Colab Notebooks/EANN - (src에 코드)/Data/coupang/word_embedding.pickle", "wb") as w_file:
        pickle.dump([W, W2, word_idx_map, vocab, max_l], w_file)
    
    return train_data, validate_data, test_data
