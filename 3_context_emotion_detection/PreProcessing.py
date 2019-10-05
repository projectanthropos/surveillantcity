# THIS SCRIPT FOCUSES ON RAW DATA PRE-PROCESSING

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import re
import config
import numpy as np


# GLOABL VARIABLES
NUM_CLASSES = 4                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = 15000            # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = 35        # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = 300             # The dimension of the word embeddings

trnfile = 'data/train.txt'
devfile = 'data/dev.txt'
tstfile = 'data/test.txt'

trnfile_save = 'data/preprocessing_result/train_clean.csv'
devfile_save = 'data/preprocessing_result/dev_clean.csv'
tstfile_save = 'data/preprocessing_result/tst_clean.csv'

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


#==========================================
# Name: file_stat
# Purpose: CALCULATE STATISTICS INFORMATION ABOUT RAW DATA,
#          AND SAVE THOSE INFORMATION INTO TXT FILE
# Input Parameter(s): NONE
# Return Value(s): NONE
#============================================
def file_stat():
    # READING TRAINING FILE AS DATAFRAME
    trnDF = pd.read_table(trnfile)
    trn_num = len(trnDF)
    trn_angry_num = len(trnDF[trnDF['label']== 'angry'])
    trn_happy_num = len(trnDF[trnDF['label']== 'happy'])
    trn_sad_num = len(trnDF[trnDF['label'] == 'sad'])
    trn_other_num = len(trnDF[trnDF['label'] == 'others'])

    # READING DEV FILE AS DATAFRAME
    devDF = pd.read_table(devfile)
    dev_num = len(devDF)
    dev_angry_num = len(devDF[devDF['label'] == 'angry'])
    dev_happy_num = len(devDF[devDF['label'] == 'happy'])
    dev_sad_num = len(devDF[devDF['label'] == 'sad'])
    dev_other_num = len(devDF[devDF['label'] == 'others'])

    # READING TEST FILE AS DATAFRAME
    tstDF = pd.read_table(tstfile)
    tst_num = len(tstDF)
    tst_angry_num = len(tstDF[tstDF['label'] == 'angry'])
    tst_happy_num = len(tstDF[tstDF['label'] == 'happy'])
    tst_sad_num = len(tstDF[tstDF['label'] == 'sad'])
    tst_other_num = len(tstDF[tstDF['label'] == 'others'])

    # WRITING INFORMATION INTO TXT FILE
    statf = open('data/preprocessing_result/stat.txt','a')

    statf.write('Total number records in train set: ' + str(trn_num) + '\n')
    statf.write('Total number angry records in train set: ' + str(trn_angry_num) + '\n')
    statf.write('Total number happy records in train set: ' + str(trn_happy_num) + '\n')
    statf.write('Total number sad records in train set: ' + str(trn_sad_num) + '\n')
    statf.write('Total number other records in train set: ' + str(trn_other_num) + '\n')
    statf.write('\n')

    statf.write('Total number records in dev set: ' + str(dev_num) + '\n')
    statf.write('Total number angry records in dev set: ' + str(dev_angry_num) + '\n')
    statf.write('Total number happy records in dev set: ' + str(dev_happy_num) + '\n')
    statf.write('Total number sad records in dev set: ' + str(dev_sad_num) + '\n')
    statf.write('Total number other records in dev set: ' + str(dev_other_num) + '\n')
    statf.write('\n')

    statf.write('Total number records in test set: ' + str(tst_num) + '\n')
    statf.write('Total number angry records in test set: ' + str(tst_angry_num) + '\n')
    statf.write('Total number happy records in test set: ' + str(tst_happy_num) + '\n')
    statf.write('Total number sad records in test set: ' + str(tst_sad_num) + '\n')
    statf.write('Total number other records in test set: ' + str(tst_other_num) + '\n')
    statf.write('\n')


#==========================================
# Name: text_processor
# Purpose: DEFINE THE TextPreProcessor IN ekphrasis
#          THE DETAILED INFORMATION CAN BE FOUND VIA LINK:
#          https://github.com/cbaziotis/ekphrasis
# Input Parameter(s): NONE
# Return Value(s): NONE
#============================================
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons, config.EMOTICONS_TOKEN]
)


#==========================================
# Name: remove_info
# Purpose: REMOVE REDUNDANT INFORMATION FROM TEXT
# Input Parameter(s): text --- THE RAW TEXT
# Return Value(s): text --- THE TEXT AFTER REMOVING REDUNDANT INFORMATION
#============================================
def remove_info(text):
    # remove repeated info
    repeatedChars = ['.', '?', '!', ',', ')']
    for c in repeatedChars:
        lineSplit = text.split(c)
        while True:
            try:
                lineSplit.remove('')
            except:
                break
        cSpace = ' ' + c + ' '
        text = cSpace.join(lineSplit)

    # Remove any duplicate spaces
    duplicateSpacePattern = re.compile(r'\ +')
    text = re.sub(duplicateSpacePattern, ' ', text)

    # Remove stray punctuation
    stray_punct = ['‑', '-', "^", ":",
                   ";", "#", ")", "(", "*", "=", "\\", "/"]
    for punct in stray_punct:
        text = text.replace(punct, "")

    # Remove/Replace Emoji
    emoji_repeatedChars = config.TWEMOJI_LIST
    for emoji_meta in emoji_repeatedChars:
        if config.TWEMOJI[emoji_meta][0] == 'Crying Cat Face':
            a = 1
        text = text.replace(emoji_meta, config.TWEMOJI[emoji_meta][0].lower())

    # Check logograms
    for item in config.LOGOGRAM.keys():
        text = text.replace(' ' + item + ' ', ' ' + config.LOGOGRAM[item].lower() + ' ')
    tags = ['<url>', '<email>', '<user>', '<hashtag>', '</hashtag>',
            '<elongated>', '</elongated>', '<repeated>', '</repeated>', '<allcaps>']

    # Removing of tags added by ekphrasis
    for tag in tags:
        text = text.replace(tag,'')
    return text


#==========================================
# Name: tokenize
# Purpose: PROCESS TEXT USING THE DEFINED TextPreProcessor IN ekphrasis
# Input Parameter(s): text --- THE RAW TEXT
# Return Value(s): text --- THE TEXT AFTER BEING PROCESSED BY TextPreProcessor
#============================================
def tokenize(text):
    text = " ".join(text_processor.pre_process_doc(text))
    return text


#==========================================
# Name: clean_data
# Purpose: PROCESS RAW DATA BY USING TextPreProcessor, AND REMOVE THE REDUNDANT INFORMATION IN IT
#          AND FINALLY GENERATE CLEAN DATA AND SAVE IT AS CSV FILE
# Input Parameter(s): filepath --- THE RAW DATASET FILEPATH
#                     savepath --- THE FILEPATH FOR CLEAN DATA CSV
#                     mode --- THE CALLING MODE: 'train' or 'test'
# Return Value(s): NONE
#============================================
def clean_data(filepath, savepath, mode = 'train'):
    # READING RAW DATASET AS DATAFRAME
    if mode == 'train':
        curDF = pd.read_table(filepath)
    else:
        curDF = pd.read_csv(filepath)
    cur_num = len(curDF)
    clean_dict = {'turn1':[], 'turn2':[], 'turn3':[], 'label':[]}

    # USE FOR LOOP TO PROCESS DATA RECORD ONE-BY-ONE
    for i in range(0, cur_num):
        cur_record = curDF.iloc[i,:]
        turn1 = cur_record['turn1']
        turn2 = cur_record['turn2']
        turn3 = cur_record['turn3']
        if mode == 'train':
            label = cur_record['label']
            label = emotion2label[label]
        # pre-processing data
        # PROCESS TURN1 CONVERSATION TEXT: CALL FUNCTION tokenize AND remove_info
        convers = turn1
        convers = tokenize(convers)
        convers = remove_info(convers)
        clean_dict['turn1'].append(convers)

        # PROCESS TURN2 CONVERSATION TEXT: CALL FUNCTION tokenize AND remove_info
        convers = turn2
        convers = tokenize(convers)
        convers = remove_info(convers)
        clean_dict['turn2'].append(convers)

        # PROCESS TURN3 CONVERSATION TEXT: CALL FUNCTION tokenize AND remove_info
        convers = turn3
        convers = tokenize(convers)
        convers = remove_info(convers)
        clean_dict['turn3'].append(convers)

        # SAVE RECORD LABEL
        if mode == 'train':
            clean_dict['label'].append(label)
        else:
            clean_dict['label'].append(-1)
    # after finish clean, save the clean data into a clean csv
    clean_df = pd.DataFrame.from_dict(clean_dict)
    clean_df.to_csv(savepath, index= False)


# EMBEDDING WORD
#==========================================
# Name: get_clean_data_list
# Purpose: READ THE CLEANED CSV DATA, PUT THEM INTO A LIST AND RETURN IT
# Input Parameter(s): cleanfilepath --- THE CLEANED DATASET FILEPATH
# Return Value(s): conver_turn1 --- THE LIST FOR FIRST CONVERSATION TEXT
#                  conver_turn2 --- THE LIST FOR SECOND CONVERSATION TEXT
#                  conver_turn3 --- THE LIST FOR THIRD CONVERSATION TEXT
#                  conversation --- THE LIST FOR ALL THREE CONVERSATIONS TEXT
#                  labels --- THE LIST FOR CONVERSATIONS RECORD LABLES
#============================================
def get_clean_data_list(cleanfilepath):
    # READ THE CLEANED DATASET AS DATAFRAME
    curDF = pd.read_csv(cleanfilepath)
    # INITIALIZATION
    conver_turn1 = []
    conver_turn2 = []
    conver_turn3 = []
    conversation = []
    labels = []

    for i in range(len(curDF)):
        cur_record = curDF.iloc[i, :]
        # GET THE FIRST CONVERSATION TEXT
        turn1 = cur_record['turn1']
        turn1 = str(turn1)
        # GET THE SECOND CONVERSATION TEXT
        turn2 = cur_record['turn2']
        turn2 = str(turn2)
        # GET THE THIRD CONVERSATION TEXT
        turn3 = cur_record['turn3']
        turn3 = str(turn3)
        # GET THE LABEL OF CONVERSATION TEXT
        curlable = cur_record['label']
        conver_turn1.append(turn1)
        conver_turn2.append(turn2)
        conver_turn3.append(turn3)
        conversation.append(turn1 + '<eos>' + turn2 + '<eos>' + turn3)
        labels.append(curlable)
    return conver_turn1, conver_turn2, conver_turn3, conversation, labels



#==========================================
# Name: get_embedding
# Purpose: USING THE Tokenizer API PROVIDED BY KERAS TO CONDUCT WORD EMBEDDING
# Input Parameter(s): extrafilepath --- WHEN MODE IS 'test', IT REPRESENTS THE TEST FILE PATH
#                     mode --- THE CALLING MODE: 'train' or 'test'
# Return Value(s): t --- THE tokenizer
#                  result_trn --- THE SEQUENCES SET FOR TRAINING DATA
#                  result_val --- THE SEQUENCES SET FOR VALIDATION DATA
#                  result_tst --- THE SEQUENCES SET FOR TEST DATA
#                  result_ex --- THE SEQUENCES SET FOR USER'S NEW TEST DATA
#============================================
def convert_text2sequences(extrafilepath = '', mode = 'train'):
    trn_turn1, trn_turn2, trn_turn3, trn_all, trn_labels = get_clean_data_list(trnfile_save)
    val_turn1, val_turn2, val_turn3, val_all, val_labels = get_clean_data_list(devfile_save)
    tst_turn1, tst_turn2, tst_turn3, tst_all, tst_labels = get_clean_data_list(tstfile_save)

    # prepare tokenizer
    # vectorize the text samples into a 2D integer tensor
    t = Tokenizer(num_words=MAX_NB_WORDS)
    t.fit_on_texts(trn_all + val_all + tst_all)
    word_index = t.word_index
    print('Found %s unique tokens.' % len(word_index))

    trn_encoded = t.texts_to_sequences(trn_all)
    trn_sequences = pad_sequences(trn_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of trn_sequences tensor:', trn_sequences.shape)

    trn_t1_encoded = t.texts_to_sequences(trn_turn1)
    trn_t1_sequences = pad_sequences(trn_t1_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of trn_t1_sequences tensor:', trn_t1_sequences.shape)

    trn_t2_encoded = t.texts_to_sequences(trn_turn2)
    trn_t2_sequences = pad_sequences(trn_t2_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of trn_t2_sequences tensor:', trn_t2_sequences.shape)

    trn_t3_encoded = t.texts_to_sequences(trn_turn3)
    trn_t3_sequences = pad_sequences(trn_t3_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of trn_t3_sequences tensor:', trn_t3_sequences.shape)

    ####### Val
    val_encoded = t.texts_to_sequences(val_all)
    val_sequences= pad_sequences(val_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of val_sequences tensor:', val_sequences.shape)

    val_t1_encoded = t.texts_to_sequences(val_turn1)
    val_t1_sequences = pad_sequences(val_t1_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of val_t1_sequences tensor:', val_t1_sequences.shape)

    val_t2_encoded = t.texts_to_sequences(val_turn2)
    val_t2_sequences = pad_sequences(val_t2_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of val_t2_sequences tensor:', val_t2_sequences.shape)

    val_t3_encoded = t.texts_to_sequences(val_turn3)
    val_t3_sequences = pad_sequences(val_t3_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of val_t3_sequences tensor:', val_t3_sequences.shape)

    ####### Test
    tst_encoded = t.texts_to_sequences(tst_all)
    tst_sequences = pad_sequences(tst_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of tst_sequences tensor:', tst_sequences.shape)

    tst_t1_encoded = t.texts_to_sequences(tst_turn1)
    tst_t1_sequences = pad_sequences(tst_t1_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of tst_t1_sequences tensor:', tst_t1_sequences.shape)

    tst_t2_encoded = t.texts_to_sequences(tst_turn2)
    tst_t2_sequences = pad_sequences(tst_t2_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of tst_t2_sequences tensor:', tst_t2_sequences.shape)

    tst_t3_encoded = t.texts_to_sequences(tst_turn3)
    tst_t3_sequences = pad_sequences(tst_t3_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of tst_t3_sequences tensor:', tst_t3_sequences.shape)

    # labels to one-hot vector
    trn_labels = to_categorical(np.asarray(trn_labels))
    print('Shape of trn_labels tensor:', trn_labels.shape)

    val_labels = to_categorical(np.asarray(val_labels))
    print('Shape of val_labels tensor:', val_labels.shape)

    tst_labels = to_categorical(np.asarray(tst_labels))
    print('Shape of tst_labels tensor:', tst_labels.shape)

    result_trn = []
    result_val = []
    result_tst = []

    # FOR TRAINING
    result_trn.append(trn_sequences)
    result_trn.append(trn_t1_sequences)
    result_trn.append(trn_t2_sequences)
    result_trn.append(trn_t3_sequences)
    result_trn.append(trn_labels)

    # FOR VALIDATION
    result_val.append(val_sequences)
    result_val.append(val_t1_sequences)
    result_val.append(val_t2_sequences)
    result_val.append(val_t3_sequences)
    result_val.append(val_labels)

    # FOR TEST
    result_tst.append(tst_sequences)
    result_tst.append(tst_t1_sequences)
    result_tst.append(tst_t2_sequences)
    result_tst.append(tst_t3_sequences)
    result_tst.append(tst_labels)

    # PRE-PROCESSING USER' NEW TEST FILE WHEN MODE IS 'test'
    if mode =='test':
        if extrafilepath == '':
            print('You must provide file name in [test] mode!')
            return

        ex_turn1, ex_turn2, ex_turn3, ex_all, ex_labels = get_clean_data_list(extrafilepath)
        ####### extra file
        ex_encoded = t.texts_to_sequences(ex_all)
        ex_sequences = pad_sequences(ex_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        print('Shape of ex_sequences tensor:', ex_sequences.shape)

        ex_t1_encoded = t.texts_to_sequences(ex_turn1)
        ex_t1_sequences = pad_sequences(ex_t1_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        print('Shape of ex_t1_sequences tensor:', ex_t1_sequences.shape)

        ex_t2_encoded = t.texts_to_sequences(ex_turn2)
        ex_t2_sequences = pad_sequences(ex_t2_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        print('Shape of ex_t2_sequences tensor:', ex_t2_sequences.shape)

        ex_t3_encoded = t.texts_to_sequences(ex_turn3)
        ex_t3_sequences = pad_sequences(ex_t3_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        print('Shape of ex_t3_sequences tensor:', ex_t3_sequences.shape)

        result_ex = []
        result_ex.append(ex_sequences)
        result_ex.append(ex_t1_sequences)
        result_ex.append(ex_t2_sequences)
        result_ex.append(ex_t3_sequences)
        return result_ex
    else:
        return t, result_trn, result_val, result_tst



#==========================================
# Name: get_embedding_matrix
# Purpose: USING THE PRE-TRAINED GloVe EMBEDDING (https://nlp.stanford.edu/projects/glove/) TO GENERATE EMBEDDING MATRIX
# Input Parameter(s): NONE
# Return Value(s): embedding_matrix --- THE EMBEDDING MATRIX
#                  vocab_size --- THE LENGTH OF UNIQUE TOKEN IN THE DATASET
#                  result_trn --- THE SEQUENCES SET FOR TRAINING DATA
#                  result_val --- THE SEQUENCES SET FOR VALIDATION DATA
#                  result_tst --- THE SEQUENCES SET FOR TEST DATA
#============================================
def get_embedding_matrix():
    t, result_trn, result_val, result_tst = convert_text2sequences()
    vocab_size = len(t.word_index) + 1
    # LOAD THE ENTIRE GLOVE WORD EMBEDDING FILE INTO MEMORY AS DICTIONARY OF WORD TO EMBEDDING ARRAY
    embeddings_index = dict()
    f = open('data/glove.6B/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # CREAT A MATRIX OF ONE EMBEDDING FOR EACH WORD IN THE TRAINING DATASET.
    # WE CAN DO THAT BY ENUMERATING ALL UNIQUE WORDS IN THE TOKENIZER.word_index
    # AND LOCATING THE EMBEDDING WEIGHT VECTOR FROM THE LOADED GLOVE EMBEDDING.
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, vocab_size, result_trn, result_val, result_tst


if __name__ == '__main__':
    #file_stat()
    #clean_data(trnfile, trnfile_save)
    #clean_data(devfile, devfile_save)
    #clean_data(tstfile, tstfile_save)
    #convert_text2sequences()
    pass