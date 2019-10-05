# This script will analyze a document and break it into sentences
# Then use the keyword(s) to extract the desired sentences

# import packages
import pandas as pd
import nltk # using this tool to process English Text
from snownlp import SnowNLP # using this tool to process Chinese Text

#==========================================
# Name: parse_sentence
# Purpose: get the sentence who has the keyword from a large document
# Input Parameter(s): contents --- the content from all documents
#                     keyword --- the keyword we set up
#                     result_file_name --- the file we used to save our desired sentences
# Return Value(s): None (generate csv file under "/data")
#============================================

def parse_sentence_from_csv(csv_name, keywords, result_file_name, language = "English"):
    result_dict = {'Target Sentence': []}
    parse_df = pd.read_csv(csv_name)
    contents = parse_df['Content']
    for cur_content in contents:
        # parse content & get separate sentence
        if language == "English":
            cur_content = cur_content.lower()
            sentences = nltk.sent_tokenize(cur_content)
        if language == "Chinese":
            sentences = SnowNLP(cur_content).sentences
        # analyze each sentence to extract the sentence with keywords in
        for sentence in sentences:
            # the keyword in this sentence, keep this sentence
            # since keywords may include many different words, thus we use a bool variable
            target_tag = False
            for kw in keywords:
                if kw in sentence:
                    target_tag = True
                    break
            if target_tag == True:
                result_dict['Target Sentence'].append(sentence)
    # save all the desired sentences into a csv file
    new_df = pd.DataFrame.from_dict(result_dict)
    new_df.to_csv(result_file_name, index = False)
    print("Finish!")

#==========================================
# Name: parse_sentence_from_single_txt
# Purpose: get the sentence who has the keyword from a large document
# Input Parameter(s): txt_name --- the result txt file name
#                     keywords --- the keyword we set up
#                     result_file_name --- the file we used to save our desired sentences
#                     language --- the language used in documents
# Return Value(s): None (generate csv file under "/data")
#============================================
def parse_sentence_from_single_txt(txt_name, keywords, result_file_name, language = "English"):
    result_dict = {'Target Sentence':[]}
    try:
        cur_file = open(txt_name, 'r') # open txt file
        cur_content = cur_file.read()  # read content from txt file
        # parse content & get separate sentence
        if language == "English":
            cur_content = cur_content.lower()
            sentences = nltk.sent_tokenize(cur_content)
        if language == "Chinese":
            sentences = SnowNLP(cur_content).sentences
        # analyze each sentence to extract the sentence with keywords in
        for sentence in sentences:
            # the keyword in this sentence, keep this sentence
            # since keywords may include many different words, thus we use a bool variable
            target_tag = False
            for kw in keywords:
                if kw in sentence:
                    target_tag = True
                    break
            if target_tag == True:
                result_dict['Target Sentence'].append(sentence)
        # save all the desired sentences into a csv file
        new_df = pd.DataFrame.from_dict(result_dict)
        new_df.to_csv(result_file_name, index=False)
        print("Finish {} version!".format(language))
        print(" ")
    finally:
        if cur_file:
            cur_file.close() # close txt file



if __name__ == '__main__':
    # parse documents in English version
    result_file_name_English = 'data/target_sentence_English.csv'
    keywords_English = ['because', 'safe', 'safety', 'violent', 'violence']
    txt_name_English = 'data/parse_result_txt_English.txt'
    parse_sentence_from_single_txt(txt_name_English, keywords_English, result_file_name_English, language="English")
    print(" ")

    # parse documents in Chinese version
    result_file_name_Chinese = 'data/target_sentence_Chinese.csv'
    keywords_Chinese = ['因为', '安全', '暴力']
    txt_name_Chinese = 'data/parse_result_txt_Chinese.txt'
    parse_sentence_from_single_txt(txt_name_Chinese, keywords_Chinese, result_file_name_Chinese, language="Chinese")

    pass














