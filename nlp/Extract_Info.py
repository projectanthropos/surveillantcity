# This script will extract information from documents such as pdf
# Save the contents into a csv file
# It will call the extraction function from pdf_extractor.py

# import packages
import glob
import os
import pandas as pd

import tika
tika.initVM()
from tika import parser



# load pdf_extractor.py module
import importlib.util
pdf_extractor_dir = os.getcwd() + '/pdf_extractor.py'
spec = importlib.util.spec_from_file_location("pdf_extractor", pdf_extractor_dir)
pdf_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_extractor_module)


#==========================================
# Name: get_pdfs
# Purpose: get all the pdf documents under the specific directory
# Input Parameter(s): pdf_path --- the directory in which the pdf documents are located
# Return Value(s): pdf document names
#============================================
def get_pdfs (pdf_path):
    pdfs = glob.glob("{}/*".format(pdf_path))
    return pdfs



#==========================================
# Name: load_parse_result
# Purpose: to check whether the parse csv exists or not
# Input Parameter(s): csv_name --- the parse csv name
# Return Value(s): finished_pdf --- the name of pdf documents who have been processed;
#                  current_df --- the parse csv dataframe
#============================================
def load_parse_result (csv_name, del_tag = "1"):
    if os.path.exists(csv_name):
        current_df = pd.read_csv(csv_name)
        finished_pdf = current_df['PDF'].tolist()
        if del_tag == "1":
            os.remove(csv_name)
        return finished_pdf, current_df
    else:
        return None, None


#==========================================
# Name: parse_pdf
# Purpose: extract text content from pdf document and save the results as csv file
# Input Parameter(s): pdfs --- the pdf list need to parse;
#                     csv_name --- the parse result csv file name.
# Return Value(s): None (Generate a csv file under "/data")
#============================================
def parse_pdf(pdfs, csv_name):
    # check whether we have a result csv
    finished_pdf, current_df = load_parse_result(csv_name)
    finished_list = []
    exist_tag = "NO"
    if finished_pdf != None:
        exist_tag = "YES"
        finished_list = finished_pdf.copy()
    # skip the finished documents
    # only parse the new documents
    parse_dict = {}
    for pdf in pdfs:
        pdf_name = pdf.split('/')[-1]
        if not pdf_name in finished_list:
            finished_list.append(pdf_name)
            print("Starting to extract content from {}".format(pdf))
            try:
                parsed = parser.from_file(pdf)
                parse_dict[pdf_name] = parsed["content"].strip()
            except:
                try:
                    parse_dict[pdf_name] = pdf_extractor_module.extract_pdf_content(pdf).strip()
                except:
                    print("Cannot parse this document!")
                    continue
            print("Finish the extracting of {}".format(pdf))
    # when finish parsing, change the dictionary into dataframe and save it into csv file
    new_df = pd.DataFrame.from_dict(parse_dict, orient='index').reset_index()
    new_df.columns = ["PDF", "Content"]
    if exist_tag == "YES":
        resultDF = pd.concat([current_df, new_df], ignore_index=True)
        resultDF.to_csv(csv_name, index=False)
    else:
        new_df.to_csv(csv_name, index=False)



#==========================================
# Name: generate_single_txt
# Purpose: by taking advantage of the parsed csv file, generate a single txt file which
#          includes the content texts for all the documents.
# Input Parameter(s): csv_name --- the parse result csv file name;
#                     txt_name --- the single txt file name.
# Return Value(s): None (Generate a txt file under "/data")
#============================================
def generate_single_txt(csv_name, txt_name):
    # check whether we have a result csv
    finished_pdf, current_df = load_parse_result(csv_name, "0")
    # only having the result csv can we generate the single txt file.
    if finished_pdf != None:
        doc_number = len(current_df)
        content = current_df['Content']
        txt_file = open(txt_name, 'a')
        for i in range(doc_number):
            try:
                cur_content = content[i].strip()
                txt_file.writelines(cur_content)
            except:
                print("** Doc " + str(i) + " is empty! **")
                continue
        txt_file.close()
        print('Single txt file has been generated successfully!')
    else:
        print("No parse csv exists!")


if __name__ == '__main__':
    # parse all documents in English
    # initialize variables
    pdf_path_english = "doc_English/"  # the directory where pdfs are located
    csv_name_english = "data/parse_result_csv_English.csv"  # a csv file to keep the final parse result
    txt_name_english = "data/parse_result_txt_English.txt"
    pdfs_english = get_pdfs(pdf_path_english)
    parse_pdf(pdfs_english, csv_name_english)
    generate_single_txt(csv_name_english, txt_name_english)


    # parse all documents in Chinese
    # initialize variables
    pdf_path_chinese = "doc_Chinese/"  # the directory where pdfs are located
    csv_name_chinese = "data/parse_result_csv_Chinese.csv"  # a csv file to keep the final parse result
    txt_name_chinese = "data/parse_result_txt_Chinese.txt"
    pdfs_chinese = get_pdfs(pdf_path_chinese)
    parse_pdf(pdfs_chinese, csv_name_chinese)
    generate_single_txt(csv_name_chinese, txt_name_chinese)

    print("Finish!")
    pass



