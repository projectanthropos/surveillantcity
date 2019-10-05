# This script is able to extract the content information from PDF
# The tool used is pdfminer which can be found on https://github.com/euske/pdfminer


# Import packages
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

#==========================================
# Name: extract_pdf_content
# Purpose: Extract the pdf content from a pdf document
# Input Parameter(s): pdf indicates the pdf document name
# Return Value(s): The extracted content information
#============================================
def extract_pdf_content(pdf):
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    outfp = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr=rsrcmgr, outfp=outfp, codec=codec, laparams=laparams)

    # open the pdf document
    with open(pdf, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()
        # parse the pdf page-by-page
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)
    resultstr = outfp.getvalue()
    device.close()
    outfp.close()
    return resultstr