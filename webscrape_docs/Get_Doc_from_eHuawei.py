# This script will download the related document form https://e.huawei.com/

# import packages
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os

#==========================================
# Name: login_huawei
# Purpose: log into the e.huawei.com to download documents
# Input Parameter(s): browser represents the current page
# Return Value(s): None
#============================================
def login_huawei(browser):
    # log in to the website in order to download documents
    login_button = browser.find_element_by_id('reLogin')
    login_button.click()
    time.sleep(1)
    login_name = browser.find_element_by_name('uid')
    login_name.send_keys('lth_test')
    login_password = browser.find_element_by_name('password')
    login_password.send_keys('lth_test')
    login_button = browser.find_element_by_class_name('login_submit_pwd')
    time.sleep(1)
    login_button.click()
    time.sleep(10)


#==========================================
# Name: initialization
# Purpose: configure the browser's settings and get the webpage
# Input Parameter(s): base_url represents the url we want to visit
# Return Value(s): the browser handle
#============================================
def initialization(base_url):
    download_path = os.getcwd() + '/data_ehuawei/'
    options = webdriver.ChromeOptions()
    prefs = {'profile.default_content_settings.popups': 0, 'download.default_directory': download_path}
    options.add_experimental_option('prefs', prefs)
    browser = webdriver.Chrome(chrome_options=options)
    browser.set_page_load_timeout(20)
    browser.get(base_url)
    time.sleep(5)
    return browser


#==========================================
# Name: conduct_download
# Purpose: download the documents from website and ignore the videos
# Input Parameter(s): base_url represents the mainpage url we want to visit
# Return Value(s): None (download documents into the directory we set up)
#============================================
def conduct_download(base_url):
    # get the first page & log in huawei account
    browser = initialization(base_url)
    login_huawei(browser)
    # a flag to determine whether the last page is reached or not
    page_tag = True
    page_number = 1
    # while loop to process every page from google search result
    while page_tag:
        # process current page
        # 1. get the link of each item
        # 2. click the link and visit its page
        # 3. download the document on that page
        # 4. go back to 1. until finish all items on this page
        print("@page:" + str(page_number))
        old_url = browser.current_url
        time.sleep(1)
        soup = BeautifulSoup(browser.page_source, "lxml")
        item_list = soup.find_all('a', class_="blankClass")
        print(str(item_list) + " items @this page!")
        for item in item_list:
            desired_link = item.attrs['href']
            browser.get(desired_link)
            time.sleep(1)
            # decide whether it is an video or document
            # if it is an video, do not download it
            soup2 = BeautifulSoup(browser.page_source, "lxml")
            video_list = soup2.find_all('a', id = "player-video")
            if len(video_list) != 0:
                print("Skip downloading the video and go to next item!")
                try:
                    browser.get(old_url)
                    time.sleep(5)
                except:
                    print("The browser has already been @ the search result page!")
                continue
            # otherwise, download the document directly
            else:
                print('Start to download ...')
                download_button = browser.find_element_by_id("downagain_wap")
                time.sleep(1)
                try:
                    download_button.click()
                    time.sleep(10)
                    print("Finish download!")
                    print(" ")
                    try:
                        browser.get(old_url)
                        time.sleep(5)
                    except:
                        print("The browser has already been @ the search result page!")
                    continue
                except:
                    # this case means the browser has go to another page
                    # in this case, we have to close it and open the search result page again
                    time.sleep(10)
                    print("Finish download!")
                    print("Restart browser!")
                    browser.quit()
                    browser = initialization(old_url)
                    time.sleep(2)
                    login_huawei(browser)

        # go to next page
        try:
            time.sleep(5)
            button = browser.find_element_by_class_name('nextPage')
            time.sleep(1)
            button.click()
            new_url = browser.current_url
            if old_url != new_url:
                page_number += 1
            else:
                page_tag = False
                print("No more pages available")
        # stop when no pages are available
        except Exception as e:
            print(e)
            page_tag = False
            print("No more pages available")


if __name__ == '__main__':
    # download documents in English
    base_url = "http://e.huawei.com/enterprisesearch/?lang=en#keyword=Smart+city&lang=us&outside=0&searchType=&type=EBG_MM"

    # download docuemnts in Chinese
    base_url = "http://e.huawei.com/enterprisesearch/?lang=zh#keyword=%E6%99%BA%E6%85%A7%E5%9F%8E%E5%B8%82&lang=zh&outside=0&searchType=&type=EBG_MM"

    conduct_download(base_url)
    pass
















