# This script will fetch pdf documents from google search with the key words: huawei smart cities
# import packages
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import os

# set up the basic url with key words
# keywords: huawei smart cities
base_url = "https://www.google.com/search?q=huawei+smart+cities&oq=huawei+smart+cities&aqs=chrome.0.69i59j69i60j0l4.6926j0j7&sourceid=chrome&ie=UTF-8"

# keywords: 华为　智慧城市
# base_url = "https://www.google.com/search?q=%E5%8D%8E%E4%B8%BA+%E6%99%BA%E6%85%A7%E5%9F%8E%E5%B8%82&oq=%E5%8D%8E%E4%B8%BA%E3%80%80%E6%99%BA%E6%85%A7%E5%9F%8E%E5%B8%82&aqs=chrome..69i57j0j69i61l3.13490j0j7&sourceid=chrome&ie=UTF-8"

browser = webdriver.Chrome()
browser.get(base_url)

# a flag to determine whether the last page is reached or not
page_tag = True

# while loop to process every page from google search result
while page_tag:
    # process current page
    print(browser.current_url)
    if browser.current_url != base_url:
        browser.get(browser.current_url)
        time.sleep(10)
    soup = BeautifulSoup(browser.page_source, "lxml")
    item_list = soup.find_all(class_="r")
    for item in item_list:
        item_a_node = item.find_all('a')[0]
        item_link = item_a_node.attrs['href']
        # download pdf and save it to /data
        save_path = os.getcwd() + '/data'
        if '.pdf' in item_link:
            os.system('wget -P {} '.format(save_path) + item_link)

    # go to next page
    try:
        button = browser.find_element_by_id('pnnext')
        button.click()
    # stop when no pages are available
    except:
        page_tag = False
        print("No more pages available")

if __name__ == '__main__':
    pass







