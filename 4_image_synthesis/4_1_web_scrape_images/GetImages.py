# THIS SCRIPT WILL DOWNLOAD IMAGES FROM INTERNET

# IMPORT PACKAGES
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from urllib import parse
import time
import os

# DEFINE A CLASS CALLED WSImage TO CONDUCT THE TASK OF WEB SCRAPING
class WSImage:
    # ==========================================
    # Name: __init__
    # Purpose: CONSTRUCTOR AND INITIALIZE THE ATTRIBUTES OF THE CLASS
    # Input Parameter(s): base_url --- VALUE TO INITIALIZE self.base_url
    #                     count --- VALUE TO INITIALIZE self.count
    # Return Value(s): NONE
    # ============================================
    def __init__(self, base_url, count, pagenum):
        self.base_url = base_url
        self.count = count
        self.curpage = pagenum


    # ==========================================
    # Name: get_browser
    # Purpose: ACCORDING TO URL, CREATE THE OBJECT OF BROWSER
    # Input Parameter(s): url --- THE URL USED TO CREATE THE BROWSER OBJECT
    # Return Value(s): browser --- THE CREATED BROWSER OBJECT
    # ============================================
    def get_browser(self, url):
        browser = webdriver.Chrome()
        browser.set_page_load_timeout(100)
        browser.get(url)
        if url == self.base_url:
            # START TO SEARCH
            search_box = browser.find_element_by_class_name('q')
            search_box.send_keys('city')
            search_box.send_keys(Keys.ENTER)
        time.sleep(20)
        return browser

    # ==========================================
    # Name: download_images
    # Purpose: ACTUALLY DOWNLOAD THE IMAGES BY CALLING FUNCTION download() AND ALSO GET THE NEXT PAGE LINK
    # Input Parameter(s): browser --- THE CURRENT BROWSER OBJECT HANDLE
    # Return Value(s): next_link --- THE LINK FOR NEXT PAGE
    # ============================================
    def download_images(self, browser):
        # CREAT BeautifulSoup OBJECT USING lxml PARSER
        soup = BeautifulSoup(browser.page_source, "lxml")

        # FIND ELEMENTS THAT CONTAIN IMAGES
        cur_page = soup.find_all('div', class_="flex_grid credits search_results")
        item_list = cur_page[0].find_all('div', class_="item")
        item_list_len = len(item_list)
        print('The total images are ' + str(item_list_len))

        # USING FOR LOOP TO DOWNLOAD IMAGES ONE-BY-ONE
        for item in item_list:
            # FIND THE IMAGE LINK
            cur_obj = item.find_all('a')
            cur_image_url = parse.urljoin('https://pixabay.com/', cur_obj[0].attrs['href'])

            # CALL download FUNCTION TO DOWNLOAD THE IMAGE
            self.download(cur_image_url)
            self.count += 1
            print('The number of images:')
            print(self.count)
        print('All images in this page have been saved successfully!')
        print('Next page will be visited!')

        # AFTER DOWNLOADING ALL ITEMS
        # THEN, TO GET THE NEXT PAGE WEBSITE
        total_btn = soup.find_all('a', class_="pure-button")
        for btn in total_btn:
            btn_str = btn.string
            if btn_str == '›':
                next_link = parse.urljoin('https://pixabay.com/', btn.attrs['href'])
                self.curpage += 1
                break
        # close current web browser
        browser.quit()
        return next_link


    # ==========================================
    # Name: download
    # Purpose: DOWNLOAD IMAGE ACCORDING TO THE PROVIDED LINK/URL
    # Input Parameter(s): url --- THE IMAGE LINK/URL
    # Return Value(s): NONE
    # ============================================
    def download(self,url):
        if self.curpage < 121:
            return
        try:
            temp_browser = webdriver.Chrome()
            temp_browser.set_page_load_timeout(100)
            temp_browser.get(url)
            soup = BeautifulSoup(temp_browser.page_source, "lxml")
            cur_page = soup.find_all('div', id='media_container')
            cur_obj = cur_page[0].find_all('img')
            link_str = cur_obj[0].attrs['src']
            save_path = os.getcwd() + '/images'
            os.system('wget -P {} '.format(save_path) + link_str)
            time.sleep(10)
            temp_browser.quit()
        except:
            print('Cannot download this images!')
            temp_browser.quit()

    # ==========================================
    # Name: run_wsi
    # Purpose: THE MAIN FUNCTION IN THIS CLASS
    # Input Parameter(s): NONE
    # Return Value(s): NONE
    # ============================================
    def run_wsi(self):
        # INITIALIZATION
        cur_url = ''
        next_url = self.base_url
        # USING WHILE LOOP TO PROCESS ALL PAGES
        while cur_url != next_url:
            browser = self.get_browser(next_url)
            cur_url = browser.current_url
            next_url = self.download_images(browser)
        print('All pages have been visited successfully!')
        print('The number of images has been downloaded is:' + str(self.count))

if __name__ == '__main__':
    base_url = 'https://pixabay.com/'
    wsi = WSImage(base_url,0, 0)
    wsi.run_wsi()
    pass



