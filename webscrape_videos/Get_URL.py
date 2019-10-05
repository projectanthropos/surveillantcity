# This script will get the url with the key words: yellow vest ruptly
# We can set up how many videos we want by changing the value of "max_number"
# The result will be a csv file in "/data" with two columns: title & link

# import packages
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time

# set up basic url with key words: yellow vest ruptly
base_url = "https://www.youtube.com/results?search_query=yellow+vest+ruptly"
browser = webdriver.Chrome()
browser.get(base_url)

# the videos we want
max_number = 12

# the initial scroll step
base_step = 1

#==========================================
# Name: generate_title_link_dict
# Purpose: This function will get the title and the link for each video
#          from the youtube.com search result. Since we only interest in
#          "Ruptly", thus this function will filter out others and only keep
#          "Ruptly" videos and save their titles & links into a dictionary
# Input Parameter(s): scroll_step controls the scroll step
# Return Value(s): A dictionary with "Ruptly" videos' titles & links
#============================================
def generate_title_link_dict(scroll_step):
    # This one uses to scroll page
    for i in range(scroll_step):
        browser.execute_script("scroll(0,{})".format(i))
    # wait for 20 seconds to load the whole page
    time.sleep(20)

    # select the required nodes from source page
    soup = BeautifulSoup(browser.page_source, "lxml")
    title_link_list = soup.find_all(class_="yt-simple-endpoint style-scope ytd-video-renderer")
    ruptly_tag_list = soup.find_all(class_="yt-simple-endpoint style-scope yt-formatted-string")

    # get the number of videos
    list_length = len(title_link_list)
    print("total videos: " + str(list_length))

    # dictionary to save results
    title_link_dict = {'title':[], 'link':[]}

    # use for loop to get the "Ruptly" videos' titles & links
    # save them into the dictionary
    for i in range(list_length):
        tl_item = title_link_list[i]
        tl_title = tl_item.attrs['title']
        tl_url = urljoin(base_url,tl_item.attrs['href'])

        ruptly_item = ruptly_tag_list[i]
        ruptly_tag = ruptly_item.string

        if ruptly_tag == "Ruptly":
            current_link = title_link_dict['link']
            if tl_url not in current_link:
                title_link_dict['title'].append(tl_title)
                title_link_dict['link'].append(tl_url)
    return len(title_link_dict['link']), title_link_dict


#==========================================
# Name: generate_csv
# Purpose: After we have at least max_number videos we want,
#          then generate a csv file which includes video's
#          titles & video's links
# Input Parameter(s): max_number indicates the number of videos we want to get;
#                     base_step indicates the initial scroll step
# Return Value(s): None (Generate a csv file under "/data")
#============================================
def generate_csv(max_number, base_step):
    scroll_step = base_step
    control, result_dict = generate_title_link_dict(scroll_step)

    # use while loop to dynamically scroll down page to get enough videos
    while control < max_number:
        scroll_step = scroll_step*10
        control, result_dict = generate_title_link_dict(scroll_step)
        print("current scroll step: " + str(scroll_step))
        print("current dict length: " + str(control))
        print(" ")
    # save result to a csv file
    result_DF = pd.DataFrame.from_dict(result_dict)
    result_DF.to_csv("data/videos_link.csv", index=False)

if __name__ == '__main__':
    generate_csv(max_number, base_step)






