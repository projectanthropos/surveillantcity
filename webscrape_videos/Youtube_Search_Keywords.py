from selenium import webdriver
import time

key_words = 'Yellow Vest Ruptly'
browser = webdriver.Chrome()
browser.get("https://www.youtube.com/")
input = browser.find_element_by_id('search')
input.send_keys(key_words)

button = browser.find_element_by_id('search-icon-legacy')
# It is necessary to set up the waiting time here to make the button interactable
time.sleep(1)
button.click()