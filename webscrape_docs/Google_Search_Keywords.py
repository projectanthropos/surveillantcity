from selenium import webdriver
import time

key_words = 'Huawei Smart Cities'
browser = webdriver.Chrome()
browser.get("https://www.google.com/")
input = browser.find_element_by_name('q')
input.send_keys(key_words)

button = browser.find_element_by_name('btnK')
# or button = browser.find_element_by_class_name('gNO89b')
# It is necessary to set up the waiting time here to make the button interactable
time.sleep(1)
button.click()