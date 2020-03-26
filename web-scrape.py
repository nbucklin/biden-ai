from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as Wait
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException


driver = webdriver.Chrome(ChromeDriverManager().install())
#driver= webdriver.Chrome("/selenium/webdriver/chromedriver")
driver.get('https://obamawhitehouse.archives.gov/briefing-room/speeches-and-remarks')

n = 0
while n < 473:
    
    lems = driver.find_elements_by_xpath("//a[@href]")
    
    links = []
    for elm in lems:
        links.append(elm.get_attribute("href"))
        
    links1 = []
    for l in links:
        result = l.find('press-office')
        if result > 0:
            links1.append(l)
            
    links2 = []
    for l in links1:
        result = l.find('biden')
        if result > 0:
            links2.append(l)
    
    biden_speeches = []
    for l in links2:
        driver.get(l)
        txt = driver.find_element_by_tag_name('body')
        biden_speeches.append(txt.text)
        
    with open("biden-speeches.txt", "a") as f:
        f.write('\n'.join(biden_speeches))
        
    n = n + 1
            
    driver.get('https://obamawhitehouse.archives.gov/briefing-room/speeches-and-remarks?term_node_tid_depth=31&page={}'.format(n))
    
    
