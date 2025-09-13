"""import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests

SAVE_XLS = "psp_xls_reports" 
SAVE_PDF = "psp_pdf_reports"
os.makedirs(SAVE_XLS, exist_ok=True)
os.makedirs(SAVE_PDF, exist_ok=True)

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Open PSP page
driver.get("https://grid-india.in/en/reports/daily-psp-report")
time.sleep(5)  # wait for JS table to load

# Click "Next" until no more pages
while True:
    links = driver.find_elements(By.XPATH, "//a[contains(@href, '.xls') or contains(@href, '.pdf')]")
    for link in links:
        file_url = link.get_attribute("href")
        if file_url:
            filename = os.path.basename(file_url)
            save_path = os.path.join(SAVE_XLS if filename.endswith(".xls") else SAVE_PDF, filename)

            if not os.path.exists(save_path):  # avoid re-downloading
                r = requests.get(file_url, timeout=20)
                if r.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(r.content)
                    print(f"✅ Downloaded {filename}")
    
    # Try going to next page
    try:
        next_btn = driver.find_element(By.XPATH, "//a[contains(@class,'next')]")
        if "disabled" in next_btn.get_attribute("class"):
            break
        next_btn.click()
        time.sleep(3)
    except:
        break

driver.quit()
"""

import os  # Provides functions for interacting with the operating system
import time  # Used for adding delays in execution
import requests  # Library for making HTTP requests
from selenium import webdriver  # Main Selenium WebDriver class for browser automation
from selenium.webdriver.chrome.options import Options  # Allows setting Chrome options
from selenium.webdriver.common.by import By  # Used to specify how to locate elements
from selenium.webdriver.support.ui import Select  # For interacting with dropdown menus
from selenium.webdriver.support.ui import WebDriverWait  # For waiting for elements to load
from selenium.webdriver.support import expected_conditions as EC  # For specifying wait conditions

SAVE_XLS = "data/psp_xls_reports"  # Directory to save XLS files
SAVE_PDF = "data/psp_pdf_reports"  # Directory to save PDF files
os.makedirs(SAVE_XLS, exist_ok=True)  # Create XLS directory if it doesn't exist
os.makedirs(SAVE_PDF, exist_ok=True)  # Create PDF directory if it doesn't exist

options = Options()  # Create Chrome options object
options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
driver = webdriver.Chrome(options=options)  # Start Chrome browser with options

driver.get("https://grid-india.in/en/reports/daily-psp-report")  # Open the target webpage
try:
    # Wait up to 15 seconds for the dropdown to appear
    select_elem = WebDriverWait(driver, 15).until(  # Wait for dropdown to be present
        EC.presence_of_element_located((By.NAME, "DataTables_Table_0_length"))  # Locate dropdown by name
    )
    Select(select_elem).select_by_visible_text("100")  # Set dropdown to show 100 entries
    time.sleep(3)  # Wait for table to refresh
except Exception as e:
    print(f"⚠️ Could not find 'Show entries' dropdown: {e}")  # Print error if dropdown not found
    driver.quit()  # Close browser
    exit(1)  # Exit script

# Step 2 — Find total pages
page_buttons = driver.find_elements(By.XPATH, "//a[@class='page-link' and not(contains(@class,'previous')) and not(contains(@class,'next'))]")  # Find all page number buttons
total_pages = int(page_buttons[-1].text)  # Get the last page number (total pages)
print(f"📄 Found {total_pages} pages (100 entries each)")  # Print total pages found

# Step 3 — Loop through each page
for page in range(1, total_pages + 1):  # Loop through each page
    print(f"🔹 Scraping page {page}/{total_pages}")  # Print current page number

    # Grab all file links on this page
    links = driver.find_elements(By.XPATH, "//a[contains(@href, '.xls') or contains(@href, '.pdf')]")  # Find all XLS and PDF links
    for link in links:  # Loop through each link
        file_url = link.get_attribute("href")  # Get the URL of the file
        if not file_url:  # Skip if no URL
            continue
        filename = os.path.basename(file_url)  # Extract filename from URL
        save_path = os.path.join(SAVE_XLS if filename.endswith(".xls") else SAVE_PDF, filename)  # Set save path based on file type

        if not os.path.exists(save_path):  # Download only if file doesn't exist
            try:
                r = requests.get(file_url, timeout=15)  # Download the file
                if r.status_code == 200:  # Check if download was successful
                    with open(save_path, "wb") as f:  # Open file for writing
                        f.write(r.content)  # Write file content
                    print(f"✅ Downloaded {filename}")  # Print success message
                else:
                    print(f"❌ Failed: {filename}")  # Print failure message
            except Exception as e:
                print(f"⚠️ Error downloading {filename}: {e}")  # Print error message

    # Go to next page if not last
    if page < total_pages:  # If not on last page
        next_btn = driver.find_element(By.XPATH, "//a[contains(@class,'next')]")  # Find 'Next' button
        driver.execute_script("arguments[0].click();", next_btn)  # Click 'Next' using JS
        time.sleep(3)  # Wait for page to load

driver.quit()  # Close the browser and end the session
