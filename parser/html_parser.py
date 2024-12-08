import time
import os
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def setup_chrome_driver(chrome_driver_path):
    """Sets up and returns a Chrome WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-software-rasterizer')
    chrome_options.add_argument('--remote-debugging-port=9222')
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.64 Safari/537.36")
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")
    service = Service(executable_path=chrome_driver_path)
    return webdriver.Chrome(service=service, options=chrome_options)


def save_page(driver, url, folder_path, page_number):
    """Downloads and saves an HTML page."""
    print(f"Loading page: {url}")
    driver.get(url)
    time.sleep(15)
    filename = os.path.join(folder_path, f"page_{page_number}.html")
    with open(filename, "w", encoding='utf-8') as file:
        file.write(driver.page_source)
        print(f"Page saved: {filename}")
    return filename


def extract_article_urls(driver, base_url):
    """Extracts article URLs from the base page."""
    print(f"Loading page: {base_url}")
    driver.get(base_url)
    print("Waiting 15 seconds for the page to load...")
    time.sleep(15)
    links = driver.find_elements(By.TAG_NAME, "a")
    urls = [link.get_attribute('href')
            for link in links if link.get_attribute('href')]
    article_urls = [url for url in urls if '/handbook/ml/article/' in url]
    print(f"Found {len(article_urls)} article links.")
    return article_urls


def save_mappings(url_to_filename, filename_to_url, folder_path):
    """Saves mappings to JSON files."""
    url_to_filename_path = os.path.join(folder_path, "url2filename.json")
    filename_to_url_path = os.path.join(folder_path, "filename2url.json")

    with open(url_to_filename_path, "w", encoding='utf-8') as json_file:
        json.dump(url_to_filename, json_file, ensure_ascii=False, indent=4)
    print(f"URL - filename mapping saved to {url_to_filename_path}.")

    with open(filename_to_url_path, "w", encoding='utf-8') as json_file:
        json.dump(filename_to_url, json_file, ensure_ascii=False, indent=4)
    print(f"Filename - URL mapping saved to {filename_to_url_path}.")


def main():
    CHROME_DRIVER = '/usr/local/bin/chromedriver'  # chrome driver path
    BASE_URL = ""  # link removed, please replace with actual link before use
    SAVE_FOLDER = "yandex_ml_handbook_test"

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    driver = setup_chrome_driver(CHROME_DRIVER)

    try:
        article_urls = extract_article_urls(driver, BASE_URL)
        url_to_filename = {}
        filename_to_url = {}

        for i, article_url in enumerate(article_urls, start=1):
            filename = save_page(driver, article_url, SAVE_FOLDER, i)
            url_to_filename[article_url] = filename
            filename_to_url[filename] = article_url

        save_mappings(url_to_filename, filename_to_url, SAVE_FOLDER)
    finally:
        driver.quit()
        print("Process completed.")


if __name__ == "__main__":
    main()
