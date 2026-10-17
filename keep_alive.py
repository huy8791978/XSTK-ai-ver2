"""Mở app bằng trình duyệt thật; nếu app đang ngủ thì bấm nút đánh thức. Chạy trong GitHub Actions."""
import os
import re
import sys

from playwright.sync_api import sync_playwright

url = os.environ.get("APP_URL", "").strip()
if not url:
    sys.exit("Chưa đặt APP_URL trong GitHub Secrets hoặc Variables.")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(10_000)
    wake = page.get_by_role("button", name=re.compile(r"get this app back up", re.I))
    if wake.count():
        print("App đang ngủ: đang đánh thức...")
        wake.first.click()
        page.wait_for_timeout(90_000)
        print("Đã gửi yêu cầu đánh thức.")
    else:
        print("App đang hoạt động.")
    browser.close()
