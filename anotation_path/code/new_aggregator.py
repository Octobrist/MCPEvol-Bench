import sqlite3
import threading
import time
import json
import re
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import feedparser
from datetime import datetime

# ======================
# 1. 数据库初始化
# ======================
DB_PATH = "news.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS news
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  title
                  TEXT
                  NOT
                  NULL,
                  link
                  TEXT
                  UNIQUE
                  NOT
                  NULL,
                  summary
                  TEXT,
                  published
                  TEXT,
                  source
                  TEXT,
                  created_at
                  TIMESTAMP
                  DEFAULT
                  CURRENT_TIMESTAMP
              )
              """)
    conn.commit()
    conn.close()


# ======================
# 2. 新闻爬虫（支持多源并发）
# ======================
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "https://rss.cnn.com/rss/edition.rss"
]


def clean_html(raw_html):
    """移除HTML标签"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()


def fetch_feed(url, results, lock):
    """抓取单个RSS源"""
    try:
        parsed = feedparser.parse(url)
        source = urlparse(url).netloc
        entries = []
        for entry in parsed.entries[:5]:  # 每源取前5条
            title = getattr(entry, 'title', 'No Title')
            link = getattr(entry, 'link', '')
            summary = clean_html(getattr(entry, 'summary', ''))
            published = getattr(entry, 'published', '')
            entries.append({
                'title': title,
                'link': link,
                'summary': summary[:200],
                'published': published,
                'source': source
            })
        with lock:
            results.extend(entries)
        print(f"[+] Fetched {len(entries)} items from {source}")
    except Exception as e:
        print(f"[-] Error fetching {url}: {e}")


def crawl_news():
    """并发抓取所有RSS源"""
    threads = []
    results = []
    lock = threading.Lock()

    for url in RSS_FEEDS:
        t = threading.Thread(target=fetch_feed, args=(url, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10)  # 超时防止卡死

    # 去重（按链接）
    seen = set()
    unique_news = []
    for item in results:
        if item['link'] not in seen and item['link']:
            seen.add(item['link'])
            unique_news.append(item)

    return unique_news


# ======================
# 3. 数据库存储
# ======================
def save_to_db(news_list):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    inserted = 0
    for news in news_list:
        try:
            c.execute("""
                      INSERT
                      OR IGNORE INTO news (title, link, summary, published, source)
                VALUES (?, ?, ?, ?, ?)
                      """, (news['title'], news['link'], news['summary'], news['published'], news['source']))
            inserted += 1
        except Exception as e:
            print(f"DB insert error: {e}")
    conn.commit()
    conn.close()
    print(f"[✓] Saved {inserted} new articles to DB")


# ======================
# 4. RESTful API 服务
# ======================
class NewsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/news':
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row  # 启用列名访问
            c = conn.cursor()
            c.execute("SELECT * FROM news ORDER BY created_at DESC LIMIT 20")
            rows = c.fetchall()
            conn.close()

            news_list = [dict(row) for row in rows]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # 允许跨域（测试用）
            self.end_headers()
            self.wfile.write(json.dumps(news_list, indent=2, default=str).encode())
        else:
            self.send_error(404)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """支持多线程的HTTP服务器"""
    pass


# ======================
# 5. 主程序入口
# ======================
def main():
    print("🚀 初始化新闻聚合器...")
    init_db()

    # 启动后台爬虫（每10分钟更新一次）
    def background_crawler():
        while True:
            print("\n🔄 开始抓取新闻...")
            news = crawl_news()
            save_to_db(news)
            print(f"💤 等待10分钟... (按 Ctrl+C 停止)")
            time.sleep(600)  # 10分钟

    crawler_thread = threading.Thread(target=background_crawler, daemon=True)
    crawler_thread.start()

    # 启动API服务器
    server = ThreadedHTTPServer(('localhost', 8080), NewsHandler)
    print("\n📡 REST API 已启动: http://localhost:8080/api/news")
    print("   (在浏览器或 curl 中访问即可查看最新新闻)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()