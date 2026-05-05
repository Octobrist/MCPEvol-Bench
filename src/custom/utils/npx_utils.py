import requests
import time
import random
import sys

# --- 反爬配置（你已提供）---
REQUEST_DELAY_RANGE = (2.0, 5.0)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/json",  # npm registry 返回 JSON，建议改为此
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
})

def get_npm_package_info(package_name: str, pop_readme: bool = True):
    if package_name.startswith('@'):
        encoded = package_name.replace('/', '%2f')
    elif '@' in package_name:
        encoded = package_name.split('@')[0]
    else:
        encoded = package_name
    url = f"https://registry.npmjs.org/{encoded}"

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        response_json = response.json()
        if pop_readme:
            response_json.pop('readme')
        return response_json
    except Exception as e:
        print(f"Error fetching {package_name}: {e}", file=sys.stderr)
        return None
    finally:
        # 请求后延迟
        delay = random.uniform(*REQUEST_DELAY_RANGE)
        time.sleep(delay)


def search_npm_packages(keyword, size=250, timeout=10):
    """
    :param keyword: 搜索关键词（如 "react", "axios"）
    :param size: 返回结果数量（1~250）
    :param timeout: 请求超时（秒）
    :return: 包对象列表，每个包含 'package' 字段
    """
    BASE_SEARCH_URL = "https://registry.npmjs.org/-/v1/search"
    # if not (1 <= size <= 250):
    #     raise ValueError("size 必须在 1 到 250 之间")

    params = {
        'text': keyword.strip(),
        'size': size,
        'from': 0
    }

    try:
        response = session.get(BASE_SEARCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get('objects', [])
    except requests.RequestException as e:
        print(f"❌ 请求出错: {e}", file=sys.stderr)
        return []
    except ValueError as e:
        print(f"❌ JSON 解析失败: {e}", file=sys.stderr)
        return []