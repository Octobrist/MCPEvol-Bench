import re
import json
import requests
from bs4 import BeautifulSoup

def extract_mcp_info_from_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')

    # 1. 定位包含 __detail_data__ 的 script 标签
    script_tag = soup.find('script', string=re.compile(r'window\.__detail_data__'))
    if not script_tag:
        return None

    # 2. 提取 JSON 字符串并解析
    match = re.search(r'window\.__detail_data__\s*=\s*(".*?")\s*;', script_tag.string, re.DOTALL)
    if not match:
        return None

    json_str = match.group(1)
    # 处理转义：去除外层引号，并将 \" 还原为 "
    try:
        clean_json_str = json.loads(json_str)  # 这会自动处理 \" → "
    except json.JSONDecodeError:
        # 如果上述失败，尝试手动清理（某些情况下有尾部空格）
        json_str = json_str.strip()
        if json_str.startswith('"') and json_str.endswith('"'):
            json_str = json_str[1:-1]
        json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
        clean_json_str = json.loads(json_str)
    try:
        data = json.loads(clean_json_str)
    except json.JSONDecodeError:
        return clean_json_str

    selected_keys = [
    'Abstract', 'AbstractCN', 'AlreadyStar', 'CallVolume', 'ChineseName',
    'FromSiteUrl', 'Hosted', 'Id', 'IsTop', 'Name', 'OriginalReadme',
    'Publisher', 'Readme', 'ReadmeCN', 'ServerConfig', 'Stars',
    'StreamableHTTPParameterSchema', 'StreamableHTTPServerConfig',
    'Tags', 'Tools', 'UserHostStatus', 'Verifed'
    ]
    filtered_data = {key: data.get(key, None) for key in selected_keys if key in data}

    return filtered_data

# 示例使用
if __name__ == "__main__":
    url = "https://www.modelscope.cn/mcp/servers/slcatwujian/bing-cn-mcp-server"
    response = requests.get(url)
    html_text = response.text
    info = extract_mcp_info_from_html(html_text)

    print("\n=== 工具列表 ===")
    for t in info["Tools"]:
        print(f"- **{t['name']}**: {t['description']}")