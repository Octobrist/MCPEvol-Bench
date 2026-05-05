import subprocess
import time
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_clear_npx_cache():
    """运行 npx clear-npx-cache 并自动输入 'y'"""
    try:
        logging.info("正在运行: npx clear-npx-cache")

        # 使用 Popen 启动进程，以便发送输入
        process = subprocess.Popen(
            ['npx', 'clear-npx-cache'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        # 发送 'y\n' 作为输入（模拟用户输入 y 并回车）
        stdout, stderr = process.communicate(input='y\n', timeout=3600)

        if process.returncode == 0:
            logging.info("✅ npx clear-npx-cache 成功执行")
            if stdout.strip():
                logging.debug(f"输出:\n{stdout}")
        else:
            logging.error(f"❌ 命令失败，退出码: {process.returncode}")
            if stderr.strip():
                logging.error(f"错误输出:\n{stderr}")

    except subprocess.TimeoutExpired:
        logging.error("⚠️ 命令超时（30秒），强制终止")
        process.kill()
        stdout, stderr = process.communicate()
    except FileNotFoundError:
        logging.error("❌ 'npx' 命令未找到，请确保 Node.js 已安装并加入 PATH")
    except Exception as e:
        logging.error(f"💥 发生异常: {e}")


def main():
    logging.info("🚀 启动 npx 缓存清理守护程序（每 0.1 小时运行一次）")
    logging.info("按 Ctrl+C 可停止程序")

    try:
        while True:
            run_clear_npx_cache()
            logging.info("⏳ 等待 0.1 小时后下一次清理...")
            time.sleep(0.1 * 60 * 60)  # 1 小时 = 3600 秒
    except KeyboardInterrupt:
        logging.info("🛑 程序被用户终止")
        sys.exit(0)


if __name__ == "__main__":
    main()