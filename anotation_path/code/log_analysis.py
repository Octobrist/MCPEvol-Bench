import threading
import queue
import time
import random
import os
import signal
import sys
from datetime import datetime
from collections import defaultdict, deque

# ======================
# 配置
# ======================
LOG_FILE = "app.log"
MAX_LOG_LINES = 1000  # 日志文件最大行数（滚动）
ANALYSIS_WINDOW = 60  # 统计最近60秒的数据
REFRESH_INTERVAL = 1.5  # 控制台刷新间隔（秒）

# 全局控制标志
shutdown_event = threading.Event()


# ======================
# 1. 日志生成器（生产者）
# ======================
class LogProducer:
    def __init__(self, log_queue):
        self.queue = log_queue
        self.levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
        self.modules = ["auth", "db", "api", "cache", "worker"]

    def generate_log_line(self):
        level = random.choices(self.levels, weights=[50, 20, 10, 20])[0]
        module = random.choice(self.modules)
        message = f"User{random.randint(1000, 9999)} performed action on {module}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] {level} [{module}] {message}"

    def run(self):
        with open(LOG_FILE, "a", buffering=1) as f:  # 行缓冲
            while not shutdown_event.is_set():
                line = self.generate_log_line()
                print(f"📝 Produced: {line[:60]}...")
                f.write(line + "\n")
                self.queue.put(line)
                time.sleep(random.uniform(0.3, 1.2))  # 模拟不均匀日志流


# ======================
# 2. 日志分析器（消费者）
# ======================
class LogAnalyzer:
    def __init__(self):
        self.level_count = defaultdict(int)
        self.module_count = defaultdict(int)
        self.error_timeline = deque(maxlen=ANALYSIS_WINDOW)  # 最近60秒每秒错误数
        self.last_update = time.time()

    def process_line(self, line):
        # 简单解析：[时间] LEVEL [module] ...
        try:
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                level = parts[1]
                module_part = parts[2]
                if module_part.startswith("[") and "]" in module_part:
                    module = module_part[1:module_part.index("]")]
                    self.level_count[level] += 1
                    self.module_count[module] += 1
                    if level == "ERROR":
                        now = int(time.time())
                        # 初始化时间线
                        while len(self.error_timeline) < now - self.last_update:
                            self.error_timeline.append(0)
                        if self.error_timeline:
                            self.error_timeline[-1] += 1
                        self.last_update = now
        except Exception as e:
            print(f"⚠️ 解析错误: {e}")

    def get_stats(self):
        total = sum(self.level_count.values())
        error_rate = (self.level_count["ERROR"] / total * 100) if total > 0 else 0
        recent_errors = sum(self.error_timeline)
        return {
            "total_logs": total,
            "error_rate": error_rate,
            "recent_errors": recent_errors,
            "top_module": max(self.module_count.items(), key=lambda x: x[1], default=("none", 0))[0],
            "level_breakdown": dict(self.level_count)
        }


# ======================
# 3. 控制台渲染器
# ======================
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def render_dashboard(analyzer):
    stats = analyzer.get_stats()
    clear_console()
    print("=" * 60)
    print("📊 本地日志实时分析系统 (按 Ctrl+C 退出)")
    print("=" * 60)
    print(f"总日志数      : {stats['total_logs']}")
    print(f"错误率        : {stats['error_rate']:.2f}%")
    print(f"最近60秒错误  : {stats['recent_errors']}")
    print(f"最活跃模块    : {stats['top_module']}")
    print("\n日志级别分布:")
    for level, count in sorted(stats['level_breakdown'].items()):
        bar = "█" * (count // 10)  # 简易进度条
        print(f"  {level:8} : {count:4} {bar}")
    print("\n💡 提示: 日志文件位置 ->", os.path.abspath(LOG_FILE))
    print("=" * 60)


# ======================
# 4. 主控制器
# ======================
def main():
    # 注册信号处理器（优雅退出）
    def signal_handler(sig, frame):
        print("\n🛑 收到退出信号，正在关闭...")
        shutdown_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 初始化
    log_queue = queue.Queue(maxsize=100)
    analyzer = LogAnalyzer()

    # 启动生产者线程
    producer = LogProducer(log_queue)
    producer_thread = threading.Thread(target=producer.run, daemon=True)
    producer_thread.start()

    # 主循环：消费日志 + 刷新UI
    last_render = 0
    try:
        while not shutdown_event.is_set():
            # 处理队列中的日志
            while not log_queue.empty():
                try:
                    line = log_queue.get_nowait()
                    analyzer.process_line(line)
                except queue.Empty:
                    break

            # 定期刷新控制台
            if time.time() - last_render > REFRESH_INTERVAL:
                render_dashboard(analyzer)
                last_render = time.time()

            time.sleep(0.1)  # 避免忙等待

    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        print("\n✅ 系统已安全退出")


if __name__ == "__main__":
    main()