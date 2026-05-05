import numpy as np
import json
import random
from sentence_transformers import SentenceTransformer

# ----------------配置----------------
MODEL_PATH = '/XYAIFS00/HDD_POOL/nudt_dwfeng/nudt_dwfeng_1/lhx/llm/bge-m3'
INPUT_FILE = 'diff_results.json'
BATCH_SIZE = 32
RANDOM_SEED = 42  # 保证打混结果可复现


# ----------------------------------

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    return SentenceTransformer(MODEL_PATH, device='cuda')


def diff_lines_to_text(diff_lines: list) -> str:
    """
    将 Diff 行列表转换为纯文本字符串。
    去除 @@ 元数据行，保留代码变更语义。
    """
    if not diff_lines or not isinstance(diff_lines, list):
        return ""

    filtered_lines = []
    for line in diff_lines:
        if isinstance(line, str) and line.startswith("@@") and line.endswith("@@"):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def main():
    # 1. 加载数据
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Successfully loaded {len(results)} records.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not results:
        print("No data to process.")
        return

    # 2. 预处理：提取文本
    texts_evo = []
    texts_small = []
    texts_big = []
    print(len(results))
    print("Preprocessing diff texts...")
    for row in results:
        texts_evo.append(diff_lines_to_text(row.get("evolution_diff", [])))
        texts_small.append(diff_lines_to_text(row.get("small_diff", [])))
        texts_big.append(diff_lines_to_text(row.get("big_diff", [])))

    # 3. 加载模型
    model = load_model()

    # 4. 批量生成 Embeddings
    print("Generating embeddings...")
    emb_evo = model.encode(texts_evo, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)
    emb_small = model.encode(texts_small, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=True)
    emb_big = model.encode(texts_big, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)

    # 5. 计算原始配对相似度 (Ground Truth Pairs)
    small_big_original = np.sum(emb_big * emb_small, axis=1)
    sim_evo_small_original = np.sum(emb_evo * emb_small, axis=1)
    sim_evo_big_original = np.sum(emb_evo * emb_big, axis=1)

    # 6. 【核心步骤】打混计算相似度 (Shuffled/Mixed Pairs)
    print("Calculating shuffled similarities...")

    # 设置随机种子，确保每次运行打混顺序一致
    random.seed(RANDOM_SEED)

    # 创建索引列表并打乱
    indices = list(range(len(results)))
    random.shuffle(indices)

    # 使用打乱后的索引获取 small 和 big 的向量
    # 这意味着: 第 i 个 evolution 现在和第 indices[i] 个 small 进行对比
    emb_small_shuffled = emb_small[indices]
    emb_big_shuffled = emb_big[indices]

    # 计算打混后的相似度
    sim_evo_small_shuffled = np.sum(emb_evo * emb_small_shuffled, axis=1)
    sim_evo_big_shuffled = np.sum(emb_evo * emb_big_shuffled, axis=1)

    # 7. 统计平均值
    avg_orig_evo_small = np.mean(sim_evo_small_original)
    avg_orig_evo_big = np.mean(sim_evo_big_original)
    avg_orig_small_big = np.mean(small_big_original)

    avg_shuffle_evo_small = np.mean(sim_evo_small_shuffled)
    avg_shuffle_evo_big = np.mean(sim_evo_big_shuffled)

    # 8. 输出结果
    print("\n" + "=" * 40)
    print("Similarity Comparison: Original vs Shuffled")
    print("=" * 40)
    print(f"Sample Count: {len(results)}")
    print("-" * 40)

    print("1. Evolution vs Small Diff:")
    print(f"   Original Pairing Avg Sim: {avg_orig_evo_small:.4f}")
    print(f"   Shuffled Pairing Avg Sim: {avg_shuffle_evo_small:.4f}")
    print(f"   Drop (Discriminability):  {avg_orig_evo_small - avg_shuffle_evo_small:.4f}")

    print("-" * 40)

    print("2. Evolution vs Big Diff:")
    print(f"   Original Pairing Avg Sim: {avg_orig_evo_big:.4f}")
    print(f"   Shuffled Pairing Avg Sim: {avg_shuffle_evo_big:.4f}")
    print(f"   Drop (Discriminability):  {avg_orig_evo_big - avg_shuffle_evo_big:.4f}")

    print("3. Small Diff vs Big Diff:")
    print(f"   Original Pairing Avg : {avg_orig_small_big:.4f}")


    print("=" * 40)
    print("Interpretation:")
    print("Higher 'Drop' means the model is better at distinguishing")
    print("correct pairs from random incorrect pairs.")
    print("=" * 40)

# 1. Evolution vs Small Diff:
#    Original Pairing Avg Sim: 0.6247
#    Shuffled Pairing Avg Sim: 0.5209
#    Drop (Discriminability):  0.1038
# ----------------------------------------
# 2. Evolution vs Big Diff:
#    Original Pairing Avg Sim: 0.4996
#    Shuffled Pairing Avg Sim: 0.4848
#    Drop (Discriminability):  0.0148
# 3. Small Diff vs Big Diff:
#    Original Pairing Avg : 0.7084

if __name__ == "__main__":
    main()