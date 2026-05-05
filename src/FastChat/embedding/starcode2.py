import torch
import numpy as np
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------配置----------------
# 推荐先尝试 3B 版本，显存友好且速度较快
# MODEL_NAME = "bigcode/starcoder2-3b"
MODEL_NAME = "/XYAIFS00/HDD_POOL/nudt_dwfeng/nudt_dwfeng_1/lhx/llm/starcoder2-7b"  # 如果显存 > 16GB，用 7B 效果更好

# 如果是本地路径，请修改此处，例如: '/path/to/local/starcoder2-7b'
LOCAL_MODEL_PATH = None  # 如果为 None，则从 HF Hub 下载；否则填写本地路径

INPUT_FILE = 'diff_results.json'
BATCH_SIZE = 4  # StarCoder2 显存占用大，Batch Size 建议设小一点 (1-8)
MAX_LENGTH = 8192
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# ----------------------------------

def load_model_and_tokenizer():
    model_path = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH else MODEL_NAME
    print(f"Loading {model_path} on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 设置 padding token，StarCoder2 默认可能没有设置，导致报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用半精度节省显存
        device_map="auto",  # 自动分配层到 GPU
        trust_remote_code=True,
        output_hidden_states=True  # 必须开启，以便获取隐藏状态
    )
    model.eval()

    return tokenizer, model


def diff_lines_to_text(diff_lines: list) -> str:
    """
    将 Diff 行列表转换为纯文本字符串。
    """
    if not diff_lines or not isinstance(diff_lines, list):
        return ""

    filtered_lines = []
    for line in diff_lines:
        if isinstance(line, str) and line.startswith("@@") and line.endswith("@@"):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def diff_lines_to_text(diff_lines: list) -> str:
    """
    将 Diff 行列表转换为纯文本字符串。
    """
    if not diff_lines or not isinstance(diff_lines, list):
        return " "  # 返回空格而不是空字符串

    filtered_lines = []
    for line in diff_lines:
        if isinstance(line, str) and line.startswith("@@") and line.endswith("@@"):
            continue
        filtered_lines.append(line)

    res = "\n".join(filtered_lines).strip()
    return res if res else " "  # 如果过滤后为空，返回空格


def get_batch_embeddings(model, tokenizer, texts: list, batch_size: int, max_length: int):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # 预处理：确保 batch 中没有绝对的空字符串，否则 tokenizer 可能返回空张量
        batch_texts = [t if (t and t.strip()) else tokenizer.eos_token for t in batch_texts]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8
        )

        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        # --- 关键检查：如果 input_ids 的长度为 0，跳过或处理 ---
        if input_ids.shape[1] == 0:
            # 如果整组都是空的，返回全零向量
            print(f"Warning: Batch {i // batch_size} is empty after tokenization. Using zero vectors.")
            hidden_dim = model.config.hidden_size
            all_embeddings.append(np.zeros((len(batch_texts), hidden_dim)))
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # StarCoder2 的 outputs 包含 hidden_states (因为设置了 output_hidden_states=True)
            # 最后一层的索引是 -1
            last_hidden_state = outputs.hidden_states[-1]

        # Mean Pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # L2 Norm
        norms = torch.linalg.norm(mean_embeddings, ord=2, dim=1, keepdim=True)
        normalized_embeddings = mean_embeddings / norms

        all_embeddings.append(normalized_embeddings.cpu().numpy())

        del inputs, last_hidden_state
        torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)

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

    # 2. 预处理
    texts_evo = []
    texts_small = []
    texts_big = []

    print("Preprocessing diff texts...")
    for row in results:
        texts_evo.append(diff_lines_to_text(row.get("evolution_diff", [])))
        texts_small.append(diff_lines_to_text(row.get("small_diff", [])))
        texts_big.append(diff_lines_to_text(row.get("big_diff", [])))

    # 3. 加载模型
    tokenizer, model = load_model_and_tokenizer()

    # 4. 生成 Embeddings
    print("Generating embeddings...")
    emb_evo = get_batch_embeddings(model, tokenizer, texts_evo, BATCH_SIZE, MAX_LENGTH)
    emb_small = get_batch_embeddings(model, tokenizer, texts_small, BATCH_SIZE, MAX_LENGTH)
    emb_big = get_batch_embeddings(model, tokenizer, texts_big, BATCH_SIZE, MAX_LENGTH)

    # 5. 计算相似度逻辑

    # 准备打混索引
    random.seed(RANDOM_SEED)
    n_samples = len(results)
    indices = list(range(n_samples))
    random.shuffle(indices)

    emb_small_shuffled = emb_small[indices]
    emb_big_shuffled = emb_big[indices]

    # --- Group 1: Evolution vs Small ---
    sim_evo_small_orig = np.sum(emb_evo * emb_small, axis=1)
    sim_evo_small_shuffle = np.sum(emb_evo * emb_small_shuffled, axis=1)

    avg_evo_small_orig = np.mean(sim_evo_small_orig)
    avg_evo_small_shuffle = np.mean(sim_evo_small_shuffle)
    drop_evo_small = avg_evo_small_orig - avg_evo_small_shuffle

    # --- Group 2: Evolution vs Big ---
    sim_evo_big_orig = np.sum(emb_evo * emb_big, axis=1)
    sim_evo_big_shuffle = np.sum(emb_evo * emb_big_shuffled, axis=1)

    avg_evo_big_orig = np.mean(sim_evo_big_orig)
    avg_evo_big_shuffle = np.mean(sim_evo_big_shuffle)
    drop_evo_big = avg_evo_big_orig - avg_evo_big_shuffle

    # --- Group 3: Small vs Big (Original Only) ---
    sim_small_big_orig = np.sum(emb_small * emb_big, axis=1)
    avg_small_big_orig = np.mean(sim_small_big_orig)

    # 6. 输出结果
    print("\n" + "#" * 40)
    print(f"# StarCoder2 Similarity Results")
    print(f"# Model: {MODEL_NAME}")
    print("#" * 40)

    print(f"# 1. Evolution vs Small Diff:")
    print(f"#    Original Pairing Avg Sim: {avg_evo_small_orig:.4f}")
    print(f"#    Shuffled Pairing Avg Sim: {avg_evo_small_shuffle:.4f}")
    print(f"#    Drop (Discriminability):  {drop_evo_small:.4f}")

    print("# " + "-" * 40)

    print(f"# 2. Evolution vs Big Diff:")
    print(f"#    Original Pairing Avg Sim: {avg_evo_big_orig:.4f}")
    print(f"#    Shuffled Pairing Avg Sim: {avg_evo_big_shuffle:.4f}")
    print(f"#    Drop (Discriminability):  {drop_evo_big:.4f}")

    print("# " + "-" * 40)

    print(f"# 3. Small Diff vs Big Diff:")
    print(f"#    Original Pairing Avg : {avg_small_big_orig:.4f}")

    print("#" * 40)

# 1. Evolution vs Small Diff:
#    Original Pairing Avg Sim: 0.5289
#    Shuffled Pairing Avg Sim: 0.4042
#    Drop (Discriminability):  0.1246
# ----------------------------------------
# 2. Evolution vs Big Diff:
#    Original Pairing Avg Sim: 0.1229
#    Shuffled Pairing Avg Sim: 0.0902
#    Drop (Discriminability):  0.0327
# ----------------------------------------
# 3. Small Diff vs Big Diff:
#    Original Pairing Avg : 0.4493

if __name__ == "__main__":
    main()

# evol vs real
# real_version_gap
# real shuffle
