
import torch
import numpy as np
import json
import random
# 修改导入：使用具体的 RobertaTokenizer 和 T5ForConditionalGeneration
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# ----------------配置----------------
# 如果是本地路径，请修改这里，例如: '/path/to/local/codet5p-220m'
MODEL_PATH = '/XYAIFS00/HDD_POOL/nudt_dwfeng/nudt_dwfeng_1/lhx/llm/codet5'
# MODEL_PATH = '/your/local/path/to/codet5p-220m'

INPUT_FILE = 'diff_results.json'
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42


# ----------------------------------

def load_model_and_tokenizer():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")

    # 1. 加载 Tokenizer (使用 RobertaTokenizer 以匹配 CodeT5 的分词逻辑)
    # trust_remote_code=True 防止某些版本报错
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 2. 加载 Model (使用 T5ForConditionalGeneration)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


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


def get_batch_embeddings(model, tokenizer, texts: list, batch_size: int, max_length: int):
    """
    批量生成 CodeT5+ 嵌入向量 (Mean Pooling + L2 Norm)
    注意：对于 T5ForConditionalGeneration，我们需要访问 model.encoder
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # 1. Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )

        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)

        # 2. Forward Pass through Encoder Only
        with torch.no_grad():
            # T5ForConditionalGeneration 有一个 .encoder 属性
            # 我们直接调用 encoder 来获取隐藏状态，避免 decoder 的计算开销
            encoder_outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # last_hidden_state: [batch, seq_len, hidden_dim]
            last_hidden_state = encoder_outputs.last_hidden_state

        # 3. Mean Pooling (忽略 Padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask

        # 4. L2 Normalization
        norms = torch.linalg.norm(mean_embeddings, ord=2, dim=1, keepdim=True)
        normalized_embeddings = mean_embeddings / norms

        all_embeddings.append(normalized_embeddings.cpu().numpy())

        # 清理显存
        del inputs, last_hidden_state, mean_embeddings
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
    print(f"# CodeT5 Similarity Results")
    print(f"# Model: {MODEL_PATH}")
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


if __name__ == "__main__":
    main()