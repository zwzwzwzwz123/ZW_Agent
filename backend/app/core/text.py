import re
from collections import Counter
from math import sqrt


# 这个文件是 RAG 链路的“文本地基”：
# 文档和问题在进入检索器之前，都要先被清洗、切分、转成 token 或向量。
# 你学习时可以重点问自己：文本如何从自然语言变成可计算的形式？

# 匹配两类最基础的文本单元：
# 1. 英文、数字、下划线组成的词，例如 RAG、token、self_attention。
# 2. 连续中文，例如 “可以降低幻觉”。
# 这里先不用复杂分词库，是为了让第一版检索逻辑足够透明。
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", re.UNICODE)
CJK_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")


def normalize_text(text: str) -> str:
    """压缩空白字符，让后续切分更稳定。

    RAG 中很多“看起来奇怪”的检索问题，源头其实是数据清洗没做好：
    换行、多个空格、复制 PDF 带来的不可见字符，都可能影响 chunking。
    """
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """把文本拆成检索器可以使用的 token。

    英文可以靠空格和正则得到比较自然的词；中文没有天然空格，
    如果把一整段中文当成一个 token，query 和 chunk 很难产生交集。

    当前版本对中文做两个层次的特征：
    - 单字：提高召回，哪怕词边界不准也能匹配到一部分。
    - bigram：保留短语信息，例如 “降低”“幻觉”。

    这不是最终方案。后续接入 embedding 或中文分词时，这里就是重点替换点。
    """
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        if CJK_PATTERN.match(token):
            # 中文单字能提高“粗召回”，但只用单字会丢失短语含义。
            tokens.extend(token)
            # 中文 bigram 能表达相邻字组成的短语，是一个轻量但有效的折中。
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def term_vector(text: str) -> Counter[str]:
    """把 token 列表变成词频向量。

    Counter 可以理解成一个稀疏向量：
    {"rag": 1, "降低": 1, "幻觉": 1}

    当前 v0 用词频向量学习检索原理。未来升级到 embedding 后，
    这里会变成“调用模型得到稠密向量”。
    """
    return Counter(tokenize(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """计算两个词频向量的余弦相似度。

    余弦相似度关注“方向是否相似”，而不是文本绝对长度。
    在 RAG 检索里，它用来回答一个问题：
    query 和某个 chunk 是否谈论了相近的内容？

    它的局限也很明显：只看字面 token，不懂同义词和深层语义。
    例如“降低幻觉”和“减少编造”语义接近，但关键词检索未必能匹配。
    """
    if not left or not right:
        return 0.0

    common_terms = left.keys() & right.keys()
    dot_product = sum(left[term] * right[term] for term in common_terms)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def chunk_text(text: str, max_tokens: int = 120, overlap: int = 24) -> list[str]:
    """把长文档切成较小 chunk。

    为什么 RAG 要切 chunk：
    - 检索器通常不能直接把整篇长文档当作一个检索单元。
    - chunk 太大会混入噪声，影响相似度。
    - chunk 太小会丢上下文，回答时证据不完整。

    overlap 是为了减少“边界切断”：
    一个关键信息如果刚好跨越两个 chunk，没有重叠就可能被拆散。
    """
    normalized = normalize_text(text)
    words = TOKEN_PATTERN.findall(normalized)
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        # 下一段从 end - overlap 开始，让相邻 chunk 保留一部分上下文。
        start = max(0, end - overlap)

    return chunks
