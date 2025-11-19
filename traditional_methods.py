import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 手動計算 TF-IDF ---

def calculate_tf(word_dict, total_words):
    """
    計算詞頻 (Term Frequency, TF)
    Args:
        word_dict: 詞彙計數字典 (e.g., {'詞A': 2, '詞B': 1})
        total_words: 該文件的總詞數
    Returns:
        tf_dict: TF 值字典
    """
    # 您的實作
    tf_dict = {}
    if total_words == 0:
        return tf_dict

    # TF = (該詞彙在文件中的出現次數) / (文件總詞數)
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words
    return tf_dict

def calculate_idf(documents, word):
    """
    計算逆文件頻率 (Inverse Document Frequency, IDF)
    Args:
        documents: 文件列表 (list of strings)
        word: 目標詞彙
    Returns:
        idf: IDF 值
    """
    # 您的實作
    total_documents = len(documents)
    doc_containing_word_count = 0

    for doc in documents:
        if word in doc:
            doc_containing_word_count += 1

    numerator = total_documents + 1
    denominator = doc_containing_word_count + 1

    idf = math.log(numerator / denominator) + 1

    return idf

# --- 測試資料 ---
documents = [
    "人工智慧正在改變世界, 機器學習是其核心技術",
    "深度學習推動了人工智慧的發展, 特別是在圖像識別領域",
    "今天天氣很好, 適合出去走動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康, 每天都應該保持運動習慣"
]

# --- 2. 使用 scikit-learn 實作 ---

# 您的實作

vectorizer = TfidfVectorizer(analyzer='char')
tfidf_matrix = vectorizer.fit_transform(documents)
cosine_sim_matrix = cosine_similarity(tfidf_matrix)


# --- 執行與顯示結果 ---

print("=== 1. 手動計算 IDF 測試 ===")
word_to_test = "人工智慧"
idf_value = calculate_idf(documents, word_to_test)
print(f"'{word_to_test}' 的 IDF 值為: {idf_value:.4f}")

word_to_test = "運動"
idf_value = calculate_idf(documents, word_to_test)
print(f"'{word_to_test}' 的 IDF 值為: {idf_value:.4f}")

print("\n=== 2. scikit-learn 實作 ===")
print(cosine_sim_matrix)
