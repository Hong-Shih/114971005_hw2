# ------------------------------------ partA_1 ------------------------------------
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

# ------------------------------------ partA_2 ------------------------------------

import re

# ==================================
# 1. 情感分類器
# ==================================
class RuleBasedSentimentClassifier:
    def __init__(self):
        # 建立正負面詞彙庫
        self.positive_words = [
            '好', '棒', '優秀', '喜歡', '推薦', '滿意', '開心', '值得', '精彩', '完美'
        ]
        self.negative_words = [
            '差', '爛', '糟', '失望', '討厭', '不推薦', '浪費', '無聊', '爛', '糟糕', '差勁'
        ]
        # 加入否定詞處理
        self.negation_words = ['不', '沒', '無', '非', '別']

        # 為了實現 "考慮程度副詞的加權" 這一點，我們額外定義一個列表
        # 這些詞在 __init__ 中沒有提供，所以我們在這裡補充
        self.degree_adverbs = ['太', '很', '非常', '超級']

    def classify(self, text):
        """
        分類邏輯:
        1. 計算正負詞彙數量
        2. 處理否定詞 (否定詞+正面詞=負面)
        3. 考慮程度副詞的加權
        4. 返回: 正面/負面/中性
        """

        pos_score = 0
        neg_score = 0

        # 為了避免重複計算，我們使用一個副本
        text_to_check = text

        # 1. 處理否定詞 + 正面詞 (例如: "不好") -> 這應該算負面
        for neg_word in self.negation_words:
            for pos_word in self.positive_words:
                phrase = neg_word + pos_word
                if phrase in text_to_check:
                    neg_score += 1
                    # 從文本中 "移除" 這個詞組，避免被 "好" 再次計為正面
                    text_to_check = text_to_check.replace(phrase, "", 1)

        # 2. 處理正面詞 (包含程度副詞)
        for pos_word in self.positive_words:
            # 檢查是否有程度副詞加權
            weight = 1
            for adv in self.degree_adverbs:
                phrase = adv + pos_word
                if phrase in text_to_check:
                    weight = 2  # 加權
                    text_to_check = text_to_check.replace(phrase, "", 1)
                    break # 找到一個副詞就夠了

            # 計算基礎分數
            # 這裡使用 re.findall 來計算剩餘文本中該詞的出現次數
            count = len(re.findall(pos_word, text_to_check))
            if count > 0:
                pos_score += (weight * count)
                text_to_check = text_to_check.replace(pos_word, "") # 避免重複計算

        # 3. 處理負面詞 (包含程度副詞)
        for neg_word in self.negative_words:
            # 檢查是否有程度副詞加權
            weight = 1
            for adv in self.degree_adverbs:
                phrase = adv + neg_word
                if phrase in text_to_check:
                    weight = 2  # 加權
                    text_to_check = text_to_check.replace(phrase, "", 1)
                    break

            # 計算基礎分數
            count = len(re.findall(neg_word, text_to_check))
            if count > 0:
                neg_score += (weight * count)
                text_to_check = text_to_check.replace(neg_word, "")

        # 4. 返回結果
        final_score = pos_score - neg_score

        if final_score > 0:
            return "正面"
        elif final_score < 0:
            return "負面"
        else:
            return "中性"

# ==================================
# 2. 主題分類器
# ==================================
class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            '科技': ['AI', '人工智慧', '電腦', '軟體', '程式', '演算法'],
            '運動': ['運動', '健身', '跑步', '游泳', '球類', '比賽'],
            '美食': ['吃', '食物', '餐廳', '美味', '料理', '烹飪'],
            '旅遊': ['旅行', '景點', '飯店', '機票', '觀光', '度假']
        }

    def classify(self, text):
        """
        返回最可能的主題
        """
        # 您的實作

        # 建立一個字典來儲存每個主題的分數
        topic_scores = {topic: 0 for topic in self.topic_keywords}

        # 遍歷所有主題及其關鍵字
        for topic, keywords in self.topic_keywords.items():
            # 遍歷該主題下的所有關鍵字
            for keyword in keywords:
                # 如果關鍵字出現在文本中，就為該主題加分
                if keyword in text:
                    topic_scores[topic] += 1

        # 找出分數最高的的主題
        # 我們先過濾掉分數為0的主題
        non_zero_scores = {topic: score for topic, score in topic_scores.items() if score > 0}

        if not non_zero_scores:
            # 如果所有分數都是0，則返回 "未知"
            return "未知"
        else:
            # 返回分數最高的主題名稱
            # max 函數的 key 參數可以讓我們指定比較依據
            best_topic = max(non_zero_scores, key=non_zero_scores.get)
            return best_topic

# ==================================
# 測試資料 \
# ==================================
test_texts = [
    "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
    "最新的AI技術突破讓人驚艷，深度學習模型的表現越來越好",
    "這部電影劇情空洞，演技糟糕，完全是浪費時間",
    "每天慢跑5公里，配合適當的重訓，體能進步很多"
]

# ==================================
# 執行測試
# ==================================
print("--- 開始測試 ---")

# 實例化分類器
sentiment_classifier = RuleBasedSentimentClassifier()
topic_classifier = TopicClassifier()

# 遍歷測試文本並輸出結果
for i, text in enumerate(test_texts):
    sentiment = sentiment_classifier.classify(text)
    topic = topic_classifier.classify(text)

    print(f"\n[文本 {i+1}]: \"{text}\"")
    print(f"  -> 情感: {sentiment}")
    print(f"  -> 主題: {topic}")

print("\n--- 測試結束 ---")

# ------------------------------------ partA_3 ------------------------------------

import re

# ==================================
# A-3: 統計式自動摘要 (來自 截圖 2025-11-16 上午8.00.24.png)
# ==================================
class StatisticalSummarizer:

    def __init__(self):
        # 載入停用詞 (已提供)
        self.stop_words = set(['的', '了', '在', '是', '我', '有', '和',
                               '就', '不', '人', '都', '一', '個', '上',
                               '也', '很', '到', '說', '要', '去', '你'])

    def sentence_score(self, sentence, word_freq):
        """
        計算句子重要性分數
        考量因素 (根據 docstring):
        1. 包含高頻詞的數量
        2  句子位置(首尾句加權)
        3. 句子長度 (太短或太長扣分)
        4. 是否包含數字或專有名詞 (簡化為檢查數字)

        """

        chars = list(sentence)

        valid_chars = [
            char for char in chars
            if char not in self.stop_words and char.strip()
        ]

        if not valid_chars:
            return 0

        # 因素 1: 包含高頻詞(字)的數量 (這裡使用分數加總)
        score = 0
        for char in valid_chars:
            if char in word_freq:
                score += word_freq[char]

        # 因素 3: 句子長度 (太短或太長扣分)
        # 假設理想的 "有效字" 長度在 10 到 50 之間
        valid_length = len(valid_chars)
        if valid_length < 10 or valid_length > 50:
            score *= 0.5  # 長度不理想，分數打折

        # 因素 4: 是否包含數字 (加分)
        if re.search(r'\d', sentence):
            score *= 1.2  # 包含數字，給予 20% 加權

        return score

    def summarize(self, text, ratio=0.3):
        """
        生成摘要步驟:
        1. 分句 (處理中文標點)
        2. 分詞並計算詞頻
        3. 計算每個句子的重要性分數
        4. 選擇最高分的句子
        5. 按原文順序排列
        """

        # 1. 分句 (處理中文標點)
        sentences = re.split(r'[。！？\n]+', text)
        original_sentences = [s.strip() for s in sentences if s.strip()]

        if not original_sentences:
            return "" # 如果沒有有效句子，返回空字串

        # 2. 分 "字" 並計算 "字頻"
        char_freq = {}
        for char in list(text):
            # 必須是有效字 (非停用詞、非標點、非空白)
            if char not in self.stop_words and \
               char.strip() and \
               not re.match(r'[。！？\n]', char):
                char_freq[char] = char_freq.get(char, 0) + 1

        # 3. 計算每個句子的重要性分數
        sentence_scores = {}
        total_sentences = len(original_sentences)

        for i, sentence in enumerate(original_sentences):
            # 獲取基礎分數 (因素 1, 3, 4)
            base_score = self.sentence_score(sentence, char_freq)

            # 處理 因素 2: 句子位置 (首尾句加權)
            # 給第一句和最後一句 1.5 倍的權重
            if i == 0 or i == (total_sentences - 1):
                base_score *= 1.5

            sentence_scores[sentence] = base_score

        # 4. 選擇最高分的句子
        # 決定要保留的句子數量 (至少保留 1 句)
        num_to_keep = max(1, int(total_sentences * ratio))

        # 對句子按分數進行排序
        sorted_scores = sorted(sentence_scores.items(),
                               key=lambda item: item[1],
                               reverse=True)

        # 獲取分數最高的句子 (存入 set 以便快速查找)
        top_sentences = set()
        for sentence, score in sorted_scores[:num_to_keep]:
            top_sentences.add(sentence)

        # 5. 按原文順序排列
        summary_list = []
        for sentence in original_sentences:
            if sentence in top_sentences:
                summary_list.append(sentence)

        # 將摘要句子重新組合
        return "。".join(summary_list) + "。"

# ==================================
# 測試文章
# ==================================
article = """
人工智慧 (AI) 的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。

在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。

教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。

然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。

面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
"""

# ==================================
# 執行測試
# ==================================
print("--- 執行統計式自動摘要 ---")
print(f"原文總字數: {len(article)}字")

# 實例化摘要器
summarizer = StatisticalSummarizer()

# 執行摘要，使用預設的 30% 比例
summary = summarizer.summarize(article, ratio=0.3)

print(f"\n--- 摘要結果 (ratio=0.3) ---")
print(summary)
print(f"\n摘要總字數: {len(summary)}字")
