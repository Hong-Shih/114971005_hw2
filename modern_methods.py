# ------------------------------------ partB_1 ------------------------------------

import google.generativeai as genai
import re
import os

# -----------------------------------------------------------------
# 題目 B-1 的函式實作
# -----------------------------------------------------------------
def ai_similarity(text1: str, text2: str, api_key: str) -> int:
    """
    使用 Google Gemini Pro (gemini-pro) 判斷語意相似度。
    """

    try:
        genai.configure(api_key=api_key)

        prompt = f"""請評估以下兩段文字的語意相似度。
考量因素：
1. 主題相關性
2. 語意重疊程度
3. 表達的觀點是否一致

文字1：{text1}
文字2：{text2}

請只回答一個0-100的整數數字，代表相似度百分比。不要包含任何其他文字、解釋或符號。
"""

        model = genai.GenerativeModel('gemini-pro-latest')
        generation_config = genai.types.GenerationConfig(
            temperature=0.0
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        text_response = response.text.strip()
        match = re.search(r'\d+', text_response)

        if match:
            score = int(match.group(0))
            return max(0, min(100, score))
        else:
            print(f"無法從模型回應中解析出分數: {text_response}")
            return 0

    except Exception as e:
        print(f"API 呼叫或處理時發生錯誤: {e}")
        return 0

# -----------------------------------------------------------------
# 如何使用
# -----------------------------------------------------------------
try:
    # 假設這在 Google Colab 中執行
    from google.colab import userdata

    # 1. 載入金鑰 (您的程式碼)
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

    # 2. 設定全域金鑰 (您的程式碼)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("✅ Google API Key 已成功載入並全域設定。")

    # 3. 執行範例測試 (已更新)
    print("\n--- 測試開始  ---")

    # 範例 1：高度相似 (都關於 AI, ML, DL 的關係)
    text_1 = "人工智慧正在改變世界，機器學習是其核心技術"
    text_2 = "機器學習和深度學習都是人工智慧的重要分支"

    score1 = ai_similarity(text_1, text_2, GOOGLE_API_KEY)
    print(f"'{text_1}'\nvs\n'{text_2}'\n相似度分數: {score1}\n")


    # 範例 2：完全不相關 (AI vs 天氣)
    text_3 = "深度學習推動了人工智慧的發展，特別是在圖像識別領域"
    text_4 = "今天天氣很好，適合出去運動"
    score2 = ai_similarity(text_3, text_4, GOOGLE_API_KEY)
    print(f"'{text_3}'\nvs\n'{text_4}'\n相似度分數: {score2}\n")

except ImportError:
    print("無法 import 'google.colab.userdata'。")
    print("請確認您在 Google Colab 環境中執行，")
    print("或將 GOOGLE_API_KEY = '...' 手動貼上來進行測試。")
except Exception as e:
    print(f"❌ 金鑰載入失敗。")
    print(f"請確認您已在 Colab 的 'Secrets' 標籤中 (左側鑰匙圖示)")
    print(f"設定了 'GOOGLE_API_KEY'。 錯誤: {e}")v

# ------------------------------------ partB_2 ------------------------------------

import google.generativeai as genai
import json  # 用於解析模型回傳的 JSON 字串
import re
import os

# -----------------------------------------------------------------
# 題目 B-2 的函式實作 (ai_classify)
# -----------------------------------------------------------------
def ai_classify(text: str, api_key: str) -> dict:
    """
    使用 Google Gemini Pro (gemini-pro) 進行多維度分類 (B-2)。

    返回一個字典 (dict)，包含:
    {
      "sentiment": "正面/負面/中性",
      "topic": "主題類別",
      "confidence": 0.0-1.0
    }
    """

    # 定義一個預設的錯誤回傳值
    error_response = {"sentiment": "錯誤", "topic": "錯誤", "confidence": 0.0}

    try:
        # 1. 配置 API Key
        genai.configure(api_key=api_key)

        # 2. 設計 Prompt
        prompt = f"""
        請對以下文字進行多維度分析：

        文字：「{text}」

        請判斷：
        1. 情感 (sentiment): 必須是 "正面"、"負面" 或 "中性" 之一。
        2. 主題 (topic): 為這段文字訂一個最貼切的主題類別 (例如: 美食, 科技, 娛樂, 運動, 健康)。
        3. 信心指數 (confidence): 提供一個 0.0 到 1.0 之間的小數，代表你對 sentiment 和 topic 判斷的整體信心。

        請嚴格依照 JSON 格式回傳。
        """

        # 3. 初始化模型並設定回傳格式
        model = genai.GenerativeModel('gemini-pro-latest')

        # 關鍵：要求模型回傳 JSON 格式，並設定低溫以確保一致性
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )

        # 4. 發送請求
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # 5. 解析 JSON 回應
        # 由於設定了 response_mime_type，response.text 應為標準 JSON 字串
        result_dict = json.loads(response.text)

        # 確保必要的鍵存在
        if "sentiment" in result_dict and "topic" in result_dict and "confidence" in result_dict:
            return result_dict
        else:
            print(f"回傳的 JSON 缺少必要的鍵: {result_dict}")
            return error_response

    except json.JSONDecodeError as je:
        # 處理模型回傳的字串不是有效 JSON 的情況
        print(f"JSON 解析失敗: {je}")
        print(f"收到的原始回應: {response.text}")
        return error_response
    except Exception as e:
        # 處理 API Key 錯誤、網路問題等
        print(f"API 呼叫或處理時發生錯誤: {e}")
        return error_response

# -----------------------------------------------------------------
# 如何使用
# -----------------------------------------------------------------
try:
    # 假設這在 Google Colab 中執行
    from google.colab import userdata

    # 1. 載入金鑰 (您的程式碼)
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

    # 2. 設定全域金鑰 (您的程式碼)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("✅ Google API Key 已成功載入並全域設定。")

    # 3. 題目 B-2 附帶的測試資料
    test_texts = [
        "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下一次一定再來！",
        "最新的AI技術突破讓人驚豔，深度學習模型的表現越來越好",
        "這部電影劇情空洞，演技糟糕，完全是浪費時間",
        "每天慢跑5公里，配合適當的重訓，體能進步很多"
    ]

    print("\n--- 測試 B-2 (AI 文本分類) 開始 ---")

    for i, text in enumerate(test_texts):
        print(f"\n--- 測試 {i+1} ---")
        print(f"輸入: {text}")

        # 呼叫 B-2 函式
        result = ai_classify(text, GOOGLE_API_KEY)

        print(f"輸出: {result}")

except ImportError:
    print("無法 import 'google.colab.userdata'。")
    print("請確認您在 Google Colab 環境中執行，")
    print("或將 GOOGLE_API_KEY = '...' 手動貼上來進行測試。")
except Exception as e:
    print(f"❌ 金鑰載入失敗。")
    print(f"請確認您已在 Colab 的 'Secrets' 標籤中 (左側鑰匙圖示)")
    print(f"設定了 'GOOGLE_API_KEY'。 錯誤: {e}")

# ------------------------------------ partB_3 ------------------------------------

import google.generativeai as genai
import json  # (保留 B-2 的 import)
import re    # (保留 B-1 的 import)
import os

# -----------------------------------------------------------------
# 題目 B-3 的函式實作 (ai_summarize)
# -----------------------------------------------------------------
def ai_summarize(text: str, max_length: int, api_key: str) -> str:
    """
    使用 Google Gemini Pro (gemini-pro) 生成摘要 (B-3)。

    要求:
    1. 可控制摘要長度 (max_length)
    2. 保留關鍵資訊
    3. 語句通順
    """

    try:
        # 1. 配置 API Key
        genai.configure(api_key=api_key)

        # 2. 設計 Prompt (整合所有要求)
        prompt = f"""
        請為以下文章生成一段高品質的摘要。

        要求：
        1. 摘要必須精確總結文章的核心觀點和關鍵資訊。
        2. 摘要必須語句通順、精煉、易於理解。
        3. 請將摘要的長度嚴格控制在 {max_length} 字左右。

        文章：
        「{text}」

        請開始生成摘要：
        """

        # 3. 初始化模型並發送請求
        model = genai.GenerativeModel('gemini-pro-latest')

        # 使用低溫 (temperature=0.0) 確保摘要的穩定性和客觀性
        generation_config = genai.types.GenerationConfig(
            temperature=0.0
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # 4. 回傳結果
        return response.text.strip()

    except Exception as e:
        # 處理 API 錯誤
        print(f"API 呼叫或處理時發生錯誤: {e}")
        return "摘要生成失敗。"

# -----------------------------------------------------------------
# 如何使用
# -----------------------------------------------------------------

# 題目 B-3 附帶的測試文章
article = """
人工智慧（AI）的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，
到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。

在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析
大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好
的治療方案。

教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化
的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。

然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會
被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人
隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產
生偏見或歧視。

面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。
只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
"""

try:
    # 假設這在 Google Colab 中執行
    from google.colab import userdata

    # 1. 載入金鑰 (您的程式碼)
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

    # 2. 設定全域金鑰 (您的程式碼)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("✅ Google API Key 已成功載入並全域設定。")

    # 3. 執行 B-3 範例測試
    print("\n--- 測試 B-3 (AI 自動摘要) 開始 ---")

    # 設定目標摘要長度 (例如：100 字)
    target_length = 100

    print(f"原文長度: {len(article)} 字")
    print(f"目標摘要長度: {target_length} 字左右")
    print("--- 生成中... ---")

    # 呼叫 B-3 函式
    summary = ai_summarize(article, target_length, GOOGLE_API_KEY)

    print("\n--- 摘要結果 ---")
    print(summary)
    print(f"\n實際摘要長度: {len(summary)} 字")

    # (可選) 測試不同的長度
    target_length_2 = 50
    print(f"\n--- 測試 (目標 {target_length_2} 字) ---")
    summary_2 = ai_summarize(article, target_length_2, GOOGLE_API_KEY)
    print(summary_2)
    print(f"實際摘要長度: {len(summary_2)} 字")


except ImportError:
    print("無法 import 'google.colab.userdata'。")
    print("請確認您在 Google Colab 環境中執行，")
    print("或將 GOOGLE_API_KEY = '...' 手動貼上來進行測試。")
except Exception as e:
    print(f"❌ 金鑰載入失敗。")
    print(f"請確認您已在 Colab 的 'Secrets' 標籤中 (左側鑰匙圖示)")
    print(f"設定了 'GOOGLE_API_KEY'。 錯誤: {e}")
