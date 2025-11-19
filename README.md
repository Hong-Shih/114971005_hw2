此包含了**專案簡介**、**環境安裝**、**如何執行**（特別針對 Part B 的 API Key 設定），以及**預期輸出**的說明。

-----

# NLP 與 GenAI 功能實作比較專案 (Part A & Part B)

本專案旨在比較 **傳統自然語言處理演算法 (Part A)** 與 **Google Gemini 生成式 AI (Part B)** 在文本分析任務上的實作差異與成效。

## 📂 專案結構

  * **Part A (Traditional NLP):**
      * 使用數學統計方法 (TF-IDF, Cosine Similarity)。
      * 使用規則式 (Rule-based) 進行情感與主題分類。
      * 使用統計式方法進行自動摘要。
  * **Part B (Generative AI):**
      * 串接 **Google Gemini Pro API**。
      * 實作語意相似度判斷、多維度分類 (JSON 輸出)、生成式摘要。

## 🛠️ 環境需求 (Prerequisites)

  * **Python 版本**: 3.9 或以上
  * **Google Cloud API Key**: 執行 Part B 需具備 Gemini API 權限的金鑰。

### 📦 安裝必要套件

請在終端機 (Terminal) 或命令提示字元執行以下指令安裝依賴套件：

```bash
pip install -U scikit-learn google-generativeai
```

-----

## 🚀 執行說明 (Execution Guide)

### 1️⃣ 執行 Part A (傳統 NLP)

Part A 完全在本地端運算，無需網路連線或 API Key。

1.  將 Part A 的程式碼存為 `part_a_nlp.py`。
2.  執行指令：
    ```bash
    python part_a_nlp.py
    ```
3.  **預期輸出**:
      * 手動與 Scikit-learn 計算的 TF-IDF/IDF 數值。
      * 規則式分類器對測試文本的判定結果 (部分可能顯示為「未知」)。
      * 統計式摘要結果 (從原文中擷取的句子)。

-----

### 2️⃣ 執行 Part B (Google Gemini AI)

Part B 需要呼叫 Google 伺服器，請確保網路連線正常，並設定好 API Key。

#### 🔑 設定 API Key (兩種方式)

**方式一：在 Google Colab 上執行 (推薦)**

1.  在左側欄位點選「🔑 (Secrets)」。
2.  新增一個 Secret，名稱設為 `GOOGLE_API_KEY`，值填入你的 API Key。
3.  開啟「Notebook access」權限。
4.  直接執行程式碼即可。

**方式二：在本地端 (Local) 執行**
若你在自己的電腦執行，程式碼會報錯找不到 `google.colab`。請修改程式碼中的 `try...except`區塊，直接指定 Key：

```python
# 在程式碼開頭修改
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
except ImportError:
    # 在這裡填入你的真實 Key
    GOOGLE_API_KEY = "你的_API_KEY_貼在這裡" 
```

#### ▶️ 開始執行

1.  將 Part B 的程式碼存為 `part_b_genai.py`。
2.  執行指令：
    ```bash
    python part_b_genai.py
    ```
3.  **預期輸出**:
      * **B-1**: 兩段文字的語意相似度分數 (0-100)。
      * **B-2**: 針對測試文本回傳的 JSON 格式分析 (包含情感、精確主題、信心指數)。
      * **B-3**: 指定長度 (如 100 字) 的流暢文章摘要。

-----

## 📊 效能與結果比較

執行完畢後，你將觀察到以下差異 (詳見 `performance_metrics.json`)：

| 特性 | Part A (傳統 NLP) | Part B (GenAI) |
| :--- | :--- | :--- |
| **速度** | ⚡️ 極快 (毫秒級) | 🐢 較慢 (需等待 API 回應) |
| **準確度** | ⚠️ 受限於關鍵字庫 (易出現"未知") | 🎯 極高 (具備語意理解能力) |
| **摘要品質** | ✂️ 生硬 (句子拼接) | 📝 通順 (重新組織語言) |
| **成本** | 💻 僅消耗本機 CPU | 💰 需消耗 API Token |

-----
