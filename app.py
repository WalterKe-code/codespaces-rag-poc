import streamlit as st
import os
import json
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── 持久化路徑設定 ──────────────────────────────────────────
DATA_DIR        = Path("data")
CHROMA_DIR      = Path("chroma_db")
HISTORY_FILE    = Path("chat_history.json")
INDEXED_FILE    = Path("indexed_files.json")
# ────────────────────────────────────────────────────────────

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Codespaces RAG PoC", layout="wide")
st.title("📚 Codespaces RAG 概念驗證系統 (PoC)")

# ── 輔助函式 ─────────────────────────────────────────────────

def load_indexed_files() -> dict:
    if INDEXED_FILE.exists():
        return json.loads(INDEXED_FILE.read_text(encoding="utf-8"))
    return {}

def save_indexed_files(indexed: dict):
    INDEXED_FILE.write_text(json.dumps(indexed, ensure_ascii=False, indent=2), encoding="utf-8")

def load_chat_history() -> list:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return []

def save_chat_history(history: list):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── 側邊欄 ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 設定")
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    if env_api_key:
        gemini_api_key = env_api_key
        st.success("🔑 API Key 已透過環境變數載入", icon="✅")
    else:
        gemini_api_key = st.text_input("輸入 Google Gemini API Key", type="password")
        st.markdown("[免費取得 Gemini API Key](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    st.subheader("📂 上傳新文件")
    uploaded_files = st.file_uploader(
        "上傳後將永久保留在知識庫中",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )

    indexed = load_indexed_files()
    if indexed:
        st.divider()
        st.subheader("📚 已索引的文件")
        for fname in indexed:
            st.markdown(f"✅ `{fname}`")
        if st.button("🗑️ 清除全部知識庫", type="secondary", use_container_width=True):
            import shutil
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            INDEXED_FILE.unlink(missing_ok=True)
            CHROMA_DIR.mkdir(exist_ok=True)
            DATA_DIR.mkdir(exist_ok=True)
            st.cache_resource.clear()
            st.success("已清除！請重新整理頁面。")
            st.rerun()

    st.divider()
    if st.button("🗑️ 清除對話紀錄", use_container_width=True):
        HISTORY_FILE.unlink(missing_ok=True)
        st.session_state.chat_history = []
        st.rerun()

if not gemini_api_key:
    st.info("👈 請在左側輸入 Gemini API Key 以開始體驗。")
    st.stop()

# ── 建立 Embeddings 與向量庫 ─────────────────────────────────

@st.cache_resource
def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)

@st.cache_resource
def get_vectorstore(api_key: str):
    embeddings = get_embeddings(api_key)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

vectorstore = get_vectorstore(gemini_api_key)

# ── 處理新上傳的文件（加入分批與限流重試保護）─────────────────

if uploaded_files:
    indexed = load_indexed_files()
    new_files = [f for f in uploaded_files if f.name not in indexed]

    if new_files:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for file in new_files:
            save_path = DATA_DIR / file.name
            save_path.write_bytes(file.getvalue())

            try:
                st.info(f"📄 正在解析文件 `{file.name}` ...")
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(str(save_path))
                else:
                    loader = TextLoader(str(save_path), encoding="utf-8")

                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                total_chunks = len(splits)

                st.write(f"文件共切分為 **{total_chunks}** 個文字區塊，開始分批向量化...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 每批次 20 塊，避免打滿 100 RPM 限制
                BATCH_SIZE = 20
                for i in range(0, total_chunks, BATCH_SIZE):
                    batch = splits[i:i + BATCH_SIZE]
                    # 為每個 chunk 加上專屬 ID 避免重複
                    batch_ids = [f"{file.name}_{idx}" for idx in range(i, i + len(batch))]

                    # 失敗自動重試機制
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            vectorstore.add_documents(documents=batch, ids=batch_ids)
                            break
                        except Exception as e:
                            err_msg = str(e)
                            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                                if attempt < max_retries - 1:
                                    wait_sec = 15 * (attempt + 1)
                                    status_text.warning(f"⚠️ 達到 API 頻率限制 (429)，自動等待 {wait_sec} 秒後繼續第 {attempt+1} 次重試...")
                                    time.sleep(wait_sec)
                                else:
                                    raise e
                            else:
                                raise e

                    # 批次間間隔 1.5 秒，主動避開 100 RPM
                    time.sleep(1.5)
                    processed_count = min(i + len(batch), total_chunks)
                    progress_bar.progress(processed_count / total_chunks)
                    status_text.text(f"已完成 {processed_count} / {total_chunks} 塊...")

                progress_bar.empty()
                status_text.empty()
                indexed[file.name] = file.size
                save_indexed_files(indexed)
                st.success(f"✅ `{file.name}` 索引完成！")

            except Exception as e:
                st.error(f"處理 `{file.name}` 時發生錯誤：{e}")

        st.rerun()
    else:
        st.info("ℹ️ 上傳的文件皆已在知識庫中，無需重新索引。")

# ── 建立 RAG 鏈 ─────────────────────────────────────────────

indexed = load_indexed_files()
if not indexed:
    st.write("👈 請先在左側上傳至少一份 PDF 或 TXT 文件以建立知識庫。")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=gemini_api_key, temperature=0.2)

prompt = ChatPromptTemplate.from_template(
    """你是一個專業的知識庫助手。請根據以下檢索到的參考內容回答使用者的問題。

【重要規則】
- 請只根據下方「參考內容」中有明確記載的資訊來回答。
- 如果參考內容中「完全沒有」能夠回答該問題的相關資訊，你必須一字不差地回覆以下這句話，不得自行補充或修改：
  「您提出的問題，目前知識庫內尚無適當的資料可以回答您，請聯繫環保局(049-2233753)詢問辦理」
- 不可推測、猜測或編造任何未出現在參考內容中的資訊。

參考內容：
{context}

問題：{question}
"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 對話介面（含歷史紀錄）───────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

st.success(f"✅ 知識庫已就緒，共 {len(indexed)} 份文件。可以開始提問！")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("請輸入你想詢問這份文件的問題...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("檢索知識庫並生成回答中..."):
            source_docs = retriever.invoke(user_query)
            answer = rag_chain.invoke(user_query)
            st.markdown(answer)

            with st.expander("🔍 檢視本次參考的文本片段 (Sources)"):
                for idx, doc in enumerate(source_docs):
                    st.markdown(f"**片段 {idx+1}：**\n{doc.page_content}")
                    st.divider()

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_chat_history(st.session_state.chat_history)