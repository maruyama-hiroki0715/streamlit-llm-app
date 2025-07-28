from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
# Streamlitアプリのタイトル
st.title("🤖 LangChain + Streamlit チャットアプリ")

# OpenAI API キーの確認
if not OPENAI_API_KEY:
    st.error("OpenAI API キーが設定されていません。.envファイルにOPENAI_API_KEYを設定してください。")
    st.stop()

# LLMの初期化
@st.cache_resource
def initialize_llm():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )

llm = initialize_llm()

# 専門家の種類を定義
expert_types = {
    "プログラミング専門家": "あなたは経験豊富なプログラミング専門家です。プログラミング言語、フレームワーク、ソフトウェア開発のベストプラクティス、デバッグ、アルゴリズム、データ構造について詳しく、実践的なアドバイスを提供します。コードの例も含めて具体的に説明してください。",
    "ビジネス戦略専門家": "あなたは戦略的思考に長けたビジネス専門家です。市場分析、競合分析、事業計画、マーケティング戦略、組織運営について深い知識を持ち、実践的なビジネスソリューションを提供します。データに基づいた分析的なアプローチで回答してください。",
    "健康・医療専門家": "あなたは健康と医療分野の専門家です。予防医学、栄養学、運動生理学、メンタルヘルス、一般的な健康管理について知識を持ち、科学的根拠に基づいたアドバイスを提供します。ただし、具体的な医療診断や治療については医師への相談を推奨してください。",
    "教育専門家": "あなたは教育と学習の専門家です。効果的な学習方法、教育心理学、カリキュラム設計、学習者のモチベーション向上について深い知識を持ち、個人の学習スタイルに合わせた指導方法を提案します。年齢や学習レベルに応じたアプローチで回答してください。"
}

# 専門家選択のラジオボタン
st.subheader("🎯 専門家を選択してください")
selected_expert = st.radio(
    "どの分野の専門家に相談しますか？",
    list(expert_types.keys()),
    index=0
)

# 選択された専門家を表示
st.info(f"選択中: {selected_expert}")

# 入力フォーム
with st.form("chat_form"):
    user_input = st.text_area(
        "質問を入力してください：",
        placeholder="ここに質問を入力してください...",
        height=100
    )
    submitted = st.form_submit_button("送信")

# フォームが送信された場合の処理
if submitted and user_input:
    with st.spinner("回答を生成中..."):
        try:
            # 選択された専門家に応じてシステムメッセージを設定
            system_message = SystemMessage(content=expert_types[selected_expert])
            user_message = HumanMessage(content=user_input)
            
            # LangChainを使ってLLMに質問を送信
            messages = [system_message, user_message]
            response = llm(messages)
            
            # 結果を表示
            st.success(f"【{selected_expert}】からの回答:")
            st.write(response.content)
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

elif submitted and not user_input:
    st.warning("質問を入力してください。")

# サイドバーに使用方法を表示
with st.sidebar:
    st.header("使用方法")
    st.write("""
    1. 専門家の分野を選択
    2. テキストエリアに質問を入力
    3. 「送信」ボタンをクリック
    4. 選択した専門家としてAIが回答します
    """)
    
    st.header("専門家の詳細")
    st.write(f"**現在選択中:** {selected_expert}")
    
    with st.expander("各専門家について"):
        st.write("**プログラミング専門家**")
        st.write("- プログラミング言語、フレームワーク")
        st.write("- ソフトウェア開発、デバッグ")
        st.write("- アルゴリズム、データ構造")
        
        st.write("**ビジネス戦略専門家**")
        st.write("- 市場分析、競合分析")
        st.write("- 事業計画、マーケティング戦略")
        st.write("- 組織運営、経営戦略")
        
        st.write("**健康・医療専門家**")
        st.write("- 予防医学、栄養学")
        st.write("- 運動生理学、メンタルヘルス")
        st.write("- 一般的な健康管理")
        
        st.write("**教育専門家**")
        st.write("- 効果的な学習方法")
        st.write("- 教育心理学、カリキュラム設計")
        st.write("- モチベーション向上")
    
    st.header("設定情報")
    st.write(f"モデル: gpt-3.5-turbo")
    st.write(f"Temperature: 0.7")
