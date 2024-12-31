import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # 如果需要，可以替换为其他嵌入模型
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from typing import Optional, List


# 自定义 QwenLLM 类
class QwenLLM(LLM):
    model_name = "qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_name: str = model_name, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True  # 根据需要启用
        )
        self.device = device

    @property
    def _llm_type(self):
        return "qwen"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                do_sample=True
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if stop:
            for stop_seq in stop:
                if stop_seq in text:
                    text = text.split(stop_seq)[0]
        return text


# 读取数据
data_path = r'E:\大模型\llm\PUS.csv'  # 替换为实际路径
df = pd.read_csv(data_path)
documents = df['content'].tolist()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
split_docs = []
for doc in documents:
    split_docs.extend(text_splitter.split_text(doc))
print(f"总共分割成 {len(split_docs)} 块文档")

# 生成向量并构建 FAISS 向量数据库
embeddings = OpenAIEmbeddings()  # 可根据需要替换为其他嵌入模型
vector_store = FAISS.from_texts(split_docs, embeddings)
vector_store.save_local("faiss_presidents_vector_store")

# 创建检索器
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 定义提示模板
template = """
你是一个提供美国总统信息的助手。

请根据以下上下文回答用户的问题。

上下文：
{context}

问题：{question}

回答：
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 初始化自定义语言模型
llm = QwenLLM()

# 创建 RAG 方案
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
# 测试函数
def ask_question(question):
    result = qa({"query": question})
    answer = result['result']
    sources = result.get('source_documents', [])

    print(f"问题: {question}\n")
    print(f"回答: {answer}\n")
    print("来源:")
    for doc in sources:
        print(f"- {doc.metadata.get('source', 'Unknown')}")
    print("\n" + "=" * 50 + "\n")


# 测试问题
test_questions = [
     "谁是南北战争时期的美国总统？",
    "哪位总统签署了《平价医疗法案》？",
    "请告诉我亚伯拉罕·林肯的成就。"
]

for q in test_questions:
    ask_question(q)
