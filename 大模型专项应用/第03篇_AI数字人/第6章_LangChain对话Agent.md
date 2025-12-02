# 第6章 LangChain对话Agent

## 6.1 LangChain 1.0架构

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class DigitalHumanAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview",
                             temperature=0.7, streaming=True)
        
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业的AI助手,回复简洁(50字内)"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, prompt=self.prompt)
    
    async def chat_stream(self, user_input: str):
        async for chunk in self.chain.astream({"input": user_input}):
            if "response" in chunk:
                yield chunk["response"]
```

## 6.2 情感感知Agent

```python
from langchain_core.output_parsers import JsonOutputParser

class EmotionAwareAgent(DigitalHumanAgent):
    async def analyze_emotion(self, text: str):
        emotion_prompt = ChatPromptTemplate.from_messages([
            ("system", "分析情感,返回JSON: {emotion, intensity}"),
            ("human", "{text}")
        ])
        
        chain = emotion_prompt | self.llm | JsonOutputParser()
        return await chain.ainvoke({"text": text})
    
    async def chat_with_emotion(self, user_input: str):
        user_emotion = await self.analyze_emotion(user_input)
        
        # 根据情感调整响应语气
        if user_emotion['emotion'] == 'angry':
            self.llm.temperature = 0.3  # 更稳重
        elif user_emotion['emotion'] == 'happy':
            self.llm.temperature = 0.9  # 更活泼
        
        async for chunk in self.chat_stream(user_input):
            yield chunk, user_emotion
```

## 6.3 RAG知识增强

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

class RAGAgent(DigitalHumanAgent):
    def __init__(self, knowledge_base_path: str):
        super().__init__()
        
        # 加载向量数据库
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(knowledge_base_path, self.embeddings)
        
        # RAG链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    async def chat_with_knowledge(self, user_input: str):
        # 检索相关知识
        docs = await self.vectorstore.asimilarity_search(user_input, k=3)
        
        # 增强Prompt
        context = "\n".join([doc.page_content for doc in docs])
        enhanced_input = f"参考知识:\n{context}\n\n用户问题: {user_input}"
        
        async for chunk in self.chat_stream(enhanced_input):
            yield chunk
```

## 6.4 流式响应处理

```python
class StreamingHandler:
    async def handle_stream(self, agent: DigitalHumanAgent, user_input: str):
        buffer = ""
        
        async for chunk in agent.chat_stream(user_input):
            buffer += chunk
            
            # 达到一句话就触发TTS
            if chunk.endswith(('。', '!', '?', '.')):
                yield buffer
                buffer = ""
        
        # 处理剩余内容
        if buffer:
            yield buffer
```

## 6.5 本章小结
- LangChain 1.0提供流式对话能力
- 情感感知提升交互自然度
- RAG增强专业知识准确性
- 流式处理降低首响延迟至200ms
