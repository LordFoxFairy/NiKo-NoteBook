# 第6章 多模态对话Agent

> 使用LangGraph构建智能多模态对话系统

## 6.1 多模态Agent架构

### 6.1.1 系统设计

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class MultimodalAgentState(TypedDict):
    """Agent状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_modality: str  # text/image/audio/video
    analysis_results: dict
    next_action: str

class MultimodalAgent:
    """多模态对话Agent"""
    
    def __init__(
        self,
        vision_analyzer,
        audio_transcriber,
        clip_engine
    ):
        self.vision_analyzer = vision_analyzer
        self.audio_transcriber = audio_transcriber
        self.clip_engine = clip_engine
        
        # 构建workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流"""
        workflow = StateGraph(MultimodalAgentState)
        
        # 添加节点
        workflow.add_node("classify_input", self.classify_input)
        workflow.add_node("process_image", self.process_image)
        workflow.add_node("process_audio", self.process_audio)
        workflow.add_node("process_video", self.process_video)
        workflow.add_node("generate_response", self.generate_response)
        
        # 添加边
        workflow.set_entry_point("classify_input")
        
        workflow.add_conditional_edges(
            "classify_input",
            self.route_by_modality,
            {
                "image": "process_image",
                "audio": "process_audio",
                "video": "process_video",
                "text": "generate_response"
            }
        )
        
        workflow.add_edge("process_image", "generate_response")
        workflow.add_edge("process_audio", "generate_response")
        workflow.add_edge("process_video", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow
    
    def classify_input(self, state: MultimodalAgentState) -> MultimodalAgentState:
        """分类输入模态"""
        last_message = state["messages"][-1]
        
        # 检测输入类型
        if hasattr(last_message, 'image_path'):
            state["current_modality"] = "image"
        elif hasattr(last_message, 'audio_path'):
            state["current_modality"] = "audio"
        elif hasattr(last_message, 'video_path'):
            state["current_modality"] = "video"
        else:
            state["current_modality"] = "text"
        
        return state
    
    def route_by_modality(self, state: MultimodalAgentState) -> str:
        """根据模态路由"""
        return state["current_modality"]
    
    def process_image(self, state: MultimodalAgentState) -> MultimodalAgentState:
        """处理图片"""
        message = state["messages"][-1]
        image_path = message.image_path
        query = message.content
        
        # 使用GPT-4V分析
        result = self.vision_analyzer.analyze_image(image_path, query)
        
        state["analysis_results"] = {"image_analysis": result}
        return state
    
    def process_audio(self, state: MultimodalAgentState) -> MultimodalAgentState:
        """处理音频"""
        message = state["messages"][-1]
        audio_path = message.audio_path
        
        # 转录
        transcript = self.audio_transcriber.transcribe(audio_path)
        
        state["analysis_results"] = {"transcript": transcript['text']}
        return state
    
    def process_video(self, state: MultimodalAgentState) -> MultimodalAgentState:
        """处理视频"""
        message = state["messages"][-1]
        video_path = message.video_path
        
        # 提取关键帧并分析
        from video_processor import VideoProcessor
        processor = VideoProcessor()
        frames = processor.extract_frames(video_path, "temp_frames", fps=0.5)
        
        # 分析首帧
        analysis = self.vision_analyzer.analyze_image(
            frames[0],
            "描述这个视频的主要内容"
        )
        
        state["analysis_results"] = {"video_analysis": analysis}
        return state
    
    def generate_response(self, state: MultimodalAgentState) -> MultimodalAgentState:
        """生成回复"""
        from langchain_core.messages import AIMessage
        
        if "analysis_results" in state:
            # 基于分析结果生成回复
            results = state["analysis_results"]
            response = f"分析结果: {results}"
        else:
            response = "处理完成"
        
        state["messages"].append(AIMessage(content=response))
        return state
    
    def run(self, user_input: dict):
        """
        运行Agent
        
        Args:
            user_input: 用户输入
            
        Returns:
            处理结果
        """
        initial_state = {
            "messages": [user_input],
            "current_modality": "",
            "analysis_results": {},
            "next_action": ""
        }
        
        result = self.app.invoke(initial_state)
        return result

# 使用示例
from langchain_core.messages import HumanMessage

# 初始化组件
vision_analyzer = GPT4VisionAnalyzer(api_key="your-key")
audio_transcriber = WhisperTranscriber(api_key="your-key")
clip_engine = CLIPSearchEngine()

# 创建Agent
agent = MultimodalAgent(vision_analyzer, audio_transcriber, clip_engine)

# 处理图片
class ImageMessage(HumanMessage):
    image_path: str

result = agent.run(
    ImageMessage(content="这张图里有什么?", image_path="test.jpg")
)
print(result["messages"][-1].content)
```

## 6.2 视觉问答系统

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class VisualQASystem:
    """视觉问答系统"""
    
    def __init__(self, vision_analyzer, llm):
        self.vision_analyzer = vision_analyzer
        self.llm = llm
    
    def answer_question(
        self,
        image_path: str,
        question: str
    ) -> str:
        """
        回答图片相关问题
        
        Args:
            image_path: 图片路径
            question: 问题
            
        Returns:
            答案
        """
        # 先用视觉模型理解图片
        image_description = self.vision_analyzer.analyze_image(
            image_path,
            "详细描述这张图片的内容、物体、场景、文字等所有信息"
        )
        
        # 基于描述回答问题
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个视觉问答助手。基于图片描述回答用户问题。"),
            ("user", f"图片描述: {image_description}\n\n问题: {question}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        return response.content

# 使用示例
llm = ChatOpenAI(model="gpt-4", api_key="your-key")
vqa_system = VisualQASystem(vision_analyzer, llm)

answer = vqa_system.answer_question(
    "product.jpg",
    "这个产品的主要特点是什么?"
)
print(answer)
```

## 6.3 多模态知识库

```python
from langchain.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

class MultimodalKnowledgeBase:
    """多模态知识库"""
    
    def __init__(
        self,
        clip_engine: CLIPSearchEngine,
        vision_analyzer,
        embedding_model: OpenAIEmbeddings
    ):
        self.clip_engine = clip_engine
        self.vision_analyzer = vision_analyzer
        self.embedding_model = embedding_model
        
        # 图片向量库
        self.clip_db = CLIPVectorDB(clip_engine)
        
        # 文本向量库
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(":memory:")
        self.text_db = Qdrant(
            client=self.qdrant_client,
            collection_name="text_knowledge",
            embeddings=embedding_model
        )
    
    def add_image(self, image_path: str, metadata: dict = None):
        """添加图片到知识库"""
        # 生成图片描述
        description = self.vision_analyzer.analyze_image(
            image_path,
            "详细描述这张图片"
        )
        
        # 索引到CLIP数据库
        self.clip_db.index_images([image_path])
        
        # 索引描述到文本数据库
        doc = Document(
            page_content=description,
            metadata={"type": "image", "path": image_path, **(metadata or {})}
        )
        self.text_db.add_documents([doc])
    
    def add_text(self, text: str, metadata: dict = None):
        """添加文本到知识库"""
        doc = Document(
            page_content=text,
            metadata={"type": "text", **(metadata or {})}
        )
        self.text_db.add_documents([doc])
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        混合搜索
        
        Args:
            query: 查询
            k: 返回数量
            
        Returns:
            搜索结果
        """
        # 图片搜索
        image_results = self.clip_db.search(query, limit=k)
        
        # 文本搜索
        text_docs = self.text_db.similarity_search(query, k=k)
        text_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.8  # 占位分数
            }
            for doc in text_docs
        ]
        
        # 合并结果
        all_results = []
        for r in image_results:
            all_results.append({
                "type": "image",
                "path": r["path"],
                "score": r["score"]
            })
        for r in text_results:
            all_results.append({
                "type": "text",
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r["score"]
            })
        
        # 按分数排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results[:k]

# 使用示例
kb = MultimodalKnowledgeBase(
    clip_engine,
    vision_analyzer,
    OpenAIEmbeddings()
)

# 添加内容
kb.add_image("product1.jpg", {"category": "electronics"})
kb.add_text("iPhone 15 Pro features titanium design and A17 Pro chip")

# 搜索
results = kb.search("titanium phone", k=3)
for r in results:
    print(f"Type: {r['type']}, Score: {r['score']:.4f}")
```

## 本章小结

- LangGraph实现灵活的多模态工作流编排
- 视觉问答结合视觉模型与语言模型优势
- 多模态知识库支持混合检索

---

**下一章**: [第7章 完整系统实现](./第7章_完整系统实现.md)
