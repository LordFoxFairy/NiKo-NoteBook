# 第5章 CLIP跨模态检索

> 掌握CLIP模型实现图文互搜

## 5.1 CLIP模型原理

CLIP (Contrastive Language-Image Pre-training) 通过对比学习建立图文语义对齐。

### 5.1.1 基础使用

```python
import clip
import torch
from PIL import Image

class CLIPSearchEngine:
    """CLIP搜索引擎"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
    
    def encode_images(self, image_paths: list[str]) -> torch.Tensor:
        """
        编码图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            图片特征向量
        """
        images = [self.preprocess(Image.open(p)) for p in image_paths]
        image_input = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本特征向量
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def search_images_by_text(
        self,
        query: str,
        image_paths: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        文本搜索图片
        
        Args:
            query: 搜索查询
            image_paths: 候选图片路径
            top_k: 返回数量
            
        Returns:
            (图片路径, 相似度分数)列表
        """
        text_features = self.encode_text([query])
        image_features = self.encode_images(image_paths)
        
        # 计算相似度
        similarity = (text_features @ image_features.T).squeeze(0)
        
        # 排序
        values, indices = similarity.topk(top_k)
        
        results = [
            (image_paths[idx], score.item())
            for idx, score in zip(indices, values)
        ]
        
        return results
    
    def search_images_by_image(
        self,
        query_image_path: str,
        candidate_image_paths: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        图片搜索图片
        
        Args:
            query_image_path: 查询图片
            candidate_image_paths: 候选图片
            top_k: 返回数量
            
        Returns:
            (图片路径, 相似度)列表
        """
        query_features = self.encode_images([query_image_path])
        candidate_features = self.encode_images(candidate_image_paths)
        
        similarity = (query_features @ candidate_features.T).squeeze(0)
        values, indices = similarity.topk(top_k + 1)  # +1排除自己
        
        # 排除查询图片本身
        results = []
        for idx, score in zip(indices, values):
            path = candidate_image_paths[idx]
            if path != query_image_path:
                results.append((path, score.item()))
            if len(results) >= top_k:
                break
        
        return results

# 使用示例
search_engine = CLIPSearchEngine()

# 文本搜图
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = search_engine.search_images_by_text("a dog playing", image_paths)
for path, score in results:
    print(f"{path}: {score:.4f}")

# 以图搜图
similar_images = search_engine.search_images_by_image(
    "query.jpg",
    image_paths
)
for path, score in similar_images:
    print(f"{path}: {score:.4f}")
```

## 5.2 向量数据库集成

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

class CLIPVectorDB:
    """CLIP向量数据库"""
    
    def __init__(self, clip_engine: CLIPSearchEngine, collection_name: str = "images"):
        self.clip_engine = clip_engine
        self.client = QdrantClient(":memory:")  # 内存模式,生产环境用服务器地址
        self.collection_name = collection_name
        
        # 创建集合
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    
    def index_images(self, image_paths: list[str]):
        """
        索引图片
        
        Args:
            image_paths: 图片路径列表
        """
        # 批量编码
        features = self.clip_engine.encode_images(image_paths)
        
        # 上传到向量库
        points = [
            PointStruct(
                id=idx,
                vector=features[idx].cpu().numpy().tolist(),
                payload={"path": path}
            )
            for idx, path in enumerate(image_paths)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"已索引{len(image_paths)}张图片")
    
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        搜索
        
        Args:
            query: 查询文本
            limit: 返回数量
            
        Returns:
            搜索结果
        """
        # 编码查询
        query_vector = self.clip_engine.encode_text([query])[0]
        
        # 搜索
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.cpu().numpy().tolist(),
            limit=limit
        )
        
        return [
            {
                "path": r.payload["path"],
                "score": r.score
            }
            for r in results
        ]

# 使用示例
clip_db = CLIPVectorDB(search_engine)

# 索引图片
clip_db.index_images(["img1.jpg", "img2.jpg", "img3.jpg"])

# 搜索
results = clip_db.search("红色的汽车", limit=5)
for r in results:
    print(f"{r['path']}: {r['score']:.4f}")
```

## 5.3 零样本分类

```python
def zero_shot_classification(
    image_path: str,
    categories: list[str],
    clip_engine: CLIPSearchEngine
) -> dict:
    """
    零样本图像分类
    
    Args:
        image_path: 图片路径
        categories: 类别列表
        clip_engine: CLIP引擎
        
    Returns:
        分类结果
    """
    # 编码图片
    image_features = clip_engine.encode_images([image_path])
    
    # 编码类别(添加提示词模板)
    category_prompts = [f"a photo of a {cat}" for cat in categories]
    text_features = clip_engine.encode_text(category_prompts)
    
    # 计算相似度
    similarity = (image_features @ text_features.T).squeeze(0)
    
    # 转换为概率
    probs = similarity.softmax(dim=0)
    
    # 排序
    values, indices = probs.topk(len(categories))
    
    results = {
        categories[idx]: prob.item()
        for idx, prob in zip(indices, values)
    }
    
    return results

# 使用示例
categories = ["dog", "cat", "bird", "fish", "horse"]
result = zero_shot_classification("animal.jpg", categories, search_engine)

for category, prob in result.items():
    print(f"{category}: {prob*100:.2f}%")
```

## 本章小结

- CLIP实现图文语义对齐,支持零样本学习
- 向量数据库可高效存储和检索图片特征
- 零样本分类无需训练数据即可分类

---

**下一章**: [第6章 多模态对话Agent](./第6章_多模态对话Agent.md)
