from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import datetime

class RAGService:
    def __init__(self, collection_name="meeting_transcripts", path="./qdrant_data"):
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)
        # Using a lightweight model for local embedding
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2') 
        print("Embedding model loaded.")
        
        self._init_collection()

    def _init_collection(self):
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name not in collection_names:
            print(f"Creating collection: {self.collection_name}")
            # all-MiniLM-L6-v2 produces 384-dimensional vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        else:
            print(f"Collection {self.collection_name} already exists.")

    def add_transcript(self, text):
        if not text or not text.strip():
            return
            
        vector = self.encoder.encode(text).tolist()
        point_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={"text": text, "timestamp": timestamp}
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        print(f"Stored transcript: {text[:30]}...")

    def query(self, query_text, limit=3):
        vector = self.encoder.encode(query_text).tolist()
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit
        ).points
        
        results = []
        for hit in search_result:
            results.append(hit.payload["text"])
            
        return results

    def get_all_vectors(self, limit=100):
        """
        Retrieve vectors and payloads for visualization.
        """
        # Scroll through points to get data
        response, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_vectors=True,
            with_payload=True
        )
        return response
