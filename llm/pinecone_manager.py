"""
Pinecone Manager for LLM Module
Optimized for Render deployment - lazy loading of embeddings
Handles vector storage and retrieval with MongoDB for metadata
"""

import os
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from .helper import chunk_pdf_bytes, get_embeddings
from config import Config
from logging_config import logger
from mongo_storage import mongo_storage

PINECONE_INDEXES = Config.PINECONE_INDEXES


class PineconeManager:
    """Manage Pinecone vector stores with MongoDB backup"""

    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)
        # Don't initialize embeddings here - lazy load when needed
        self._embeddings = None
        self.vector_stores = {}
        self.data_expiry = timedelta(hours=24)

        self._setup_indexes()

    @property
    def embeddings(self):
        """
        Lazy-load embeddings property
        Loads ONLY when first accessed
        """
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _setup_indexes(self):
        """Setup all required Pinecone indexes"""
        for index_type, index_name in PINECONE_INDEXES.items():
            try:
                existing_indexes = [i.name for i in self.pc.list_indexes()]

                if index_name not in existing_indexes:
                    logger.info(f"Creating index: {index_name}")
                    self.pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    time.sleep(10)

                # Don't initialize vector store with embeddings yet
                # Will be lazy-loaded when first used
                self.vector_stores[index_type] = None
                logger.info(f"Index {index_name} configured for {index_type}")

            except Exception as e:
                logger.error(f"Failed to setup index {index_name}: {e}")

    def _get_vector_store(self, index_type: str):
        """
        Lazy-load vector store for specific index type
        """
        if index_type not in PINECONE_INDEXES:
            raise ValueError(f"Unknown index type: {index_type}")

        if self.vector_stores.get(index_type) is None:
            index_name = PINECONE_INDEXES[index_type]
            logger.info(f"Initializing vector store for {index_type}...")

            self.vector_stores[index_type] = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings  # This triggers lazy load
            )

            logger.info(f"Vector store ready for {index_type}")

        return self.vector_stores[index_type]

    def _format_timestamp(self, dt: datetime) -> str:
        """Convert datetime to string for Pinecone compatibility"""
        return dt.isoformat()

    def add_weather_data(self, weather_data: Dict, location: str):
        """Add weather data to Pinecone with expiry"""
        try:
            document = Document(
                page_content=f"""
                Weather Update for {location}:
                Temperature: {weather_data.get('temperature_avg', 'N/A')}°C
                Humidity: {weather_data.get('humidity_avg', 'N/A')}%
                Precipitation: {weather_data.get('precipitation', 'N/A')}mm
                Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
                Location: {location}
                Timestamp: {weather_data.get('timestamp', 'N/A')}
                """,
                metadata={
                    "type": "weather",
                    "location": location,
                    "timestamp": self._format_timestamp(datetime.now()),
                    "expiry": self._format_timestamp(datetime.now() + self.data_expiry),
                    "source": "open-meteo"
                }
            )

            vector_store = self._get_vector_store("weather")
            vector_store.add_documents([document])
            logger.info(f"Weather data added for {location}")

        except Exception as e:
            logger.error(f"Failed to add weather data: {e}")
            raise

    def add_news_data(self, news_items: List[Dict]):
        """Add news data to Pinecone with expiry"""
        try:
            documents = []
            for news in news_items:
                document = Document(
                    page_content=f"""
                    News: {news.get('title', '')}
                    Source: {news.get('source', '')}
                    Summary: {news.get('summary', news.get('title', ''))}
                    Published: {news.get('published_at', '')}
                    """,
                    metadata={
                        "type": "news",
                        "source": news.get('source', ''),
                        "timestamp": self._format_timestamp(datetime.now()),
                        "expiry": self._format_timestamp(datetime.now() + self.data_expiry),
                        "url": news.get('url', '')
                    }
                )
                documents.append(document)

            vector_store = self._get_vector_store("news")
            vector_store.add_documents(documents)
            logger.info(f"Added {len(news_items)} news items")

        except Exception as e:
            logger.error(f"Failed to add news data: {e}")
            raise

    def add_bulletins_data(self, bulletins: List[Dict]):
        """Add bulletin data to Pinecone"""
        try:
            documents = []
            for bulletin in bulletins:
                document = Document(
                    page_content=f"""
                    Agricultural Bulletin for {bulletin.get('state', '')}:
                    {bulletin.get('content', '')}
                    Source: {bulletin.get('source', 'IMD')}
                    Date: {bulletin.get('date', '')}
                    """,
                    metadata={
                        "type": "bulletin",
                        "state": bulletin.get('state', ''),
                        "timestamp": self._format_timestamp(datetime.now()),
                        "expiry": self._format_timestamp(datetime.now() + self.data_expiry),
                        "source": "IMD"
                    }
                )
                documents.append(document)

            vector_store = self._get_vector_store("bulletins")
            vector_store.add_documents(documents)
            logger.info(f"Added {len(bulletins)} bulletins")

        except Exception as e:
            logger.error(f"Failed to add bulletins: {e}")
            raise

    def add_disease_data(self, diseases: List[Dict]):
        """Add disease information to Pinecone"""
        try:
            documents = []
            for disease in diseases:
                document = Document(
                    page_content=f"""
                    Crop Disease: {disease.get('disease', '')}
                    Affected Crop: {disease.get('crop', '')}
                    Symptoms: {disease.get('symptoms', '')}
                    Treatment: {disease.get('treatment', '')}
                    Prevention: {disease.get('prevention', '')}
                    Source: {disease.get('source', 'Agricultural Database')}
                    """,
                    metadata={
                        "type": "disease",
                        "crop": disease.get('crop', ''),
                        "disease": disease.get('disease', ''),
                        "timestamp": self._format_timestamp(datetime.now()),
                        "expiry": self._format_timestamp(datetime.now() + timedelta(days=7)),
                        "source": disease.get('source', 'Agricultural Database')
                    }
                )
                documents.append(document)

            vector_store = self._get_vector_store("diseases")
            vector_store.add_documents(documents)
            logger.info(f"Added {len(diseases)} disease entries")

        except Exception as e:
            logger.error(f"Failed to add disease data: {e}")
            raise

    def process_and_index_pdf_bytes(self, file_bytes: bytes, filename: str, index_type: str) -> bool:
        """Process PDF from bytes and add to Pinecone index (no local storage)"""
        try:
            # Chunk PDF from bytes
            chunks = chunk_pdf_bytes(file_bytes, filename)

            if not chunks:
                logger.error("No content extracted from PDF")
                return False

            logger.info(f"Created {len(chunks)} chunks from PDF {filename}")

            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": filename,
                    "upload_type": "admin_upload",
                    "upload_timestamp": datetime.now().isoformat(),
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "type": index_type
                })

            # Get vector store (lazy loads if needed)
            vector_store = self._get_vector_store(index_type)

            # Add to Pinecone
            vector_store.add_documents(chunks)

            # Store document info in MongoDB
            mongo_storage.save_uploaded_file_metadata(
                filename=filename,
                index_type=index_type,
                file_size=len(file_bytes),
                file_path=f"pinecone://{index_type}/{filename}"
            )

            logger.info(f"PDF {filename} indexed to {index_type}")
            return True

        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return False

    def get_uploaded_files(self) -> List[Dict]:
        """Get list of uploaded files from MongoDB"""
        return mongo_storage.get_uploaded_files()

    def delete_uploaded_file(self, filename: str, index_type: str) -> bool:
        """Delete uploaded file metadata from MongoDB"""
        try:
            # Note: Actual vectors in Pinecone would need deletion by ID
            # For now, just remove metadata
            deleted = mongo_storage.delete_uploaded_file_metadata(filename, index_type)
            return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False

    def get_retriever(self, index_type: str, search_kwargs: Dict = None):
        """Get retriever for specific index type"""
        vector_store = self._get_vector_store(index_type)

        search_kwargs = search_kwargs or {"k": 3}
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        