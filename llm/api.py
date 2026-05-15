"""
AgriBot LLM API - Chatbot with RAG using Pinecone and MongoDB storage
All data stored in MongoDB, no local file dependencies
"""

from flask import Blueprint, request, jsonify, session
import os
import logging
import json
import io
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.chat_models import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

# from config import SCRAPING_INTERVALS
from logging_config import logger
from .pinecone_manager import PineconeManager
from .admin_manager import AdminManager
from mongo_storage import mongo_storage
from logging_config import logger as root_logger

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY_GPT = os.getenv("API_KEY_GPT")

# Initialize components
Agribot_bp1 = Blueprint("Agribot", __name__)

# Use MongoDB for storing questionnaire data instead of local file
def load_questionnaires_from_mongodb():
    """Load questionnaire data from MongoDB"""
    try:
        doc = mongo_storage.db["questionnaire_data"].find_one({"_id": "master"})
        if doc and "data" in doc:
            return doc["data"]
        return {}
    except Exception as e:
        logger.error(f"Error loading questionnaires from MongoDB: {e}")
        return {}

def save_questionnaires_to_mongodb(data: dict):
    """Save questionnaire data to MongoDB"""
    try:
        mongo_storage.db["questionnaire_data"].update_one(
            {"_id": "master"},
            {"$set": {"data": data, "updated_at": datetime.utcnow()}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error saving questionnaires to MongoDB: {e}")

# Initialize managers
pinecone_manager = PineconeManager(PINECONE_API_KEY)
admin_manager = AdminManager(pinecone_manager)
questionnaire_data = load_questionnaires_from_mongodb()

# Enhanced Session Context Management (using MongoDB for persistence)
class SessionContextManager:
    def __init__(self):
        self.session_collection = mongo_storage.db["chat_sessions"]
        self._init_collections()

    def _init_collections(self):
        self.session_collection.create_index([("session_id", 1), ("timestamp", -1)])

    def get_session_id(self):
        """Get or create session ID"""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session['created_at'] = datetime.now().isoformat()
        return session['session_id']

    def add_to_history(self, query: str, response: str, context_types: List[str],
                       top_contexts: List[Tuple[str, float]]):
        """Add query-response pair to session history in MongoDB"""
        session_id = self.get_session_id()

        # Update context scores
        self.session_collection.update_one(
            {"session_id": session_id, "type": "context_scores"},
            {"$inc": {f"scores.{ctx}": score for ctx, score in top_contexts}},
            upsert=True
        )

        # Add message
        self.session_collection.insert_one({
            "session_id": session_id,
            "type": "message",
            "query": query,
            "response": response,
            "context_types": context_types,
            "top_contexts": top_contexts,
            "timestamp": datetime.utcnow()
        })

    def get_conversation_context(self, current_contexts: List[str] = None) -> str:
        """Get enhanced conversation context for current session"""
        session_id = self.get_session_id()

        # Get recent messages
        recent_messages = list(self.session_collection.find(
            {"session_id": session_id, "type": "message"}
        ).sort("timestamp", -1).limit(5))

        if not recent_messages:
            return ""

        context_lines = ["**Conversation History:**"]
        recent_messages.reverse()

        for i, msg in enumerate(recent_messages):
            context_lines.append(f"{i+1}. **Q:** {msg['query']}")
            context_lines.append(f"   **A:** {msg['response'][:150]}...")
            if msg.get('context_types'):
                context_lines.append(f"   **Contexts used:** {', '.join(msg['context_types'])}")
            context_lines.append("")

        # Get context preferences
        scores_doc = self.session_collection.find_one(
            {"session_id": session_id, "type": "context_scores"}
        )

        if scores_doc and scores_doc.get("scores"):
            top_contexts = sorted(
                scores_doc["scores"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            if top_contexts:
                context_lines.append("**Session Context Preferences:**")
                for ctx, score in top_contexts:
                    context_lines.append(f"- {ctx}: {score:.2f}")
                context_lines.append("")

        return "\n".join(context_lines)

    def get_context_preferences(self) -> List[str]:
        """Get preferred contexts for current session based on history"""
        session_id = self.get_session_id()
        scores_doc = self.session_collection.find_one(
            {"session_id": session_id, "type": "context_scores"}
        )

        if not scores_doc or not scores_doc.get("scores"):
            return []

        sorted_contexts = sorted(
            scores_doc["scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [ctx for ctx, score in sorted_contexts[:2]]

    def clear_session(self, session_id: str = None):
        """Clear session history"""
        if not session_id:
            session_id = self.get_session_id()
        self.session_collection.delete_many({"session_id": session_id})
        session.clear()


session_manager = SessionContextManager()

# Advanced Context Classifier (same as before, but loads from MongoDB)
class AdvancedContextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.8,
            sublinear_tf=True
        )

        self.context_keywords = {
            'weather': [
                'weather', 'temperature', 'rain', 'humidity', 'forecast', 'climate',
                'monsoon', 'drought', 'flood', 'storm', 'precipitation', 'rainfall',
                'wind', 'sunny', 'cloudy', 'hot', 'cold', 'season', 'weather forecast',
                'monsoon prediction', 'rain prediction', 'climate suitable', 'heat wave'
            ],
            'news': [
                'news', 'update', 'latest', 'recent', 'headline', 'breaking',
                'announcement', 'current', 'today', 'happening', 'development',
                'agricultural news', 'farming update', 'market news', 'government announcement'
            ],
            'diseases': [
                'disease', 'pest', 'infection', 'symptom', 'blight', 'rust',
                'fungus', 'treatment', 'cure', 'pesticide', 'fungicide',
                'prevention', 'control', 'crop disease', 'plant infection',
                'leaf spot', 'root rot', 'powdery mildew', 'bacterial blight'
            ],
            'bulletins': [
                'bulletin', 'advisory', 'imd', 'report', 'alert', 'warning',
                'guideline', 'recommendation', 'official', 'government',
                'agricultural advisory', 'farming recommendation', 'crop advisory'
            ],
            'general': [
                'farming', 'agriculture', 'crop', 'cultivation', 'harvest', 'yield',
                'soil', 'fertilizer', 'irrigation', 'seeds', 'planting', 'sowing',
                'how to', 'what is', 'why', 'when', 'where', 'explain'
            ]
        }

        self.context_weights = {
            'diseases': 1.2,
            'weather': 1.1,
            'bulletins': 1.0,
            'news': 0.9,
            'general': 0.8
        }

        self._fit_vectorizer()

    def _fit_vectorizer(self):
        """Fit TF-IDF vectorizer with training data"""
        sample_texts = []

        for context_type, keywords in self.context_keywords.items():
            sample_texts.extend([' '.join(keywords)] * 10)

            for i in range(len(keywords)):
                if i + 2 <= len(keywords):
                    sample_texts.append(' '.join(keywords[i:i+2]))

        if sample_texts:
            self.vectorizer.fit(sample_texts)

    def classify(self, query: str) -> List[Tuple[str, float]]:
        """Classify query into context types"""
        try:
            query_lower = query.lower().strip()
            query_vec = self.vectorizer.transform([query_lower])

            scores = {}
            for context_type, keywords in self.context_keywords.items():
                context_text = ' '.join(keywords)
                context_vec = self.vectorizer.transform([context_text])
                similarity = cosine_similarity(query_vec, context_vec)[0][0]
                scores[context_type] = similarity * self.context_weights.get(context_type, 1.0)

            # Sort by score and take top 2
            top_contexts = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]

            # Filter low confidence
            return [(ctx, score) for ctx, score in top_contexts if score > 0.1]

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return [('general', 0.5)]


context_classifier = AdvancedContextClassifier()

def determine_top_contexts(query: str) -> List[Tuple[str, float]]:
    """Determine top 2 context types for a query"""
    return context_classifier.classify(query)

# System prompt
system_prompt = """You are an expert agriculture assistant with access to multiple specialized knowledge sources.

CONVERSATION CONTEXT:
{conversation_context}

AVAILABLE DOCUMENTS FROM RELEVANT SOURCES:
{context}

USER QUESTION: {input}

SELECTED KNOWLEDGE DOMAINS: {selected_contexts}

CRITICAL GUIDELINES:
1. Synthesize information from all available relevant sources
2. Maintain conversation continuity and reference previous discussions
3. For weather-related aspects, use weather data
4. For recent developments, use news sources
5. For disease/pest issues, use disease database
6. For official recommendations, use bulletin data
7. Format responses with clear headings and bullet points
8. Acknowledge when information is limited and suggest expert consultation

Provide a comprehensive, well-structured response:"""

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# from typing import List
from pydantic import Field
# from langchain_core.documents import Document
# from langchain_core.retrievers import BaseRetriever


class MultiContextRetriever(BaseRetriever):
    documents: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
        **kwargs
    ) -> List[Document]:
        return self.documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
        **kwargs
    ) -> List[Document]:
        return self.documents


# ==================== API ROUTES ====================

@Agribot_bp1.route("/chat", methods=["GET", "POST"])
def chat():
    """Enhanced chat endpoint with multi-index retrieval"""
    try:
        if request.method == "POST":
            msg = request.form.get("msg", "").strip()
        else:
            msg = request.args.get("msg", "").strip()

        if not msg:
            return jsonify({"error": "Please provide a question"}), 400

        logger.info(f"Processing query: {msg}")

        # Determine context types
        top_contexts = determine_top_contexts(msg)
        context_types = [ctx for ctx, score in top_contexts]

        # Add session preferences
        session_preferences = session_manager.get_context_preferences()
        if session_preferences and len(context_types) < 2:
            for pref_ctx in session_preferences:
                if pref_ctx not in context_types and len(context_types) < 2:
                    context_types.append(pref_ctx)

        if not context_types:
            context_types = ['general']

        if 'general' not in context_types:
            context_types.append('general')

        logger.info(f"Selected contexts: {context_types}")

        # Get documents from Pinecone
        all_documents = []
        for context_type in context_types:
            try:
                retriever = pinecone_manager.get_retriever(context_type)
                documents = retriever.get_relevant_documents(msg)
                for doc in documents:
                    doc.metadata['source_context'] = context_type
                all_documents.extend(documents)
                logger.info(f"Retrieved {len(documents)} documents from {context_type}")
            except Exception as e:
                logger.warning(f"Failed to retrieve from {context_type}: {e}")

        # Get conversation context
        conversation_context = session_manager.get_conversation_context(context_types)

        # Initialize LLM
        llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            openai_api_key=API_KEY_GPT,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=1500
        )

        # Create RAG chain
        prompt_template = ChatPromptTemplate.from_template(system_prompt)
        question_answer_chain = create_stuff_documents_chain(
            llm, prompt_template, document_variable_name="context"
        )

        multi_retriever = MultiContextRetriever(documents=all_documents)
        rag_chain = create_retrieval_chain(multi_retriever, question_answer_chain)

        response = rag_chain.invoke({
            "input": msg,
            "conversation_context": conversation_context,
            "selected_contexts": ", ".join(context_types)
        })
        answer = response.get("answer", "").strip()

        # Store in session
        session_manager.add_to_history(msg, answer, context_types, top_contexts)

        # Save to MongoDB chat history
        mongo_storage.save_chat_message(
            session_manager.get_session_id(),
            msg,
            answer,
            context_types,
            {"top_contexts": top_contexts}
        )

        return _format_response(answer, context_types, top_contexts)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 500


def _format_response(answer: str, context_types: List[str], top_contexts: List[Tuple[str, float]]) -> str:
    """Format enhanced response"""
    if not answer or "don't have specific information" in answer.lower():
        return "🌱 **Information Not Available**\n\nI don't have specific information about this. Please consult with local agricultural experts."

    context_headers = {
        "weather": "🌤️ Weather Information",
        "news": "📰 Agricultural News",
        "diseases": "🦠 Crop Health",
        "bulletins": "📋 Official Advisories",
        "general": "🌾 Farming Insights"
    }

    header_parts = [context_headers.get(ctx, "🌾 General Agriculture") for ctx in context_types]
    main_header = " • ".join(header_parts)

    confidence_info = ""
    if top_contexts:
        conf_items = [f"{ctx}({score:.2f})" for ctx, score in top_contexts]
        confidence_info = f"\n\n🔍 *Sources: {', '.join(conf_items)}*"

    return f"**{main_header}**{confidence_info}\n\n{answer}"


# ==================== ADMIN ROUTES ====================

@Agribot_bp1.route("/admin/status", methods=["GET"])
def admin_status():
    """Admin endpoint to check status"""
    try:
        status = admin_manager.get_scraping_status()
        return jsonify({"status": "success", "data": status})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/scrape/weather", methods=["POST"])
def admin_scrape_weather():
    """Admin endpoint to force weather scraping"""
    try:
        data = request.get_json() or {}
        locations = data.get('locations')
        count = admin_manager.scrape_weather_data(locations)
        return jsonify({"status": "success", "locations_processed": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/scrape/news", methods=["POST"])
def admin_scrape_news():
    """Admin endpoint to force news scraping"""
    try:
        count = admin_manager.scrape_news_data()
        return jsonify({"status": "success", "news_items_processed": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/scrape/bulletins", methods=["POST"])
def admin_scrape_bulletins():
    """Admin endpoint to force bulletin scraping"""
    try:
        data = request.get_json() or {}
        states = data.get('states')
        count = admin_manager.scrape_bulletins(states)
        return jsonify({"status": "success", "states_processed": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/scrape/diseases", methods=["POST"])
def admin_scrape_diseases():
    """Admin endpoint to force disease info scraping"""
    try:
        count = admin_manager.scrape_disease_info()
        return jsonify({"status": "success", "disease_entries_processed": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/scrape/all", methods=["POST"])
def admin_scrape_all():
    """Admin endpoint to scrape all data types"""
    try:
        results = {
            "weather": admin_manager.scrape_weather_data(),
            "news": admin_manager.scrape_news_data(),
            "bulletins": admin_manager.scrape_bulletins(),
            "diseases": admin_manager.scrape_disease_info()
        }
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/upload-pdf", methods=["POST"])
def admin_upload_pdf():
    """Admin endpoint to upload PDF to Pinecone index"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400

        file = request.files['file']
        index_type = request.form.get('index_type', 'general')

        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Only PDF files are allowed"}), 400

        allowed_indexes = ['weather', 'news', 'diseases', 'bulletins', 'general']
        if index_type not in allowed_indexes:
            return jsonify({"status": "error", "message": f"Invalid index type"}), 400

        # Save temporarily to process
        file_bytes = file.read()
        success = pinecone_manager.process_and_index_pdf_bytes(
            file_bytes, file.filename, index_type
        )

        if success:
            return jsonify({
                "status": "success",
                "message": f"PDF uploaded and indexed in {index_type}",
                "filename": file.filename,
                "index_type": index_type
            })
        else:
            return jsonify({"status": "error", "message": "Failed to process PDF"}), 500

    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "indexes_ready": list(pinecone_manager.vector_stores.keys()),
            "scraping_service_running": admin_manager.is_running,
            "mongodb_connected": mongo_storage.health_check()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Session management routes
@Agribot_bp1.route("/session/clear", methods=["POST"])
def clear_session():
    """Clear current session history"""
    try:
        session_manager.clear_session()
        return jsonify({"status": "success", "message": "Session cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/session/history", methods=["GET"])
def get_session_history():
    """Get current session history"""
    try:
        session_id = session_manager.get_session_id()
        messages = list(mongo_storage.db["chat_sessions"].find(
            {"session_id": session_id, "type": "message"}
        ).sort("timestamp", -1).limit(15))

        for msg in messages:
            msg["_id"] = str(msg["_id"])

        return jsonify({"status": "success", "history": messages})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= ENHANCED ADMIN ROUTES (MISSING) =================

@Agribot_bp1.route("/admin/context-analysis", methods=["POST"])
def analyze_context():
    """Analyze query context classification"""
    try:
        data = request.get_json()
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "Query required"}), 400

        top_contexts = determine_top_contexts(query)
        questionnaire_count = len(context_classifier.questionnaire_patterns)

        return jsonify({
            "status": "success",
            "query": query,
            "top_contexts": top_contexts,
            "questionnaire_patterns_loaded": questionnaire_count,
            "all_contexts": list(context_classifier.context_keywords.keys())
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/admin/reload-questionnaires", methods=["POST"])
def reload_questionnaires():
    """Reload questionnaire data"""
    try:
        global questionnaire_data, context_classifier
        questionnaire_data = load_questionnaires_from_mongodb()
        context_classifier = AdvancedContextClassifier()  # Reinitialize with new data

        return jsonify({
            "status": "success",
            "message": "Questionnaires reloaded successfully",
            "patterns_loaded": len(questionnaire_data)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ================= SESSION MANAGEMENT ROUTES (MISSING) =================

# @Agribot_bp1.route("/session/clear", methods=["POST"])
# def clear_session():
#     """Clear current session history"""
#     try:
#         session_id = session_manager.get_session_id()
#         if session_id in session_manager.conversation_history:
#             del session_manager.conversation_history[session_id]
#         if session_id in session_manager.context_scores:
#             del session_manager.context_scores[session_id]
#         session.clear()
#         return jsonify({"status": "success", "message": "Session cleared"})
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500


# @Agribot_bp1.route("/session/history", methods=["GET"])
# def get_session_history():
#     """Get current session history with context analysis"""
#     try:
#         session_id = session_manager.get_session_id()
#         history = session_manager.conversation_history.get(session_id, [])
#         preferences = session_manager.context_scores.get(session_id, {})

#         return jsonify({
#             "status": "success",
#             "session_id": session_id,
#             "history": history,
#             "context_preferences": preferences,
#             "total_messages": len(history)
#         })
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500


@Agribot_bp1.route("/session/contexts", methods=["GET"])
def get_session_contexts():
    """Get context usage statistics for current session"""
    try:
        session_id = session_manager.get_session_id()
        preferences = session_manager.context_scores.get(session_id, {})

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "context_usage": preferences,
            "top_contexts": session_manager.get_context_preferences()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ================= PDF UPLOAD MANAGEMENT ROUTES (MISSING) =================

# @Agribot_bp1.route("/admin/upload-pdf", methods=["POST"])
# def admin_upload_pdf():
#     """Admin endpoint to upload PDF to specific Pinecone index"""
#     try:
#         # Check if files are present
#         if 'file' not in request.files:
#             return jsonify({
#                 "status": "error",
#                 "message": "No file provided"
#             }), 400

#         file = request.files['file']
#         index_type = request.form.get('index_type', 'general')

#         # Validate file
#         if file.filename == '':
#             return jsonify({
#                 "status": "error",
#                 "message": "No file selected"
#             }), 400

#         if not allowed_file(file.filename):
#             return jsonify({
#                 "status": "error",
#                 "message": "Only PDF files are allowed"
#             }), 400

#         # Validate index type
#         allowed_indexes = ['weather', 'news', 'diseases', 'bulletins', 'general']
#         if index_type not in allowed_indexes:
#             return jsonify({
#                 "status": "error",
#                 "message": f"Invalid index type. Allowed: {', '.join(allowed_indexes)}"
#             }), 400

#         # Read file bytes for in-memory processing (no temp file)
#         file_bytes = file.read()

#         # Process PDF and add to Pinecone using PineconeManager
#         success = pinecone_manager.process_and_index_pdf_bytes(
#             file_bytes, file.filename, index_type
#         )

#         if success:
#             logger.info(f"PDF {file.filename} successfully added to {index_type} index")
#             return jsonify({
#                 "status": "success",
#                 "message": f"PDF successfully uploaded and indexed in {index_type}",
#                 "filename": file.filename,
#                 "index_type": index_type,
#                 "timestamp": datetime.now().isoformat()
#             })
#         else:
#             return jsonify({
#                 "status": "error",
#                 "message": "Failed to process and index PDF"
#             }), 500

#     except Exception as e:
#         logger.error(f"PDF upload error: {e}")
#         return jsonify({
#             "status": "error",
#             "message": f"Upload failed: {str(e)}"
#         }), 500


@Agribot_bp1.route("/admin/uploaded-files", methods=["GET"])
def get_uploaded_files():
    """Get list of uploaded PDF files"""
    try:
        uploaded_files = pinecone_manager.get_uploaded_files()

        # Convert ObjectId to string
        for file in uploaded_files:
            if "_id" in file:
                file["_id"] = str(file["_id"])

        return jsonify({
            "status": "success",
            "files": uploaded_files,
            "total_count": len(uploaded_files)
        })

    except Exception as e:
        logger.error(f"Error getting uploaded files: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@Agribot_bp1.route("/admin/delete-uploaded-file", methods=["DELETE"])
def delete_uploaded_file():
    """Delete an uploaded PDF file"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        index_type = data.get('index_type')

        if not filename or not index_type:
            return jsonify({
                "status": "error",
                "message": "Filename and index_type are required"
            }), 400

        success = pinecone_manager.delete_uploaded_file(filename, index_type)

        if success:
            return jsonify({
                "status": "success",
                "message": f"File {filename} deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "File not found or could not be deleted"
            }), 404

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
