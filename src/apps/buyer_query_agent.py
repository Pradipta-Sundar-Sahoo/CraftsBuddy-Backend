"""
Buyer Query Processing System using LangGraph
Implements the complete buyer query workflow with multiple agents
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
import pinecone
from pinecone import Pinecone
import numpy as np

# Local imports
from src.core.settings import config
from src.core.database_service import DatabaseService

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Intent types for buyer queries"""
    SHOPPING = "shopping"
    INFO = "info"
    UNCLEAR = "unclear"

class SentimentType(Enum):
    """Sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class QueryState:
    """State object for the query processing workflow"""
    # Input
    query: str = ""
    user_id: Optional[int] = None
    
    # Processing state
    preprocessed_query: str = ""
    intent: Optional[IntentType] = None
    sentiment: Optional[SentimentType] = None
    intent_confidence: float = 0.0
    
    # Search and retrieval
    embeddings: Optional[List[float]] = None
    retrieved_products: List[Dict] = field(default_factory=list)
    reranked_products: List[Dict] = field(default_factory=list)
    related_products: List[Dict] = field(default_factory=list)
    
    # Decision making
    needs_clarification: bool = False
    clarification_message: str = ""
    
    # Final response
    final_response: Dict = field(default_factory=dict)
    
    # Metadata
    processing_time: float = 0.0
    workflow_path: List[str] = field(default_factory=list)

class PreprocessorAgent:
    """Preprocesses buyer queries for better understanding"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.1
        ) if config.USE_GEMINI else None
    
    async def process(self, state: QueryState) -> QueryState:
        """Preprocess the query"""
        try:
            state.workflow_path.append("preprocessor")
            
            if not self.llm:
                # Simple preprocessing without LLM
                state.preprocessed_query = state.query.strip().lower()
                return state
            
            system_prompt = """
            You are a query preprocessor for an artisan marketplace. 
            Clean and standardize the user query while preserving intent.
            
            Tasks:
            1. Fix spelling and grammar
            2. Expand abbreviations
            3. Standardize product names
            4. Remove unnecessary words
            
            Return only the cleaned query, nothing else.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {state.query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            state.preprocessed_query = response.content.strip()
            
            logger.info(f"Preprocessed query: '{state.query}' -> '{state.preprocessed_query}'")
            return state
            
        except Exception as e:
            logger.error(f"Preprocessor error: {e}")
            state.preprocessed_query = state.query
            return state

class IntentDetectorAgent:
    """Detects buyer intent using Gemini LLM"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.1
        ) if config.USE_GEMINI else None
    
    async def detect_intent(self, state: QueryState) -> QueryState:
        """Detect intent from the query"""
        try:
            state.workflow_path.append("intent_detector")
            
            if not self.llm:
                # Fallback intent detection
                return self._fallback_intent_detection(state)
            
            system_prompt = """
            You are an intent classifier for an artisan marketplace. 
            Classify the user query into one of these intents:
            
            1. SHOPPING - User wants to buy/find products (e.g., "I need handmade pottery", "show me blue scarves")
            2. INFO - User wants information about products/sellers (e.g., "tell me about this artist", "what materials are used")
            3. UNCLEAR - Query is ambiguous or unclear
            
            Respond with JSON format:
            {
                "intent": "SHOPPING|INFO|UNCLEAR",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            }
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {state.preprocessed_query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                result = json.loads(response.content)
                state.intent = IntentType(result["intent"].lower())
                state.intent_confidence = float(result["confidence"])
                
                logger.info(f"Intent detected: {state.intent.value} (confidence: {state.intent_confidence})")
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to parse intent response: {e}")
                return self._fallback_intent_detection(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return self._fallback_intent_detection(state)
    
    def _fallback_intent_detection(self, state: QueryState) -> QueryState:
        """Fallback intent detection using keywords"""
        query_lower = state.preprocessed_query.lower()
        
        shopping_keywords = ["buy", "purchase", "need", "want", "looking for", "show me", "find"]
        info_keywords = ["what", "how", "why", "tell me", "about", "information"]
        
        shopping_score = sum(1 for word in shopping_keywords if word in query_lower)
        info_score = sum(1 for word in info_keywords if word in query_lower)
        
        if shopping_score > info_score:
            state.intent = IntentType.SHOPPING
            state.intent_confidence = min(0.8, shopping_score * 0.2)
        elif info_score > 0:
            state.intent = IntentType.INFO
            state.intent_confidence = min(0.8, info_score * 0.2)
        else:
            state.intent = IntentType.UNCLEAR
            state.intent_confidence = 0.3
        
        return state

class SentimentAnalyzerAgent:
    """Analyzes sentiment of buyer queries"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.1
        ) if config.USE_GEMINI else None
    
    async def analyze_sentiment(self, state: QueryState) -> QueryState:
        """Analyze sentiment of the query"""
        try:
            state.workflow_path.append("sentiment_analyzer")
            
            if not self.llm:
                # Fallback sentiment analysis
                state.sentiment = SentimentType.NEUTRAL
                return state
            
            system_prompt = """
            Analyze the sentiment of this buyer query for an artisan marketplace.
            
            Classify as:
            - POSITIVE: Happy, excited, satisfied
            - NEGATIVE: Frustrated, disappointed, angry
            - NEUTRAL: Factual, neutral tone
            
            Respond with only: POSITIVE, NEGATIVE, or NEUTRAL
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {state.preprocessed_query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            sentiment_str = response.content.strip().upper()
            
            try:
                state.sentiment = SentimentType(sentiment_str.lower())
                logger.info(f"Sentiment analyzed: {state.sentiment.value}")
            except ValueError:
                state.sentiment = SentimentType.NEUTRAL
                logger.warning(f"Unknown sentiment: {sentiment_str}, defaulting to NEUTRAL")
            
            return state
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            state.sentiment = SentimentType.NEUTRAL
            return state

class EmbeddingAgent:
    """Generates embeddings for queries and products using OpenAI"""
    
    def __init__(self):
        if config.USE_OPENAI:
            self.embeddings_model = OpenAIEmbeddings(
                openai_api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_EMBEDDINGS_MODEL
            )
        else:
            self.embeddings_model = None
            logger.warning("OpenAI not configured, embeddings will be disabled")
    
    async def generate_embeddings(self, state: QueryState) -> QueryState:
        """Generate embeddings for the query"""
        try:
            state.workflow_path.append("embedding_generator")
            
            if not self.embeddings_model:
                logger.warning("Embeddings model not available")
                state.embeddings = []
                return state
            
            # Generate embeddings for the preprocessed query
            embeddings = await self.embeddings_model.aembed_query(state.preprocessed_query)
            state.embeddings = embeddings
            
            logger.info(f"Generated OpenAI embeddings of dimension: {len(embeddings)}")
            return state
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            state.embeddings = []
            return state

class VectorSearchDatabase:
    """Vector search database using Pinecone"""
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.pc = None
        self.index = None
        
        if config.USE_PINECONE:
            try:
                # Initialize Pinecone
                self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
                self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
                logger.info(f"Connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                self.pc = None
                self.index = None
        else:
            logger.warning("Pinecone not configured, vector search will be disabled")
    
    async def search_similar_products(self, embeddings: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar products using embeddings"""
        try:
            if not self.index or not embeddings:
                logger.warning("Pinecone index not available or no embeddings provided")
                return []
            
            # Query Pinecone for similar vectors
            results = self.index.query(
                vector=embeddings,
                top_k=limit,
                include_metadata=True,
                include_values=False
            )
            
            # Convert Pinecone results to product dictionaries
            products = []
            for match in results.matches:
                try:
                    product_id = int(match.id)
                    # Get full product details from database
                    product = self.db_service.get_product_by_id(product_id)
                    if product:
                        product['similarity_score'] = float(match.score)
                        products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to process product {match.id}: {e}")
            
            logger.info(f"Found {len(products)} similar products from Pinecone")
            return products
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def add_product_embeddings(self, product_id: int, embeddings: List[float], metadata: Dict):
        """Add product embeddings to Pinecone"""
        try:
            if not self.index:
                logger.warning("Pinecone index not available")
                return False
            
            # Upsert vector to Pinecone
            self.index.upsert(
                vectors=[{
                    "id": str(product_id),
                    "values": embeddings,
                    "metadata": metadata
                }]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add product embeddings to Pinecone: {e}")
            return False
    
    async def create_index_if_not_exists(self, dimension: int = 1536):
        """Create Pinecone index if it doesn't exist"""
        try:
            if not self.pc:
                logger.error("Pinecone client not initialized")
                return False
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if config.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {config.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=dimension,
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info("Pinecone index created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False

class RetrieverAgent:
    """Retrieves relevant products based on query"""
    
    def __init__(self):
        self.vector_db = VectorSearchDatabase()
        self.db_service = DatabaseService()
    
    async def retrieve_products(self, state: QueryState) -> QueryState:
        """Retrieve products based on embeddings and keywords"""
        try:
            state.workflow_path.append("retriever")
            
            # Vector search
            vector_products = []
            if state.embeddings:
                vector_products = await self.vector_db.search_similar_products(state.embeddings, limit=20)
            
            # Keyword search as fallback
            keyword_products = []
            try:
                keyword_products, _ = self.db_service.search_products_advanced(state.preprocessed_query, 1, 20)
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
            
            # Combine and deduplicate
            all_products = {}
            
            # Add vector search results
            for product in vector_products:
                product['retrieval_method'] = 'vector'
                all_products[product['id']] = product
            
            # Add keyword search results
            for product in keyword_products:
                if product['id'] not in all_products:
                    product['retrieval_method'] = 'keyword'
                    product['similarity_score'] = 0.5  # Default score for keyword matches
                    all_products[product['id']] = product
            
            state.retrieved_products = list(all_products.values())
            
            logger.info(f"Retrieved {len(state.retrieved_products)} products")
            return state
            
        except Exception as e:
            logger.error(f"Product retrieval error: {e}")
            state.retrieved_products = []
            return state

class RerankerAgent:
    """Re-ranks retrieved products based on relevance"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.1
        ) if config.USE_GEMINI else None
    
    async def rerank_products(self, state: QueryState) -> QueryState:
        """Re-rank products based on query relevance"""
        try:
            state.workflow_path.append("reranker")
            
            if not state.retrieved_products:
                state.reranked_products = []
                return state
            
            if not self.llm:
                # Simple reranking by similarity score
                state.reranked_products = sorted(
                    state.retrieved_products,
                    key=lambda x: x.get('similarity_score', 0),
                    reverse=True
                )[:10]
                return state
            
            # Advanced reranking using LLM
            products_text = "\n".join([
                f"{i+1}. {p['title']} - {p.get('description', '')[:100]}..."
                for i, p in enumerate(state.retrieved_products[:15])
            ])
            
            system_prompt = f"""
            Rerank these products based on relevance to the query: "{state.preprocessed_query}"
            
            Products:
            {products_text}
            
            Return only the numbers of the top 10 most relevant products in order, comma-separated.
            Example: 3,1,7,2,5,8,4,6,9,10
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Rank the products by relevance:")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                rankings = [int(x.strip()) - 1 for x in response.content.strip().split(',')]
                reranked = []
                
                for rank in rankings:
                    if 0 <= rank < len(state.retrieved_products):
                        product = state.retrieved_products[rank].copy()
                        product['rerank_score'] = len(rankings) - len(reranked)
                        reranked.append(product)
                
                state.reranked_products = reranked[:10]
                
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse reranking: {e}")
                # Fallback to similarity score ranking
                state.reranked_products = sorted(
                    state.retrieved_products,
                    key=lambda x: x.get('similarity_score', 0),
                    reverse=True
                )[:10]
            
            logger.info(f"Reranked to {len(state.reranked_products)} products")
            return state
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            state.reranked_products = state.retrieved_products[:10]
            return state

class RelatedProductsAgent:
    """Finds related products based on current results"""
    
    def __init__(self):
        self.db_service = DatabaseService()
    
    async def find_related_products(self, state: QueryState) -> QueryState:
        """Find products related to the current results"""
        try:
            state.workflow_path.append("related_products")
            
            if not state.reranked_products:
                state.related_products = []
                return state
            
            # Get categories and sellers from top results
            categories = set()
            sellers = set()
            
            for product in state.reranked_products[:3]:
                if product.get('category'):
                    categories.add(product['category'])
                if product.get('user_id'):
                    sellers.add(product['user_id'])
            
            # Find related products by category and seller
            related = []
            existing_ids = {p['id'] for p in state.reranked_products}
            
            # Related by category
            for category in categories:
                try:
                    category_products, _ = self.db_service.search_products_by_category(category, 1, 5)
                    for product in category_products:
                        if product['id'] not in existing_ids and len(related) < 5:
                            product['relation_type'] = 'category'
                            related.append(product)
                            existing_ids.add(product['id'])
                except Exception as e:
                    logger.warning(f"Category search failed: {e}")
            
            # Related by seller (same artisan's other products)
            for seller_id in sellers:
                try:
                    seller_products = self.db_service.get_products_by_user_id(seller_id, limit=3)
                    for product in seller_products:
                        if product['id'] not in existing_ids and len(related) < 8:
                            product['relation_type'] = 'seller'
                            related.append(product)
                            existing_ids.add(product['id'])
                except Exception as e:
                    logger.warning(f"Seller products search failed: {e}")
            
            state.related_products = related
            
            logger.info(f"Found {len(state.related_products)} related products")
            return state
            
        except Exception as e:
            logger.error(f"Related products error: {e}")
            state.related_products = []
            return state

class SuggestionAgent:
    """Provides suggestions when query intent is unclear"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.7
        ) if config.USE_GEMINI else None
        self.db_service = DatabaseService()
    
    async def generate_suggestions(self, state: QueryState) -> QueryState:
        """Generate helpful suggestions for unclear queries"""
        try:
            state.workflow_path.append("suggestion_agent")
            
            if not self.llm:
                # Fallback suggestions
                state.final_response = {
                    "type": "suggestions",
                    "message": "I'm not sure what you're looking for. Here are some popular categories:",
                    "suggestions": [
                        "Handmade pottery and ceramics",
                        "Artisan jewelry and accessories",
                        "Traditional textiles and fabrics",
                        "Wooden crafts and furniture",
                        "Paintings and artwork"
                    ]
                }
                return state
            
            # Get trending categories or products
            try:
                trending_products, _ = self.db_service.get_products_paginated(1, 5)
                trending_text = ", ".join([p['title'] for p in trending_products])
            except:
                trending_text = "pottery, jewelry, textiles, paintings"
            
            system_prompt = f"""
            The user query "{state.query}" is unclear. Generate helpful suggestions for an artisan marketplace.
            
            Consider these trending items: {trending_text}
            
            Provide 3-5 specific suggestions that might help the user find what they're looking for.
            Be friendly and helpful.
            
            Respond in JSON format:
            {{
                "message": "clarifying question or helpful message",
                "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
            }}
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate suggestions for: {state.query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                result = json.loads(response.content)
                state.final_response = {
                    "type": "suggestions",
                    "message": result["message"],
                    "suggestions": result["suggestions"]
                }
            except json.JSONDecodeError:
                # Fallback
                state.final_response = {
                    "type": "suggestions",
                    "message": "Could you be more specific? Here are some popular categories:",
                    "suggestions": [
                        "Handmade pottery and ceramics",
                        "Artisan jewelry and accessories",
                        "Traditional textiles and fabrics"
                    ]
                }
            
            return state
            
        except Exception as e:
            logger.error(f"Suggestion generation error: {e}")
            state.final_response = {
                "type": "error",
                "message": "I'm having trouble understanding your request. Please try rephrasing."
            }
            return state

class ClarificationAgent:
    """Asks for clarification when needed"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3
        ) if config.USE_GEMINI else None
    
    async def ask_for_clarification(self, state: QueryState) -> QueryState:
        """Ask user for clarification"""
        try:
            state.workflow_path.append("clarification_agent")
            
            if not self.llm:
                state.final_response = {
                    "type": "clarification",
                    "message": "Could you please provide more details about what you're looking for?",
                    "follow_up_questions": [
                        "What type of product are you interested in?",
                        "Do you have a specific price range in mind?",
                        "Are you looking for a particular style or material?"
                    ]
                }
                return state
            
            system_prompt = """
            Generate a helpful clarification request for an artisan marketplace query.
            The user's query is unclear or ambiguous.
            
            Provide:
            1. A friendly clarifying message
            2. 2-3 specific follow-up questions to help narrow down what they want
            
            Respond in JSON format:
            {
                "message": "clarifying message",
                "follow_up_questions": ["question 1", "question 2", "question 3"]
            }
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {state.query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                result = json.loads(response.content)
                state.final_response = {
                    "type": "clarification",
                    "message": result["message"],
                    "follow_up_questions": result["follow_up_questions"]
                }
            except json.JSONDecodeError:
                state.final_response = {
                    "type": "clarification",
                    "message": "I'd like to help you find the perfect artisan product. Could you tell me more about what you're looking for?",
                    "follow_up_questions": [
                        "What category of product interests you?",
                        "Do you have a specific use case in mind?",
                        "Are you looking for something with particular materials or colors?"
                    ]
                }
            
            return state
            
        except Exception as e:
            logger.error(f"Clarification error: {e}")
            state.final_response = {
                "type": "clarification",
                "message": "Could you please provide more details about what you're looking for?"
            }
            return state

class SupportAgent:
    """Handles negative sentiment and support requests"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3
        ) if config.USE_GEMINI else None
    
    async def provide_support(self, state: QueryState) -> QueryState:
        """Provide support for negative sentiment queries"""
        try:
            state.workflow_path.append("support_agent")
            
            if not self.llm:
                state.final_response = {
                    "type": "support",
                    "message": "I understand you may be having some concerns. How can I help you today?",
                    "support_options": [
                        "Contact customer service",
                        "View return policy",
                        "Get help with your order"
                    ]
                }
                return state
            
            system_prompt = """
            The user seems frustrated or has negative sentiment. Provide empathetic support.
            
            Generate a supportive response that:
            1. Acknowledges their concern
            2. Offers helpful solutions
            3. Maintains a positive, professional tone
            
            Respond in JSON format:
            {
                "message": "supportive message",
                "support_options": ["option 1", "option 2", "option 3"]
            }
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User query: {state.query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                result = json.loads(response.content)
                state.final_response = {
                    "type": "support",
                    "message": result["message"],
                    "support_options": result["support_options"]
                }
            except json.JSONDecodeError:
                state.final_response = {
                    "type": "support",
                    "message": "I'm here to help! Let me know how I can assist you with finding the perfect artisan products.",
                    "support_options": [
                        "Browse featured artisans",
                        "Contact customer support",
                        "View our quality guarantee"
                    ]
                }
            
            return state
            
        except Exception as e:
            logger.error(f"Support error: {e}")
            state.final_response = {
                "type": "support",
                "message": "I'm here to help! How can I assist you today?"
            }
            return state

class FallbackAgent:
    """Fallback agent for when no matches are found"""
    
    def __init__(self):
        self.db_service = DatabaseService()
    
    async def provide_fallback(self, state: QueryState) -> QueryState:
        """Provide fallback response with trending or curated picks"""
        try:
            state.workflow_path.append("fallback_agent")
            
            # Get some trending or featured products
            try:
                trending_products, _ = self.db_service.get_products_paginated(1, 6)
            except:
                trending_products = []
            
            state.final_response = {
                "type": "fallback",
                "message": "I couldn't find exact matches, but here are some trending artisan products you might like:",
                "products": trending_products,
                "suggestion": "Try browsing our categories or contact our artisans directly for custom requests."
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Fallback error: {e}")
            state.final_response = {
                "type": "error",
                "message": "I'm having trouble processing your request right now. Please try again later."
            }
            return state

class BuyerQueryAgent:
    """Main orchestrator for the buyer query processing workflow"""
    
    def __init__(self):
        # Initialize all agents
        self.preprocessor = PreprocessorAgent()
        self.intent_detector = IntentDetectorAgent()
        self.sentiment_analyzer = SentimentAnalyzerAgent()
        self.embedding_agent = EmbeddingAgent()
        self.retriever = RetrieverAgent()
        self.reranker = RerankerAgent()
        self.related_products = RelatedProductsAgent()
        self.suggestion_agent = SuggestionAgent()
        self.clarification_agent = ClarificationAgent()
        self.support_agent = SupportAgent()
        self.fallback_agent = FallbackAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the workflow graph
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("preprocessor", self.preprocessor.process)
        workflow.add_node("intent_detector", self.intent_detector.detect_intent)
        workflow.add_node("sentiment_analyzer", self.sentiment_analyzer.analyze_sentiment)
        workflow.add_node("embedding_generator", self.embedding_agent.generate_embeddings)
        workflow.add_node("retriever", self.retriever.retrieve_products)
        workflow.add_node("reranker", self.reranker.rerank_products)
        workflow.add_node("related_products", self.related_products.find_related_products)
        workflow.add_node("suggestion_agent", self.suggestion_agent.generate_suggestions)
        workflow.add_node("clarification_agent", self.clarification_agent.ask_for_clarification)
        workflow.add_node("support_agent", self.support_agent.provide_support)
        workflow.add_node("fallback_agent", self.fallback_agent.provide_fallback)
        workflow.add_node("final_results", self._prepare_final_results)
        
        # Set entry point
        workflow.set_entry_point("preprocessor")
        
        # Add edges
        workflow.add_edge("preprocessor", "intent_detector")
        workflow.add_edge("intent_detector", "sentiment_analyzer")
        
        # Decision node after sentiment analysis
        workflow.add_conditional_edges(
            "sentiment_analyzer",
            self._route_after_sentiment,
            {
                "shopping": "embedding_generator",
                "info": "embedding_generator", 
                "unclear": "suggestion_agent",
                "support": "support_agent"
            }
        )
        
        # Shopping/Info flow
        workflow.add_edge("embedding_generator", "retriever")
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("reranker", "related_products")
        
        # Decision node after product retrieval
        workflow.add_conditional_edges(
            "related_products",
            self._route_after_retrieval,
            {
                "results": "final_results",
                "no_results": "fallback_agent",
                "unclear": "clarification_agent"
            }
        )
        
        # End nodes
        workflow.add_edge("suggestion_agent", END)
        workflow.add_edge("clarification_agent", END)
        workflow.add_edge("support_agent", END)
        workflow.add_edge("fallback_agent", END)
        workflow.add_edge("final_results", END)
        
        return workflow.compile()
    
    def _route_after_sentiment(self, state: QueryState) -> str:
        """Route based on intent and sentiment"""
        
        # Handle negative sentiment first
        if state.sentiment == SentimentType.NEGATIVE:
            return "support"
        
        # Route based on intent
        if state.intent == IntentType.SHOPPING and state.intent_confidence > 0.6:
            return "shopping"
        elif state.intent == IntentType.INFO and state.intent_confidence > 0.6:
            return "info"
        else:
            return "unclear"
    
    def _route_after_retrieval(self, state: QueryState) -> str:
        """Route based on retrieval results"""
        
        if len(state.reranked_products) >= 3:
            return "results"
        elif len(state.reranked_products) > 0:
            return "results"  # Show what we found
        elif state.intent_confidence < 0.5:
            return "unclear"
        else:
            return "no_results"
    
    async def _prepare_final_results(self, state: QueryState) -> QueryState:
        """Prepare final results for shopping queries"""
        state.workflow_path.append("final_results")
        
        state.final_response = {
            "type": "products",
            "message": f"Found {len(state.reranked_products)} products matching your query",
            "products": state.reranked_products,
            "related_products": state.related_products,
            "query_intent": state.intent.value if state.intent else None,
            "query_sentiment": state.sentiment.value if state.sentiment else None
        }
        
        return state
    
    async def process_query(self, query: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Process a buyer query through the complete workflow"""
        start_time = datetime.now()
        
        try:
            # Initialize state
            initial_state = QueryState(
                query=query,
                user_id=user_id
            )
            
            logger.info(f"Processing buyer query: '{query}'")
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            final_state.processing_time = processing_time
            
            # Prepare response
            response = {
                **final_state.final_response,
                "processing_time": processing_time,
                "workflow_path": final_state.workflow_path,
                "query_metadata": {
                    "original_query": query,
                    "preprocessed_query": final_state.preprocessed_query,
                    "intent": final_state.intent.value if final_state.intent else None,
                    "intent_confidence": final_state.intent_confidence,
                    "sentiment": final_state.sentiment.value if final_state.sentiment else None
                }
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            logger.info(f"Workflow path: {' -> '.join(final_state.workflow_path)}")
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query processing failed after {processing_time:.2f}s: {e}")
            
            return {
                "type": "error",
                "message": "I'm having trouble processing your request right now. Please try again later.",
                "processing_time": processing_time,
                "error": str(e)
            }

# Global instance
buyer_query_agent = BuyerQueryAgent()

# Utility functions for integration
async def process_buyer_query(query: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    """Main entry point for processing buyer queries"""
    return await buyer_query_agent.process_query(query, user_id)

async def initialize_product_embeddings():
    """Initialize embeddings for existing products using OpenAI and Pinecone"""
    try:
        logger.info("Initializing product embeddings with OpenAI and Pinecone...")
        
        db_service = DatabaseService()
        embedding_agent = EmbeddingAgent()
        vector_db = VectorSearchDatabase()
        
        # Create Pinecone index if it doesn't exist
        await vector_db.create_index_if_not_exists(dimension=1536)  # OpenAI text-embedding-3-small dimension
        
        if not embedding_agent.embeddings_model:
            logger.error("OpenAI embeddings not available. Please configure OPENAI_API_KEY.")
            return
        
        if not vector_db.index:
            logger.error("Pinecone index not available. Please configure Pinecone credentials.")
            return
        
        # Get all active products
        products, _ = db_service.get_products_paginated(1, 1000)  # Adjust as needed
        
        logger.info(f"Processing {len(products)} products for embeddings...")
        
        for i, product in enumerate(products):
            try:
                # Create embedding text from product
                embedding_text = f"{product.get('product_name', product.get('title', ''))} {product.get('description', '')} {product.get('category', '')}"
                
                # Generate embeddings using OpenAI
                embeddings = await embedding_agent.embeddings_model.aembed_query(embedding_text)
                
                # Prepare metadata for Pinecone
                metadata = {
                    'title': product.get('product_name', product.get('title', '')),
                    'category': product.get('category', ''),
                    'price': product.get('price', 0),
                    'user_id': product.get('user_id', 0),
                    'is_active': product.get('is_active', True)
                }
                
                # Add to Pinecone
                success = await vector_db.add_product_embeddings(
                    product['id'],
                    embeddings,
                    metadata
                )
                
                if success:
                    logger.info(f"✅ Processed product {i+1}/{len(products)}: {product.get('product_name', product.get('title', 'Unknown'))}")
                else:
                    logger.warning(f"❌ Failed to process product {product['id']}")
                
            except Exception as e:
                logger.error(f"Failed to process product {product['id']}: {e}")
        
        logger.info(f"🎉 Successfully initialized embeddings for {len(products)} products in Pinecone!")
        
    except Exception as e:
        logger.error(f"Failed to initialize product embeddings: {e}")

async def setup_buyer_query_system():
    """Complete setup of the buyer query system"""
    try:
        logger.info("🚀 Setting up Buyer Query System...")
        
        # Check configurations
        if not config.USE_GEMINI:
            logger.warning("⚠️  Gemini API not configured. Some features will use fallback methods.")
        
        if not config.USE_OPENAI:
            logger.error("❌ OpenAI API not configured. Please set OPENAI_API_KEY environment variable.")
            return False
        
        if not config.USE_PINECONE:
            logger.error("❌ Pinecone not configured. Please set PINECONE_API_KEY and PINECONE_ENVIRONMENT.")
            return False
        
        # Initialize embeddings
        await initialize_product_embeddings()
        
        logger.info("✅ Buyer Query System setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False
