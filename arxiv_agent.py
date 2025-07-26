import os
import json
import time
import logging
import re
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
import arxiv
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data class to store paper information"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published_date: str
    categories: List[str]
    pdf_url: str
    content: Optional[str] = None

@dataclass
class SearchResult:
    """Data class for search results"""
    paper: Paper
    similarity_score: float
    relevant_chunks: List[str]
    rerank_score: Optional[float] = None
    relevance_explanation: Optional[str] = None

class ArxivAgent:
    """Arxiv Agent with RAG capabilities using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Arxiv Agent"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize semantic search models
        logger.info("Loading semantic search models...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize FAISS index for vector search
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.papers: List[Paper] = []
        self.chunks: List[str] = []
        self.chunk_to_paper: List[int] = []  # Maps chunk index to paper index
        self.chunk_metadata: List[Dict] = []  # Store metadata for each chunk
        
        # Routing thresholds
        self.min_similarity_threshold = 0.3
        self.min_rerank_threshold = 0.5
        
        logger.info("Arxiv Agent initialized successfully")
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search for papers on Arxiv"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    arxiv_id=result.entry_id.split('/')[-1],
                    published_date=result.published.strftime("%Y-%m-%d"),
                    categories=result.categories,
                    pdf_url=result.pdf_url
                )
                papers.append(paper)
                logger.info(f"Found paper: {paper.title}")
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def extract_paper_content(self, paper: Paper) -> str:
        """Extract full text content from paper (simplified version)"""
        try:
            # For now, we'll use the abstract and metadata
            # In a full implementation, you might want to download and parse PDFs
            content = f"""
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Abstract: {paper.abstract}
Categories: {', '.join(paper.categories)}
Published: {paper.published_date}
Arxiv ID: {paper.arxiv_id}
            """.strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {paper.arxiv_id}: {e}")
            return paper.abstract
    
    def chunk_text_advanced(self, text: str, paper: Paper) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = []
        sections = re.split(r'\n\s*\n', text)
        
        for section in sections:
            lines = section.split('\n')
            section_title = lines[0].strip()
            section_text = '\n'.join(lines[1:])
            
            # Split section into chunks
            chunk_size = 1000
            start = 0
            while start < len(section_text):
                end = start + chunk_size
                chunk_text = section_text[start:end]
                chunk = {
                    'text': chunk_text,
                    'section': section_title,
                    'type': 'text',
                    'paper_title': paper.title,
                    'arxiv_id': paper.arxiv_id
                }
                chunks.append(chunk)
                start = end
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence transformer model"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.array([])
    
    def build_knowledge_base(self, papers: List[Paper]) -> None:
        """Build knowledge base from papers"""
        try:
            self.papers = papers
            all_chunks = []
            chunk_to_paper = []
            chunk_metadata = []
            
            for paper_idx, paper in enumerate(papers):
                # Extract content
                content = self.extract_paper_content(paper)
                paper.content = content
                
                # Create chunks with better structure
                chunks = self.chunk_text_advanced(content, paper)
                
                for chunk_info in chunks:
                    all_chunks.append(chunk_info['text'])
                    chunk_to_paper.append(paper_idx)
                    chunk_metadata.append({
                        'paper_idx': paper_idx,
                        'section': chunk_info.get('section', 'content'),
                        'chunk_type': chunk_info.get('type', 'text'),
                        'paper_title': paper.title,
                        'arxiv_id': paper.arxiv_id
                    })
            
            self.chunks = all_chunks
            self.chunk_to_paper = chunk_to_paper
            self.chunk_metadata = chunk_metadata
            
            # Get embeddings
            logger.info(f"Getting embeddings for {len(all_chunks)} chunks...")
            embeddings = self.get_embeddings(all_chunks)
            
            if len(embeddings) > 0:
                # Build FAISS index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings)
                logger.info(f"Knowledge base built with {len(papers)} papers and {len(all_chunks)} chunks")
            else:
                logger.error("Failed to get embeddings")
                
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search knowledge base for relevant information with reranking"""
        try:
            if not self.chunks:
                return []
                
            # Get query embedding
            query_embedding = self.get_embeddings([query])
            if len(query_embedding) == 0:
                return []
            
            # Search index for initial candidates (get more for reranking)
            initial_k = min(top_k * 3, len(self.chunks))
            scores, indices = self.index.search(query_embedding, initial_k)
            
            # Filter by similarity threshold
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and score >= self.min_similarity_threshold:
                    paper_idx = self.chunk_to_paper[idx]
                    paper = self.papers[paper_idx]
                    chunk = self.chunks[idx]
                    metadata = self.chunk_metadata[idx]
                    
                    candidates.append({
                        'paper': paper,
                        'chunk': chunk,
                        'similarity_score': float(score),
                        'metadata': metadata,
                        'idx': idx
                    })
            
            if not candidates:
                return []
            
            # Rerank using cross-encoder
            reranked_candidates = self.rerank_results(query, candidates)
            
            # Convert to SearchResult objects
            results = []
            for candidate in reranked_candidates[:top_k]:
                result = SearchResult(
                    paper=candidate['paper'],
                    similarity_score=candidate['similarity_score'],
                    relevant_chunks=[candidate['chunk']],
                    rerank_score=candidate.get('rerank_score'),
                    relevance_explanation=candidate.get('relevance_explanation')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank search results using cross-encoder"""
        try:
            if not candidates:
                return []
            
            # Prepare query-document pairs for reranking
            pairs = [(query, candidate['chunk']) for candidate in candidates]
            
            # Get rerank scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add rerank scores to candidates
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
            
            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            # Fallback to similarity-based ranking
            return sorted(candidates, key=lambda x: x['similarity_score'], reverse=True)
    
    def should_use_direct_llm(self, query: str, search_results: List[SearchResult]) -> bool:
        """Determine if we should use direct LLM instead of RAG"""
        # Check if we have good search results
        if not search_results:
            return True
        
        # Check if best result meets quality threshold
        best_result = search_results[0]
        if (best_result.rerank_score is not None and 
            best_result.rerank_score < self.min_rerank_threshold):
            return True
        
        if best_result.similarity_score < self.min_similarity_threshold:
            return True
        
        # Check for general knowledge questions
        general_keywords = ['what is', 'define', 'explain', 'how does', 'why', 'general']
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in general_keywords):
            # If it's a general question but we have good results, use RAG
            if best_result.similarity_score > 0.6:
                return False
            return True
        
        return False
    
    def generate_direct_llm_response(self, query: str) -> str:
        """Generate response using Gemini directly without RAG"""
        try:
            prompt = f"""
You are an AI assistant with expertise in academic research and scientific papers. 
The user has asked a question that may not be directly answerable from the available paper database.

User Question: {query}

Please provide a helpful and informative response based on your general knowledge. 
If this is a question that would benefit from specific research papers, suggest what types of papers or research areas the user should look for.

Response:
            """.strip()
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating direct LLM response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate response using Gemini with smart routing between RAG and direct LLM"""
        try:
            # Smart routing: decide between RAG and direct LLM
            if self.should_use_direct_llm(query, search_results):
                logger.info("Using direct LLM response (no relevant papers found or low quality results)")
                return self.generate_direct_llm_response(query)
            
            # Use RAG with paper context
            logger.info(f"Using RAG response with {len(search_results)} relevant papers")
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results):
                rerank_info = f" (Rerank: {result.rerank_score:.3f})" if result.rerank_score else ""
                context_parts.append(f"""
Paper {i+1}: {result.paper.title}
Authors: {', '.join(result.paper.authors)}
Arxiv ID: {result.paper.arxiv_id}
Relevant Content: {result.relevant_chunks[0]}
Similarity Score: {result.similarity_score:.3f}{rerank_info}
                """.strip())
            
            context = "\n\n".join(context_parts)
            
            # Create enhanced prompt
            prompt = f"""
You are an AI assistant specialized in analyzing Arxiv papers. Use the following context from relevant papers to answer the user's question.

Context from papers (ranked by relevance):
{context}

User Question: {query}

Instructions:
1. Provide a comprehensive answer based primarily on the paper content provided
2. Cite specific papers when referencing information (use paper titles and arxiv IDs)
3. If the context doesn't fully answer the question, acknowledge the limitations
4. Suggest what additional information or papers might be needed
5. Be precise and academic in your response

Answer:
            """.strip()
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def chat(self, query: str, search_query: Optional[str] = None) -> str:
        """Main chat interface with smart routing"""
        try:
            # Smart routing: if no papers loaded and no search query, use direct LLM
            if not self.papers and not search_query:
                logger.info("No papers loaded and no search query provided, using direct LLM")
                return self.generate_direct_llm_response(query)
            
            # If no papers loaded but search query provided, search for papers
            if not self.papers and search_query:
                logger.info(f"Searching for papers with query: {search_query}")
                papers = self.search_papers(search_query, max_results=5)
                if papers:
                    self.build_knowledge_base(papers)
                else:
                    logger.info("No papers found, falling back to direct LLM")
                    return self.generate_direct_llm_response(query)
            
            # Search knowledge base with enhanced semantic search
            search_results = self.search_knowledge_base(query, top_k=5)
            
            # Generate response with smart routing
            response = self.generate_response(query, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    def get_paper_info(self, arxiv_id: str) -> Optional[Paper]:
        """Get detailed information about a specific paper"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            
            paper = Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                arxiv_id=result.entry_id.split('/')[-1],
                published_date=result.published.strftime("%Y-%m-%d"),
                categories=result.categories,
                pdf_url=result.pdf_url
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error getting paper info for {arxiv_id}: {e}")
            return None 