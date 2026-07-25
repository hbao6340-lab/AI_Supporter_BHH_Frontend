"""
Knowledge Base - AI Document Reader
Reads documents from knowledge folder and retrieves answers.
Falls back to OpenAI if no relevant content found.
"""

from .document_parser import DocumentParser, parser
from .retriever import KnowledgeRetriever, retriever

__all__ = ['DocumentParser', 'parser', 'KnowledgeRetriever', 'retriever']
