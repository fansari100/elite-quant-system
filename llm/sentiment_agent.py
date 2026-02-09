"""
Elite Quant System - LLM-Powered Sentiment Analysis Agent
Integrates FinGPT/LLaMA-based models for advanced NLP
FinCon-style multi-agent reasoning for investment decisions
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer


class SentimentLabel(Enum):
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class NewsItem:
    """Represents a news article or social media post"""
    id: str
    timestamp: datetime
    source: str
    headline: str
    content: str
    symbols: List[str]
    url: Optional[str] = None


@dataclass
class SentimentResult:
    """Sentiment analysis result with confidence"""
    symbol: str
    sentiment: SentimentLabel
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str
    sources: List[str]
    timestamp: datetime


@dataclass
class InvestmentSignal:
    """Multi-agent consensus investment signal"""
    symbol: str
    direction: int  # -1, 0, 1
    conviction: float  # 0 to 1
    timeframe: str  # "intraday", "swing", "position"
    rationale: str
    sentiment_score: float
    analyst_agents: Dict[str, Dict]


class FinancialEmbeddings:
    """Semantic embeddings for financial text using specialized models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        
        # Financial domain keywords for enhanced retrieval
        self.domain_keywords = {
            "bullish": ["upgrade", "outperform", "beat", "growth", "expansion", "surge"],
            "bearish": ["downgrade", "underperform", "miss", "decline", "contraction", "plunge"],
            "risk": ["volatility", "uncertainty", "lawsuit", "investigation", "debt"],
            "catalyst": ["earnings", "merger", "acquisition", "product launch", "FDA approval"],
        }
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def semantic_search(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most relevant documents for a query"""
        query_embedding = self.encode([query])
        doc_embeddings = self.encode(documents)
        
        # Cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]


class FinancialSentimentClassifier:
    """Fine-tuned sentiment classifier for financial text"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}
    
    @torch.no_grad()
    def classify(self, texts: List[str]) -> List[Dict]:
        """Classify sentiment of financial texts"""
        results = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            pred_idx = probs.argmax().item()
            
            # Convert to score (-1 to 1)
            positive_prob = probs[0].item()
            negative_prob = probs[1].item()
            score = positive_prob - negative_prob
            
            results.append({
                "label": self.label_map[pred_idx],
                "score": score,
                "confidence": probs[pred_idx].item(),
                "probabilities": {
                    "positive": positive_prob,
                    "negative": negative_prob,
                    "neutral": probs[2].item()
                }
            })
        
        return results


class LLMReasoningAgent:
    """
    LLM-based reasoning agent for investment analysis
    Uses FinGPT-style prompting with chain-of-thought reasoning
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        use_quantization: bool = True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Quantization config for H200 efficiency
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # System prompts for different agent roles
        self.agent_prompts = {
            "analyst": """You are a senior quantitative analyst at a top hedge fund.
Your role is to analyze financial news and provide investment insights.
Always provide structured analysis with:
1. Key facts and their implications
2. Potential market impact (short-term and long-term)
3. Risk factors to consider
4. Confidence level (low/medium/high) with reasoning""",
            
            "risk_manager": """You are a risk manager at a systematic trading firm.
Your role is to identify risks in investment decisions.
Focus on:
1. Downside scenarios
2. Correlation risks
3. Liquidity concerns
4. Event risks
Be conservative and highlight potential pitfalls.""",
            
            "macro_strategist": """You are a macro strategist analyzing market conditions.
Your role is to provide context on how macro factors affect individual securities.
Consider:
1. Interest rate environment
2. Sector rotation trends
3. Economic indicators
4. Market sentiment and positioning""",
        }
    
    def _format_prompt(self, role: str, context: str, question: str) -> str:
        """Format prompt with role-specific system message"""
        system_prompt = self.agent_prompts.get(role, self.agent_prompts["analyst"])
        
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context}

Question: {question}
[/INST]"""
    
    @torch.no_grad()
    def generate_analysis(
        self, 
        role: str,
        context: str, 
        question: str,
        max_new_tokens: int = 512
    ) -> str:
        """Generate analysis using the LLM"""
        prompt = self._format_prompt(role, context, question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated response
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response


class MultiAgentSentimentSystem:
    """
    FinCon-inspired multi-agent system for sentiment analysis
    Combines multiple specialized agents for robust signals
    """
    
    def __init__(self):
        # Initialize components
        self.embeddings = FinancialEmbeddings()
        self.sentiment_classifier = FinancialSentimentClassifier()
        
        # LLM agent (lazy load for memory efficiency)
        self._llm_agent = None
        
        # Agent weights for ensemble
        self.agent_weights = {
            "sentiment_classifier": 0.3,
            "analyst": 0.3,
            "risk_manager": 0.2,
            "macro_strategist": 0.2,
        }
    
    @property
    def llm_agent(self) -> LLMReasoningAgent:
        """Lazy load LLM agent"""
        if self._llm_agent is None:
            self._llm_agent = LLMReasoningAgent()
        return self._llm_agent
    
    def aggregate_news(
        self, 
        news_items: List[NewsItem], 
        symbol: str
    ) -> str:
        """Aggregate relevant news for a symbol"""
        relevant_news = [n for n in news_items if symbol in n.symbols]
        
        if not relevant_news:
            return "No recent news available."
        
        # Sort by relevance using embeddings
        query = f"Investment outlook and price movement for {symbol}"
        headlines = [n.headline for n in relevant_news]
        
        ranked = self.embeddings.semantic_search(query, headlines, top_k=5)
        
        context_parts = []
        for idx, score in ranked:
            news = relevant_news[idx]
            context_parts.append(f"- [{news.source}] {news.headline}")
            if news.content:
                context_parts.append(f"  {news.content[:300]}...")
        
        return "\n".join(context_parts)
    
    async def analyze_symbol(
        self, 
        symbol: str, 
        news_items: List[NewsItem],
        use_llm: bool = True
    ) -> InvestmentSignal:
        """
        Comprehensive analysis of a symbol using multi-agent approach
        """
        results = {}
        
        # 1. Sentiment classification on headlines
        relevant_news = [n for n in news_items if symbol in n.symbols]
        if relevant_news:
            headlines = [n.headline for n in relevant_news]
            sentiment_results = self.sentiment_classifier.classify(headlines)
            
            avg_score = np.mean([r["score"] for r in sentiment_results])
            avg_confidence = np.mean([r["confidence"] for r in sentiment_results])
            
            results["sentiment_classifier"] = {
                "score": avg_score,
                "confidence": avg_confidence,
                "n_articles": len(headlines)
            }
        else:
            results["sentiment_classifier"] = {
                "score": 0.0,
                "confidence": 0.0,
                "n_articles": 0
            }
        
        # 2. LLM-based analysis (if enabled)
        if use_llm and relevant_news:
            context = self.aggregate_news(news_items, symbol)
            
            # Analyst view
            analyst_response = self.llm_agent.generate_analysis(
                "analyst",
                context,
                f"What is your investment recommendation for {symbol} based on this news?"
            )
            results["analyst"] = {
                "analysis": analyst_response,
                "score": self._extract_sentiment_from_text(analyst_response)
            }
            
            # Risk manager view
            risk_response = self.llm_agent.generate_analysis(
                "risk_manager",
                context,
                f"What are the key risks to consider for {symbol}?"
            )
            results["risk_manager"] = {
                "analysis": risk_response,
                "score": self._extract_sentiment_from_text(risk_response)
            }
            
            # Macro strategist view
            macro_response = self.llm_agent.generate_analysis(
                "macro_strategist",
                context,
                f"How do macro factors affect the outlook for {symbol}?"
            )
            results["macro_strategist"] = {
                "analysis": macro_response,
                "score": self._extract_sentiment_from_text(macro_response)
            }
        
        # 3. Ensemble the scores
        final_score = 0.0
        total_weight = 0.0
        
        for agent_name, weight in self.agent_weights.items():
            if agent_name in results:
                agent_score = results[agent_name].get("score", 0.0)
                if agent_score is not None:
                    final_score += weight * agent_score
                    total_weight += weight
        
        if total_weight > 0:
            final_score /= total_weight
        
        # 4. Determine direction and conviction
        if final_score > 0.3:
            direction = 1
        elif final_score < -0.3:
            direction = -1
        else:
            direction = 0
        
        conviction = min(abs(final_score), 1.0)
        
        # 5. Generate rationale
        if "analyst" in results:
            rationale = results["analyst"].get("analysis", "No analysis available")[:500]
        else:
            rationale = f"Sentiment score: {final_score:.2f}"
        
        return InvestmentSignal(
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            timeframe="swing",
            rationale=rationale,
            sentiment_score=final_score,
            analyst_agents=results
        )
    
    def _extract_sentiment_from_text(self, text: str) -> float:
        """Extract sentiment score from LLM-generated text"""
        text_lower = text.lower()
        
        bullish_keywords = ["bullish", "positive", "upgrade", "buy", "outperform", "growth"]
        bearish_keywords = ["bearish", "negative", "downgrade", "sell", "underperform", "decline"]
        
        bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
        
        if bullish_count + bearish_count == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / (bullish_count + bearish_count)


class SentimentAPI:
    """FastAPI-compatible interface for sentiment analysis"""
    
    def __init__(self):
        self.system = MultiAgentSentimentSystem()
    
    async def analyze(self, symbol: str, news_data: List[Dict]) -> Dict:
        """Analyze sentiment for a symbol"""
        news_items = [
            NewsItem(
                id=item.get("id", str(i)),
                timestamp=datetime.fromisoformat(item["timestamp"]),
                source=item.get("source", "unknown"),
                headline=item["headline"],
                content=item.get("content", ""),
                symbols=item.get("symbols", [symbol]),
                url=item.get("url")
            )
            for i, item in enumerate(news_data)
        ]
        
        signal = await self.system.analyze_symbol(symbol, news_items)
        
        return asdict(signal)


# CLI interface for testing
if __name__ == "__main__":
    import asyncio
    
    # Test data
    test_news = [
        {
            "id": "1",
            "timestamp": datetime.now().isoformat(),
            "source": "Reuters",
            "headline": "Apple reports record quarterly revenue, beats expectations",
            "content": "Apple Inc reported record quarterly revenue on strong iPhone sales...",
            "symbols": ["AAPL"]
        },
        {
            "id": "2", 
            "timestamp": datetime.now().isoformat(),
            "source": "Bloomberg",
            "headline": "Apple faces antitrust probe in European markets",
            "content": "European regulators have launched an investigation into Apple's practices...",
            "symbols": ["AAPL"]
        }
    ]
    
    async def main():
        api = SentimentAPI()
        result = await api.analyze("AAPL", test_news)
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(main())

