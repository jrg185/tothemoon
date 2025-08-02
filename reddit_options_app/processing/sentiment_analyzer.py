"""
Sentiment Analyzer for Reddit Options App
Multi-layered sentiment analysis for WSB posts and comments
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import statistics

# Sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import openai
import json
import time

from config.settings import LLM_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """Advanced sentiment analyzer specialized for financial/trading text"""

    def __init__(self):
        """Initialize sentiment analyzer with multiple methods"""

        # Initialize VADER (rule-based, great for social media)
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Financial keywords for enhanced analysis
        self.bullish_keywords = {
            'moon', 'rocket', 'mooning', 'calls', 'bull', 'bullish', 'up', 'green', 'gains',
            'pump', 'squeeze', 'hodl', 'diamond', 'hands', 'buy', 'buying', 'long',
            'breakout', 'rally', 'surge', 'boom', 'explode', 'skyrocket', 'tendies',
            'lambos', 'rich', 'profit', 'winner', 'beat', 'earnings', 'positive',
            'strong', 'growth', 'recovery', 'upgrade', 'target', 'optimistic'
        }

        self.bearish_keywords = {
            'crash', 'dump', 'puts', 'bear', 'bearish', 'down', 'red', 'losses',
            'sell', 'selling', 'short', 'drop', 'fall', 'decline', 'collapse',
            'tank', 'drill', 'dead', 'rip', 'fucked', 'bag', 'holder', 'bagholding',
            'miss', 'negative', 'weak', 'recession', 'downgrade', 'pessimistic',
            'overvalued', 'bubble', 'correction', 'selloff'
        }

        # Initialize LLM for advanced sentiment
        self.use_llm = True
        try:
            self._initialize_llm()
        except Exception as e:
            logger.warning(f"LLM sentiment not available: {e}")
            self.use_llm = False

    def _initialize_llm(self):
        """Initialize LLM for advanced sentiment analysis"""
        provider = LLM_CONFIG['default_provider']

        if provider == 'xai':
            api_key = LLM_CONFIG['xai']['api_key']
            self.llm_client = openai.OpenAI(
                api_key=api_key,
                base_url=LLM_CONFIG['xai']['base_url']
            )
            self.llm_model = LLM_CONFIG['xai']['model']
        else:
            api_key = LLM_CONFIG['openai']['api_key']
            self.llm_client = openai.OpenAI(api_key=api_key)
            self.llm_model = LLM_CONFIG['openai']['model']

        logger.info(f"LLM sentiment analyzer initialized with {provider}")

    def analyze_vader_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)

        # Determine overall sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'bullish'
        elif compound <= -0.05:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'method': 'vader',
            'sentiment': sentiment,
            'confidence': abs(compound),
            'scores': scores,
            'compound': compound
        }

    def analyze_textblob_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment
        if polarity > 0.1:
            sentiment = 'bullish'
        elif polarity < -0.1:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'method': 'textblob',
            'sentiment': sentiment,
            'confidence': abs(polarity),
            'polarity': polarity,
            'subjectivity': subjectivity
        }

    def analyze_keyword_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using financial keywords"""
        text_lower = text.lower()

        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)

        total_keywords = bullish_count + bearish_count

        if total_keywords == 0:
            sentiment = 'neutral'
            confidence = 0.0
            score = 0.0
        else:
            score = (bullish_count - bearish_count) / total_keywords
            confidence = min(total_keywords / 10, 1.0)  # Max confidence at 10+ keywords

            if score > 0.2:
                sentiment = 'bullish'
            elif score < -0.2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'

        return {
            'method': 'keywords',
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count
        }

    def analyze_llm_sentiment(self, text: str, ticker: str = None) -> Dict:
        """Analyze sentiment using LLM for context-aware analysis - FIXED FOR GROK"""
        if not self.use_llm:
            return {'method': 'llm', 'sentiment': 'neutral', 'confidence': 0.0, 'error': 'LLM not available'}

        try:
            ticker_context = f" about {ticker}" if ticker else ""

            # GROK-SPECIFIC FIXES (same as ticker extractor)
            provider = LLM_CONFIG['default_provider']

            if provider == 'xai':
                # Key fix: Much higher token limit for Grok reasoning models
                max_tokens = 8000  # Increased from 300 to 8000
                temperature = 0.0  # Lower temperature

                # Simplified prompt to reduce reasoning overhead
                prompt = f"""Analyze sentiment of: "{text}"{ticker_context}

Respond with ONLY this JSON format:
{{
    "sentiment": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

                # FIXED: Only use parameters supported by Grok
                extra_params = {
                    'top_p': 0.1,  # More focused responses
                    'stop': ['\n\n', '```', 'Note:', 'Explanation:']  # Stop reasoning early
                    # REMOVED: frequency_penalty and presence_penalty - not supported by Grok
                }
            else:
                # OpenAI settings (unchanged)
                max_tokens = 300
                temperature = 0.1
                prompt = f"""Analyze the sentiment of this financial/trading text{ticker_context}:

"{text}"

Consider:
1. Trading context (calls/puts, bull/bear, buy/sell signals)
2. Financial keywords and slang
3. Emojis and expressions
4. Overall market sentiment

Respond with ONLY a JSON object:
{{
    "sentiment": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_phrases": ["list", "of", "important", "phrases"]
}}"""
                extra_params = {}

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": "You are a financial sentiment expert. Analyze trading sentiment accurately. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30,  # Longer timeout for reasoning models
                **extra_params
            )

            # Get raw response
            raw_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0

            # Enhanced debugging for Grok
            if provider == 'xai':
                logger.debug(f"Grok sentiment response - finish_reason: {finish_reason}, completion_tokens: {completion_tokens}")
                logger.debug(f"Raw Grok response: {repr(raw_content)}")

            if not raw_content or not raw_content.strip():
                logger.warning(f"Empty response from LLM sentiment - finish_reason: {finish_reason}, completion_tokens: {completion_tokens}")

                # If still hitting limits with Grok, try ultra-minimal approach
                if provider == 'xai' and finish_reason == 'length' and completion_tokens == 0:
                    logger.info("Attempting ultra-minimal Grok sentiment analysis...")
                    minimal_result = self._ultra_minimal_grok_sentiment(text, ticker)
                    return minimal_result

                return {'method': 'llm', 'sentiment': 'neutral', 'confidence': 0.0, 'error': 'Empty response'}

            # Clean the response - remove markdown and extra formatting
            content = raw_content.strip()

            # Remove markdown code blocks if they exist
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            elif content.startswith('```'):
                content = content[3:]  # Remove ```

            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```

            content = content.strip()

            # Find JSON object boundaries
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx + 1]
            else:
                logger.warning(f"No JSON object found in sentiment response: {repr(content)}")
                # Try to extract sentiment from text
                return self._extract_sentiment_from_text(raw_content)

            logger.debug(f"Cleaned content for JSON parsing: {repr(content)}")

            # Parse JSON
            try:
                result = json.loads(content)

                # Validate the result has required keys
                if not isinstance(result, dict):
                    logger.warning(f"LLM returned non-dict: {type(result)} - {result}")
                    return {'method': 'llm', 'sentiment': 'neutral', 'confidence': 0.0, 'error': 'Invalid response format'}

                # Ensure required keys exist with defaults
                result.setdefault('sentiment', 'neutral')
                result.setdefault('confidence', 0.0)
                result.setdefault('reasoning', 'No reasoning provided')
                result.setdefault('key_phrases', [])

                # Validate sentiment value
                if result['sentiment'] not in ['bullish', 'bearish', 'neutral']:
                    result['sentiment'] = 'neutral'

                # Validate confidence is a number between 0 and 1
                try:
                    confidence = float(result['confidence'])
                    result['confidence'] = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    result['confidence'] = 0.0

                result['method'] = 'llm'
                return result

            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON decode error for sentiment analysis: {json_error}")
                logger.warning(f"Content that failed to parse: {repr(content)}")

                # Fallback to text extraction
                return self._extract_sentiment_from_text(raw_content)

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}")
            return {
                'method': 'llm',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }

    def _ultra_minimal_grok_sentiment(self, text: str, ticker: str = None) -> Dict:
        """Ultra-minimal sentiment analysis when Grok consumes all tokens for reasoning"""
        try:
            # Absolute minimal prompt to bypass reasoning overhead
            simple_prompt = f"Sentiment of '{text}': bullish/bearish/neutral"

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": simple_prompt}],
                max_tokens=8000,  # Very high token limit
                temperature=0.0,
                top_p=0.05,  # Very focused
                stop=['\n', ' and ', ' because'],  # Stop early
                timeout=45
            )

            content = response.choices[0].message.content
            logger.info(f"Ultra-minimal Grok sentiment response: {repr(content)}")

            if content and content.strip():
                content_lower = content.lower()
                if 'bullish' in content_lower:
                    sentiment = 'bullish'
                    confidence = 0.6
                elif 'bearish' in content_lower:
                    sentiment = 'bearish'
                    confidence = 0.6
                else:
                    sentiment = 'neutral'
                    confidence = 0.5

                return {
                    'method': 'llm',
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'reasoning': f'Ultra-minimal analysis: {content[:50]}'
                }

        except Exception as e:
            logger.error(f"Ultra-minimal sentiment analysis failed: {e}")

        return {'method': 'llm', 'sentiment': 'neutral', 'confidence': 0.0, 'error': 'All methods failed'}

    def _extract_sentiment_from_text(self, text: str) -> Dict:
        """Extract sentiment from malformed LLM response"""
        try:
            text_lower = text.lower()

            # Look for sentiment keywords
            if any(word in text_lower for word in ['bullish', 'bull', 'positive', 'buy', 'calls', 'moon', 'up']):
                sentiment = 'bullish'
                confidence = 0.4
            elif any(word in text_lower for word in ['bearish', 'bear', 'negative', 'sell', 'puts', 'crash', 'down']):
                sentiment = 'bearish'
                confidence = 0.4
            else:
                sentiment = 'neutral'
                confidence = 0.3

            return {
                'method': 'llm',
                'sentiment': sentiment,
                'confidence': confidence,
                'reasoning': f'Extracted from malformed response: {text[:100]}',
                'error': 'JSON parsing failed, used text extraction'
            }

        except Exception as e:
            return {
                'method': 'llm',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': f'Text extraction failed: {str(e)}'
            }

    def analyze_comprehensive_sentiment(self, text: str, ticker: str = None) -> Dict:
        """Comprehensive sentiment analysis using all methods"""

        # Run all sentiment analysis methods
        vader_result = self.analyze_vader_sentiment(text)
        textblob_result = self.analyze_textblob_sentiment(text)
        keyword_result = self.analyze_keyword_sentiment(text)

        # LLM analysis (if available)
        llm_result = self.analyze_llm_sentiment(text, ticker) if len(text) > 50 else None

        # Combine results into weighted final sentiment
        methods = [vader_result, textblob_result, keyword_result]
        if llm_result and 'error' not in llm_result:
            methods.append(llm_result)

        # Calculate weighted sentiment
        weighted_sentiment = self._calculate_weighted_sentiment(methods)

        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'ticker': ticker,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'individual_methods': {
                'vader': vader_result,
                'textblob': textblob_result,
                'keywords': keyword_result,
                'llm': llm_result
            },
            'final_sentiment': weighted_sentiment,
            'analysis_quality': self._assess_analysis_quality(methods)
        }

    def _calculate_weighted_sentiment(self, methods: List[Dict]) -> Dict:
        """Calculate weighted final sentiment from multiple methods"""

        # Define weights for each method
        method_weights = {
            'vader': 0.25,  # Good for social media
            'textblob': 0.20,  # Good for general sentiment
            'keywords': 0.30,  # High weight for financial context
            'llm': 0.25  # High weight for context understanding
        }

        # Calculate weighted scores
        sentiment_scores = {'bullish': 0, 'neutral': 0, 'bearish': 0}
        total_confidence = 0

        for method in methods:
            if 'error' in method:
                continue

            method_name = method['method']
            weight = method_weights.get(method_name, 0.1)
            confidence = method.get('confidence', 0)
            sentiment = method.get('sentiment', 'neutral')

            # Add weighted vote
            sentiment_scores[sentiment] += weight * confidence
            total_confidence += confidence

        # Determine final sentiment
        if not sentiment_scores or total_confidence == 0:
            final_sentiment = 'neutral'
            final_confidence = 0.0
        else:
            final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            final_confidence = min(sentiment_scores[final_sentiment], 1.0)

        # Convert to numerical score (-1 to 1)
        sentiment_mapping = {'bearish': -1, 'neutral': 0, 'bullish': 1}
        numerical_score = sentiment_mapping[final_sentiment] * final_confidence

        return {
            'sentiment': final_sentiment,
            'confidence': round(final_confidence, 3),
            'numerical_score': round(numerical_score, 3),
            'method_breakdown': sentiment_scores,
            'total_methods': len([m for m in methods if 'error' not in m])
        }

    def _assess_analysis_quality(self, methods: List[Dict]) -> str:
        """Assess the quality of sentiment analysis"""
        successful_methods = len([m for m in methods if 'error' not in m])

        if successful_methods >= 4:
            return 'excellent'
        elif successful_methods >= 3:
            return 'good'
        elif successful_methods >= 2:
            return 'fair'
        else:
            return 'poor'

    def analyze_ticker_sentiment(self, posts_and_comments: List[Dict], ticker: str) -> Dict:
        """Analyze sentiment for a specific ticker across multiple posts/comments"""

        relevant_texts = []
        for item in posts_and_comments:
            if ticker.upper() in item.get('tickers', []):
                # Analyze the text
                text = item.get('text_for_analysis') or item.get('body') or item.get('title', '')
                if text and len(text.strip()) > 10:
                    sentiment_result = self.analyze_comprehensive_sentiment(text, ticker)
                    sentiment_result['post_id'] = item.get('id')
                    sentiment_result['score'] = item.get('score', 0)
                    sentiment_result['created_utc'] = item.get('created_utc', 0)
                    relevant_texts.append(sentiment_result)

        if not relevant_texts:
            return {
                'ticker': ticker,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'mention_count': 0,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }

        # Aggregate sentiment scores
        sentiment_scores = [item['final_sentiment']['numerical_score'] for item in relevant_texts]
        confidence_scores = [item['final_sentiment']['confidence'] for item in relevant_texts]

        # Weight by post score (higher scored posts have more influence)
        weighted_scores = []
        for item in relevant_texts:
            score_weight = max(1, item['score']) / 100  # Normalize post scores
            weighted_score = item['final_sentiment']['numerical_score'] * score_weight
            weighted_scores.append(weighted_score)

        # Calculate final aggregated sentiment
        avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0
        weighted_avg_sentiment = statistics.mean(weighted_scores) if weighted_scores else 0
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0

        # Determine final sentiment category
        if weighted_avg_sentiment > 0.1:
            final_sentiment = 'bullish'
        elif weighted_avg_sentiment < -0.1:
            final_sentiment = 'bearish'
        else:
            final_sentiment = 'neutral'

        return {
            'ticker': ticker,
            'sentiment': final_sentiment,
            'confidence': round(avg_confidence, 3),
            'numerical_score': round(weighted_avg_sentiment, 3),
            'mention_count': len(relevant_texts),
            'sentiment_distribution': {
                'bullish': len([s for s in sentiment_scores if s > 0.1]),
                'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1]),
                'bearish': len([s for s in sentiment_scores if s < -0.1])
            },
            'individual_analyses': relevant_texts,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }


def main():
    """Test sentiment analyzer"""
    analyzer = FinancialSentimentAnalyzer()

    test_texts = [
        "TSLA to the moon! 🚀🚀 Buying calls tomorrow",
        "AAPL looks terrible, buying puts. This is going to crash hard",
        "GME earnings next week, not sure what to expect",
        "Just made $50K on NVDA calls! Diamond hands paying off 💎🙌",
        "Lost everything on PLTR puts. This market is rigged 😭"
    ]

    print("🧠 Testing Financial Sentiment Analyzer")
    print("=" * 60)

    for text in test_texts:
        result = analyzer.analyze_comprehensive_sentiment(text)
        final = result['final_sentiment']

        print(f"Text: {text}")
        print(f"Sentiment: {final['sentiment']} (confidence: {final['confidence']})")
        print(f"Score: {final['numerical_score']}")
        print()


if __name__ == "__main__":
    main()