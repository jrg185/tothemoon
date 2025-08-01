"""
LLM-Powered Ticker Extractor for Reddit Options App
Uses OpenAI to intelligently extract stock tickers from text
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import openai
import json
import re
import time
from typing import List, Dict, Optional
import logging
from config.settings import LLM_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMTickerExtractor:
    """Extract stock tickers using LLM intelligence"""

    def __init__(self):
        """Initialize LLM ticker extractor"""
        # Get LLM configuration
        self.provider = LLM_CONFIG['default_provider']

        if self.provider == 'xai':
            # XAI configuration
            api_key = LLM_CONFIG['xai']['api_key']
            self.base_url = LLM_CONFIG['xai']['base_url']
            self.model = LLM_CONFIG['xai']['model']

            if not api_key:
                raise ValueError("XAI API key not found in config")

            # XAI uses OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
            logger.info(f"Using XAI (Grok) model: {self.model}")

        else:  # Default to OpenAI
            # OpenAI configuration
            api_key = LLM_CONFIG['openai']['api_key']
            self.model = LLM_CONFIG['openai']['model']

            if not api_key:
                raise ValueError("OpenAI API key not found in config")

            if len(api_key) < 20:
                raise ValueError(f"API key appears invalid (too short): {api_key[:10]}...")

            if not api_key.startswith('sk-'):
                raise ValueError(f"OpenAI API key should start with 'sk-': {api_key[:10]}...")

            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"Using OpenAI model: {self.model}")

        self.request_count = 0
        self.cache = {}  # Simple cache to avoid repeated API calls

        logger.info(f"LLM extractor initialized with {self.provider}")
        logger.info(f"Model: {self.model}")

        # System prompt for ticker extraction
        self.system_prompt = """You are a financial expert specializing in stock ticker identification. Your job is to extract ONLY legitimate stock ticker symbols from text.

        RULES:
        1. Only return actual stock ticker symbols (like AAPL, TSLA, GME, SPY, QQQ)
        2. Do NOT return common English words, even if capitalized (like THE, AND, IS, ON, TO, etc.)
        3. Do NOT return financial terms that aren't tickers (like EPS, P/E, CEO, IPO, etc.)  
        4. Do NOT return abbreviations or acronyms that aren't stock tickers
        5. Include crypto tickers if mentioned (BTC, ETH, DOGE, etc.)
        6. Include ETF tickers (SPY, QQQ, IWM, etc.)
        7. Be conservative - when in doubt, don't include it

        CONTEXT: This text is from Reddit's wallstreetbets forum where people discuss stock trading.

        CRITICAL: Your entire response must be ONLY a valid JSON array. No explanations, no markdown formatting, no code blocks. Just the raw JSON array.

        EXAMPLES:
        - Good: ["AAPL", "TSLA", "GME"]  
        - Good: []
        - Bad: ```json["AAPL"]```
        - Bad: Here are the tickers: ["AAPL"]
        - Bad: The tickers I found are ["AAPL", "TSLA"]

        If no legitimate tickers are found, return exactly: []"""

    def extract_tickers_llm(self, text: str) -> List[str]:
        """Extract tickers using LLM - fixed for Grok reasoning models"""
        if not text or not text.strip():
            return []

        # Check cache first
        text_key = text[:100]
        if text_key in self.cache:
            return self.cache[text_key]

        try:
            # Rate limiting
            if self.request_count > 0:
                time.sleep(0.5)

            # GROK FIX: Much higher token limits and simplified prompts
            if self.provider == 'xai':
                # Key fix: MASSIVE token increase for reasoning models
                max_tokens = 8000  # Increased from 200 to 8000
                temperature = 0.0  # Lower for consistency

                # Simplified prompt to reduce reasoning overhead
                system_prompt = "Extract stock ticker symbols. Return JSON array only."
                user_prompt = f"Text: \"{text}\"\nTickers:"

            else:
                # OpenAI (unchanged)
                max_tokens = 200
                temperature = 0.1
                system_prompt = self.system_prompt + "\n\nIMPORTANT: Return ONLY a valid JSON array, no markdown formatting, no explanatory text."
                user_prompt = f"Extract stock tickers from this text:\n\n{text}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )

            self.request_count += 1

            # Get response
            raw_content = response.choices[0].message.content

            # Debug logging
            logger.debug(f"Raw LLM response: {repr(raw_content)}")

            if not raw_content or not raw_content.strip():
                raise ValueError(f"Empty response from {self.provider} LLM")

            # Clean response
            content = raw_content.strip()

            # Remove markdown if present
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Extract JSON array
            start_idx = content.find('[')
            end_idx = content.rfind(']')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx + 1]
            else:
                raise ValueError(f"No JSON array found in LLM response: {repr(content)}")

            # Parse JSON
            tickers = json.loads(content)

            if not isinstance(tickers, list):
                raise ValueError(f"LLM returned non-list: {type(tickers)} - {tickers}")

            # Clean and validate tickers
            clean_tickers = []
            for ticker in tickers:
                if isinstance(ticker, str) and ticker.strip():
                    clean_ticker = ticker.strip().upper()
                    if 1 <= len(clean_ticker) <= 6 and clean_ticker.isalpha():
                        if not self._is_common_word(clean_ticker):
                            clean_tickers.append(clean_ticker)

            # Cache and return
            self.cache[text_key] = clean_tickers
            logger.debug(f"Successfully extracted tickers: {clean_tickers}")
            return clean_tickers

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {self.provider} LLM: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM ticker extraction failed: {e}")

    def _is_common_word(self, word: str) -> bool:
        """Check if a word is a common English word that shouldn't be a ticker"""
        common_words = {
            # Common English words that might appear in caps
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'WAS', 'HIS', 'HER',
            'HAD', 'HAS', 'HAVE', 'WILL', 'BEEN', 'WERE', 'SAID', 'EACH', 'WHICH', 'THEIR',
            'TIME', 'WORK', 'LIFE', 'DAY', 'GET', 'USE', 'MAN', 'NEW', 'NOW', 'WAY', 'MAY',
            'SAY', 'COME', 'ITS', 'OVER', 'THINK', 'ALSO', 'BACK', 'AFTER', 'FIRST', 'WELL',
            'YEAR', 'GOOD', 'JUST', 'SEE', 'HOW', 'COULD', 'PEOPLE', 'TAKE', 'THAN', 'ONLY',
            'OTHER', 'TELL', 'WHAT', 'VERY', 'EVEN', 'THROUGH', 'ANY', 'WHERE', 'MUCH', 'THOSE',

            # Finance-related terms that aren't tickers
            'CEO', 'CFO', 'IPO', 'NYSE', 'NASDAQ', 'SEC', 'FDA', 'ATH', 'ATL', 'YTD', 'QTD',
            'PE', 'PEG', 'EPS', 'EBITDA', 'ROI', 'ROE', 'RSI', 'MACD', 'SMA', 'EMA', 'VWAP',
            'DD', 'YOLO', 'HODL', 'FUD', 'FOMO', 'WSB', 'APE', 'MOON', 'LAMBO', 'TENDIES',

            # Common reddit/social media terms
            'EDIT', 'TLDR', 'IMO', 'IMHO', 'LMAO', 'ROFL', 'WTF', 'OMG', 'LOL', 'BRB', 'AFK',
            'THIS', 'THAT', 'WHEN', 'THEN', 'HERE', 'THERE', 'BECAUSE', 'SINCE', 'UNTIL',
            'WOULD', 'SHOULD', 'COULD', 'MIGHT', 'MUST', 'SHALL', 'NEED', 'WANT', 'LIKE',
            'LOOK', 'FEEL', 'SEEM', 'BECOME', 'TURN', 'KEEP', 'STAY', 'REMAIN', 'APPEAR',

            # Days and months that might appear in caps
            'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY',
            'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER',
            'OCTOBER', 'NOVEMBER', 'DECEMBER', 'JAN', 'FEB', 'MAR', 'APR', 'JUN', 'JUL',
            'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'
        }

        return word.upper() in common_words

    def _extract_from_text_response(self, text: str) -> List[str]:
        """Extract tickers from text response when JSON parsing fails"""
        # Look for ticker-like patterns in the response
        ticker_pattern = re.compile(r'\b[A-Z]{2,6}\b')
        potential_tickers = ticker_pattern.findall(text.upper())

        # Filter out common words and validate
        clean_tickers = []
        for ticker in potential_tickers:
            if not self._is_common_word(ticker) and 1 <= len(ticker) <= 6:
                clean_tickers.append(ticker)

        # Remove duplicates and limit results
        return list(dict.fromkeys(clean_tickers))[:10]  # Preserve order, limit to 10

    def _fallback_extraction(self, text: str) -> List[str]:
        """Fallback extraction when LLM fails completely"""
        # Simple dollar sign extraction + known tickers as fallback
        tickers = set()

        # Dollar sign pattern
        dollar_tickers = self.dollar_pattern.findall(text.upper())
        tickers.update(dollar_tickers)

        # Check for known tickers in the text
        text_upper = text.upper()
        for ticker in self.known_tickers:
            if re.search(rf'\b{ticker}\b', text_upper):
                tickers.add(ticker)

        return sorted(list(tickers))[:5]  # Limit fallback results


class EnhancedTickerExtractor:
    """Hybrid ticker extractor using both LLM and regex approaches"""

    def __init__(self, use_llm: bool = True):
        """Initialize hybrid extractor"""
        self.use_llm = use_llm

        if use_llm:
            try:
                self.llm_extractor = LLMTickerExtractor()
                provider = LLM_CONFIG['default_provider']
                logger.info(f"LLM ticker extractor initialized with {provider}")
            except Exception as e:
                logger.warning(f"LLM extractor failed to initialize: {e}")
                logger.info("Falling back to regex-only extraction")
                self.use_llm = False

        # Fallback regex patterns
        self.dollar_pattern = re.compile(r'\$([A-Z]{2,6})\b')
        self.known_tickers = {
            'AAPL', 'GOOGL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC',
            'NFLX', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER', 'LYFT', 'ZOOM', 'ZM', 'ROKU',
            'SQ', 'TWTR', 'SNAP', 'PINS', 'SHOP', 'SPOT', 'TTD', 'CRWD', 'OKTA', 'DDOG',
            'SPY', 'QQQ', 'IWM', 'VIX', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'EEM', 'FXI',
            'GME', 'AMC', 'BB', 'NOK', 'SNDL', 'PLTR', 'NIO', 'XPEV', 'LI', 'BABA', 'JD',
            'DKNG', 'PENN', 'MGM', 'WYNN', 'LVS', 'CZR', 'HOOD', 'COIN', 'SOFI', 'UPST',
            'AFRM', 'LCID', 'RIVN', 'F', 'GM', 'RACE', 'RBLX', 'RDDT', 'WISH', 'CLOV'
        }

    def extract_tickers(self, text: str) -> List[str]:
        """Extract tickers using the best available method"""
        if not text:
            return []

        # Try LLM first
        if self.use_llm:
            try:
                llm_tickers = self.llm_extractor.extract_tickers_llm(text)
                if llm_tickers:  # If LLM found tickers, use them
                    return llm_tickers
            except Exception as e:
                logger.warning(f"LLM extraction failed, falling back to regex: {e}")

        # Fallback to regex + known tickers
        return self._regex_extraction(text)

    def _regex_extraction(self, text: str) -> List[str]:
        """Regex-based extraction as fallback"""
        text_upper = text.upper()
        tickers = set()

        # High confidence: dollar signs
        dollar_tickers = self.dollar_pattern.findall(text_upper)
        tickers.update(dollar_tickers)

        # Medium confidence: known tickers
        for ticker in self.known_tickers:
            if re.search(rf'\b{ticker}\b', text_upper):
                tickers.add(ticker)

        return sorted(list(tickers))


def test_llm_ticker_extraction():
    """Test the LLM-based ticker extraction"""
    extractor = EnhancedTickerExtractor(use_llm=True)

    test_cases = [
        # Should extract real tickers
        "I'm buying $TSLA calls and AAPL puts tomorrow",
        "GME squeeze IS THE way! AMC also good",
        "NVDA earnings beat, UP 5% after hours",
        "Apple $AAPL just reported earnings double beat",
        "Reddit RDDT crushes estimates, strong outlook",

        # Should NOT extract common words
        "THIS IS THE BEST stock for YOU TO BUY",
        "I THINK IT WILL GO UP TOMORROW",
        "DOES ANYONE KNOW WHAT TO DO HERE",
        "BOUGHT some stocks TODAY",

        # Mixed cases
        "FIG THIS OUT - AAPL looking good",
        "ON THE OTHER HAND, TSLA calls printing"
    ]

    print("🤖 Testing LLM-Powered Ticker Extraction")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"{i}. Text: \"{text}\"")

        try:
            tickers = extractor.extract_tickers(text)
            print(f"   LLM Result: {tickers}")
        except Exception as e:
            print(f"   Error: {e}")

        print()

    return extractor


def main():
    """Test the LLM ticker extractor"""
    print("Testing LLM-based ticker extraction...")
    test_llm_ticker_extraction()


if __name__ == "__main__":
    main()