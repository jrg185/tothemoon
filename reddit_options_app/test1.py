from data.llm_ticker_extractor import LLMTickerExtractor


def test_fixed_grok():
    extractor = LLMTickerExtractor()

    test_text = "I bought $AAPL calls and TSLA puts"
    result = extractor.extract_tickers_llm(test_text)
    print(f"Result: {result}")


if __name__ == "__main__":
    test_fixed_grok()