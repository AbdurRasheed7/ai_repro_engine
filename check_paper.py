from agents.domain_detector import detect_domain
from agents.parser_agent import parse_paper

text = parse_paper('1807.01622')
text_lower = text.lower()

nlp_triggers = [
    'sentiment analysis', 'text classification',
    'named entity recognition', 'machine translation',
    'question answering', 'language model', 'bert',
    'transformer model', 'word embedding', 'tokenization',
    'natural language processing', 'text generation'
]

found = [t for t in nlp_triggers if t in text_lower]
print('NLP triggers found:', found)