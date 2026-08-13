import re

# Dictionary mapping common vocabulary blocks to determine switching criteria
MIZO_LEXICON = {"ka", "lawm", "e", "thil", "hi", "duh", "lo", "chu", "harsa", "tura", "hria", "an", "siam", "a", "tha", "lutuk", "vawiin", "nuam", "hle", "thleng", "tep", "dawr", "neitu", "pa", "kha", "sual", "muang", "khawp", "mai", "thianpa", "ti", "hian", "zel", "rawh"}
ENGLISH_LEXICON = {"for", "the", "beautiful", "birthday", "gift", "absolutely", "worst", "service", "but", "i", "failed", "anyway", "highly", "recommended", "guys", "weather", "match", "football", "ready", "in", "five", "minutes", "directly", "insulted", "me", "internet", "speed", "super", "frustrating", "great", "job"}

def detect_token_languages(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    tags = []
    
    for token in tokens:
        if token in MIZO_LEXICON and token in ENGLISH_LEXICON:
            tags.append("MIXED")
        elif token in MIZO_LEXICON:
            tags.append("MIZO")
        elif token in ENGLISH_LEXICON:
            tags.append("ENGLISH")
        else:
            tags.append("MIZO") # Default to localized low-resource class handling
            
    return " ".join(tags)
