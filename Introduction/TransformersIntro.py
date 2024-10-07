from transformers import pipeline


## Sentiment Analysis
# classifier = pipeline("sentiment-analysis", model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
# result = classifier(
#     ["I've been waiting for a HuggingFace course my whole life.",
#      "I hate this so much!"])
# print(result)

## Zero-Shot classification
classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(result)
print(classifier.framework)

## Text generation
# generator = pipeline("text-generation", model = "distilbert/distilgpt2")
# result = generator(
#     "Some of the cities you can find in Azeroth include")
# print(result)

## Mask filling
# unmasker = pipeline("fill-mask", model = 'bert-base-cased')
# result = unmasker("Arthas is a character in [MASK] of Warcraft.")
# for element in result:
#     print(element)

## Named entity recognition
# ner = pipeline("ner",model = "FacebookAI/xlm-roberta-large-finetuned-conll03-english", grouped_entities=True)
# result = ner("My name is Hidetaka and I work at From Software in Tokyo, Japan. My Cousin Alberto is stuck in Bogotá")
# for element in result:
#     print(element)

## Question answering
# question_answerer = pipeline("question-answering", model = "distilbert/distilbert-base-cased-distilled-squad")
# result = question_answerer(
#     question="How many siblings do I have?",
#     context="I have five uncles. I have three brothers and no sisters. I have eight sons and four daughters.")
# print(result)

## Summarization
# summarizer = pipeline("summarization", model = "sshleifer/distilbart-cnn-12-6")
# result = summarizer(
#   """
    # Could Hollow Knight Silksong be the spooky season treat we've all been waiting for since 2019?
    # Team Cherry's long-awaited Metroidvania has been in the works for some fives years now, and despite the long wait,
    # fans are running on little more than crumbs when it comes to intel. In fact, one of the most concrete facts we know
    # about Hollow Knight Silksong is that it will be on Xbox Game Pass on day one.
    #
    # Seven years since the first game launched, and despite the developer keeping things under wraps, the hype around
    # Hollow Knight: Silksong seems to just keep growing - even though we have precious little confirmed. The long wait has
    # made some fans antsy, getting the clown facepaint on at the drop of a hat after a hidden Hollow Knight Silksong update
    # on its Steam page made hopeful hearts soar earlier this year. A 2023 controversy surrounding engine Unity's intended
    # pay-per-download business model had made other Hollow Knight fans panic, with speculations that an engine swap could
    # incur further delays on top of  Silksong's initial postponement that pushed its launch date from early last year to
    # sometime in the as-yet unknown future. Frustrating though the lack of information is, Team Cherry has reassured its
    # restless players that Silksong is still very much in the works. Sounds like we will just have to keep being patient.
    #
    # And the wait will hopefully be worth it. Hollow Knight Silksong stills ranks high on our list of new games of 2024,
    # with or without a firm release date.
    # """
# )
# print(result)

## Translation
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-es")
# result = translator("""
#     Allons enfant de la patrie,
#     Le jour de gloire est arrivé!
#     Contre nous de la tyrannie,
#     L'étendard sanglant est levé
#     L'étendard sanglant est levé
#     Entendez-vous dans les campagnes
#     Mugir ces féroces soldats?
#     Ils viennent jusque dans vos bras
#     Égorger vos fils et vos compagnes!
#     Aux armes, citoyens! (Formez)
#     Vos bataillons!
#     Marchons! Oui, marchons!
#     Qu'un sang impur
#     Abreuve nos sillons!
#     Sillons!
#     Ouh ouh
#     """)
# print(result)
