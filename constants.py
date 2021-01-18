"""
"""

invalid_relations = [
    'og', 'men', 'eller', 'så', 'fordi', 'når', 'før', 'dog',  # conjunctions
    'åh', 'wow', 'ouch', 'ah', 'oops',
    'hvad', 'hvordan', 'hvor', 'hvornår', 'hvem',
    'dem', 'han', 'hun', 'ham', 'hende', 'det',  # pronouns
    'ti', 'hundrede', 'tusind', 'millioner', 'milliarder',  # unit
    'en', 'to', 'tre', 'fire', 'fem', 'seks', 'syv', 'otte', 'ni',  # numbers
    'år', 'måned', 'dag', 'dagligt',
]

with open('data/adjectives.txt', 'r') as f:
    adjectives = [line.strip().lower() for line in f]

with open('data/adverbs.txt', 'r') as f:
    adverbs = [line.strip().lower() for line in f]

invalid_relations = invalid_relations + adjectives + adverbs

invalid_relations_set = set(invalid_relations)