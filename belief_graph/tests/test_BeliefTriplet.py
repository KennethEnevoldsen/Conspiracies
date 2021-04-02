"""
"""



from belief_graph import BeliefTriplet, load_danish

def test_BeliefTriplet():
    nlp = load_danish()
    doc = nlp("Dette består af to sætninger")
    s = next(doc.sents)

    BeliefTriplet(path=(0, 1, 2, 3),
                  span=s,
                  confidence=1)