import belief_graph as bg

from .test_BeliefTriplet import simple_triplets


def test_POSFilter(simple_triplets):
    pos_filter = bg.filter.PosFilter(invalid = {"NUM"})
    
    print(pos_filter)
    
    triplets = pos_filter.filter(simple_triplets)
    triplets = list(triplets)
    assert len(triplets) == 1

def test_EntFilter(simple_triplets):
    ent_filter = bg.filter.EntFilter(applied_to = {"tail"})
    
    print(ent_filter)
    
    triplets = ent_filter.filter(simple_triplets)
    triplets = list(triplets)
    assert len(triplets) == 1

def test_ContinuousFilter(simple_triplets):
    ent_filter = bg.filter.EntFilter(applied_to = {"tail"})
    
    print(ent_filter)
    
    triplets = ent_filter.filter(simple_triplets)
    triplets = list(triplets)
    assert len(triplets) == 2

    triplet = bg.BeliefTriplet(head_id=0, tail_id=1, relation_id=[1, 3])
    triplet = next(ent_filter.filter(triplet))
    assert triplet is None

def test_CountFilter(simple_triplets):
    _ , triplet = simple_triplets

    triplet_ = bg.TripletGroup.from_belief_triplet(triplet)
    triplet_ = bg.add(triplet)

    c_filter = bg.filter.CountFilter(count=3)
    print(c_filter)

    filtered = next(c_filter.filter(triplet_))
    triplet_ = bg.add(triplet)
    assert filtered is None
    filtered = next(c_filter.filter(triplet_))
    assert isinstance(filtered, bg.BeliefTriplet)