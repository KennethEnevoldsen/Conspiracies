import belief_graph as bg

from .test_BeliefTriplet import simple_triplets


def test_PosFilter(simple_triplets):
    pos_filter = bg.filters.PosFilter(invalid={"NUM"})

    print(pos_filter)

    triplets = pos_filter.filter(simple_triplets)
    triplets = list(triplets)
    assert len(triplets) == 1


def test_EntFilter(simple_triplets):
    s1 = simple_triplets[0]
    s2 = simple_triplets[1]
    ent_filter = bg.filters.EntFilter(apply_to={"tail"})

    print(ent_filter)

    triplets = ent_filter.filter(simple_triplets)
    triplets = list(triplets)
    assert len(triplets) == 1


def test_ContinuousFilter(simple_triplets):
    ent_filter = bg.filters.ContinuousFilter()

    print(ent_filter)

    triplets = ent_filter.filter(simple_triplets)
    triplets = list(triplets)
    assert len(triplets) == 2

    triplet = bg.BeliefTriplet(head_id=0, tail_id=1, relation_ids=[1, 3], confidence=1)
    triplets = list(ent_filter.filter([triplet]))
    assert len(triplets) == 0


def test_CountFilter(simple_triplets):
    _, triplet = simple_triplets

    tg = bg.TripletGroup.from_belief_triplet(triplet)
    tg.add(triplet)

    c_filter = bg.filters.CountFilter(count=3)
    print(c_filter)

    filtered = list(c_filter.filter([tg]))
    tg.add(triplet)
    assert len(filtered) == 0
    filtered = next(c_filter.filter([tg]))
    assert isinstance(filtered, bg.TripletGroup)
