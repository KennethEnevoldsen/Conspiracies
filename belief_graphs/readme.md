
# Design
Uses pydantic for type checks

# Structure


## Belief extract
-  init with params such as number of beams
-  takes a sentence with tags (noun chunks etc.)
-  extract triplets pr. sentence
   -  return a list of BeliefTriplets

## BeliefTriplet
- contains
  - triplets
  - conf
  - Nc converter (Noun Chunk to everything)

## Belief Graphs
- init
  - takes a belief extraction function which return belief triplets
- method for adding singular sentences (pass to belief extraction)
- method for adding multiple sentences
- (Method for unlisting belief triplets) 
- method for filtering belief graphs
  - custom threshold etc.
- plotting
- app'en