pacman::p_load(ndjson, networkD3, tidyverse, visNetwork)

df <- read_csv("test.csv")

df <- df[1:100,]

# Make df of nodes with unique id for each node
nodes <- data.frame(label = unique(c(df$h, df$t)),
                    id = seq_len(length(unique(c(df$h, df$t)))),
                    value = 1)


# df of edges with from and to mapped to the id in nodes
edges <- tibble(from = df$h, to = df$t, labels=df$r) %>% 
  left_join(nodes, by=c("from"="label")) %>% 
  select(labels, to, id) %>% 
  select(labels, from=id, to) %>% 
  left_join(nodes, by = c("to"="label")) %>% 
  select(-to) %>% 
  select(from, to=id, label=labels) 


p <-  visNetwork(nodes, edges, width= 1200, height = 1200) %>% 
  #visIgraphLayout("layout_with_kk") %>% 
  visOptions(highlightNearest=list(enabled=T, degree=1), nodesIdSelection = T) %>% 
  visEdges(arrows="to") %>% 
  # Add psysics: more negative = more space between nodes
  visPhysics(solver = "forceAtlas2Based", forceAtlas2Based = list(gravitationalConstant=-100, damping = 2))

visSave(p, file = "network.html", background = "white")
