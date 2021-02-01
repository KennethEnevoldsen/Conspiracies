# Example use from terminal:
# Rscript plot_network.R test.csv "plotting_test.html"
library(tidyverse)
library(visNetwork)
library(optparse)
library(webshot)
library(stringr)

option_list <- list(
  make_option(c("-f", "--file"), type="character", default=NULL,
                help="Filename of relations csv", metavar="character"),
  make_option(c("-n", "--name"), type="character", default=NULL,
              help="Filename to save to", metavar="character"),
  make_option(c("-e", "--edges"), type="integer", default=0,
              help="Number of edges to display (default all)", metavar="integer")
)

opt = parse_args(OptionParser(option_list=option_list))


df <- read_csv(opt$f)

# Subset if specified
if(opt$e > 0) df <- df[1:opt$e,]

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

# To html
visSave(p, file = opt$n, background = "white")
cat("\nHTML saved. Trying to save PNG ...\n")
# To PNG
webshot(opt$n, delay=1, zoom=3, 
        file=str_replace(opt$n, ".html", ".png"),
        vwidth=900,vheight=900)
cat("Done!\n")

