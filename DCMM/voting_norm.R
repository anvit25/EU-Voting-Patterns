cc = read.csv('OneDrive - Harvard University/Courses/Stat 236/EU-Voting-Patterns/main_data/country_country_adj_1.csv',
              row.names = 1)


#diag(cc) = 0

# u = rowSums(cc)
# Z = diag(1/sqrt(u)) %*% as.matrix(cc) %*% diag(1/sqrt(u))
# 
# mu = mean(Z)
# sigma = sd(Z)
# Z = ifelse(Z>(mu - 0.19*sigma), 1, 0)
# rownames(Z) = rownames(cc)
# colnames(Z) = colnames(cc)
# Z = as.data.frame(Z)

############
# library(igraph)
# plist = seq(-1,1,by=0.01)
# for (p in plist) {
#   #print(p)
#   W = ifelse(Z>(mu+p*sigma), 1, 0)
#   graph <- graph_from_adjacency_matrix(as.matrix(W), mode = "undirected")
#   if(is_connected(graph)){
#     print(p)
#     #plot(graph, main = "Graph from Adjacency Matrix")
#   }
# }



############

# graph <- graph_from_adjacency_matrix(as.matrix(Z), mode = "undirected")
# 
# # Plot the graph
# plot(graph, main = "Graph from Adjacency Matrix")
############


dim(cc)
ms.out = mixedSCORE(cc, K = 4)
names(ms.out)

ms.out$L

##################################################

library(sf)
library(ggplot2)
library(gridExtra)
library(rnaturalearth)
library(rnaturalearthdata)

# Load Europe map data

world <- ne_download(scale = "small", type = "admin_0_countries_lakes",
                      returnclass = "sf")
#france_row = world[world$NAME_EN == "France", ]
world$ISO_A3[world$ADMIN == "France"] = "FRA"
filtered_countries <- world[world$ISO_A3 %in% colnames(cc), ]
#filtered_countries <- world[world$ISO_A3 == "FRA", ]

mship = ms.out$memberships
colnames(mship) = paste("Cluster", 1:ncol(mship), sep="_")
mship = as.data.frame(mship)
mship$country = colnames(cc)


filtered_countries <- merge(filtered_countries, mship,
                            by.x = "ISO_A3", by.y = "country", all.x = TRUE)


plot_list <- list()
for(i in 1:(ncol(mship)-1)){
  filtered_countries$color = numeric(length(filtered_countries$ISO_A3))
  cl = paste("Cluster", i, sep = "_")
  # print(i)
  for(j in 1:length(filtered_countries$Cluster_1)){
    #print(j)
    if(!is.na(filtered_countries[[cl]][j])){
      filtered_countries$color[j] = rgb(1, 0, 0, alpha = filtered_countries[[cl]][j]^2)
    }
    else{
      filtered_countries$color[j] = "transparent"
    }
  }
  p = ggplot(data = filtered_countries) +
    geom_sf(aes(fill = color), color = "black", size = 0.1) +
    coord_sf(xlim = c(-11,34), ylim = c(34,71), expand = FALSE) +
    scale_x_continuous(
      breaks = seq(-10, 30, 10),  # Set longitude breaks
      labels = seq(-10, 30, 10)   # Remove degree symbols
    ) +
    scale_y_continuous(
      breaks = seq(35, 70, 5),   # Set latitude breaks
      labels = seq(35, 70, 5)    # Remove degree symbols
    ) +
    scale_fill_identity() +
    theme_minimal() +
    # theme(
    #   aspect.ratio = 0.75  # Enforce square aspect ratio
    # ) +
    theme(
      aspect.ratio = 1  # Enforce square aspect ratio
    ) +
    labs(title = paste("Membership in Cluster", i),
         fill = "Value (Opacity)") +
    theme(plot.title = element_text(hjust = 0.5)) + 
    theme(
      plot.title = element_text(size = 18, face = "bold"),  # Title text size
      axis.title = element_text(size = 16),                # Axis label text size
      axis.text = element_text(size = 12)                  # Axis tick text size
    )
  
  plot_list[[i]] = p + theme(plot.margin = unit(c(0.1,0,0.1,0), "cm"))
}



grid.arrange(grobs = plot_list, ncol = 2,widths = c(1, 1),  # Relative column widths (equal spacing)
             heights = c(1, 1))




# plot(ms.out$R, col='grey', lwd = 2, xlab = 'R[,1]', ylab = 'R[.2]',bty="n")
# lines(ms.out$vertices[c(1,2,3,1),1], ms.out$vertices[c(1,2,3,1),2],
#       lty = 2, lwd = 2, col = 'black')
# points(ms.out$centers, lwd = 2, col = 'blue')
# 
# ###########
# #load('citee.RData')
# # dim(citee) [1] 1790 1790
# par(mfrow = c(1,4))
# for (i in 1:4){
#   
#   plot(ms.out$R, col=scales::alpha(i, ms.out$memberships[,i]^2), 
#        lwd = 2, bty="n", xlab = '',ylab = '')
#   lines(ms.out$vertices[c(1,2,3,4,1),1], ms.out$vertices[c(1,2,3,4,1),2], 
#         lty = 2, lwd = 2, col = 'black')
#   points(ms.out$centers, lwd = 2, col = 'blue')
# }
# 
# boxplot(ms.out$puritys ~as.factor(round(ms.out$degrees*5)/5), 
#         bty = 'n', xlab = 'degree', ylab = 'purity')
# 
# par(mfrow = c(1,1))


