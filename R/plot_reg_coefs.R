library("colorspace")
ROOT <- "C:/Users/jvavra/PycharmProjects/STBS/"
data_name <- "hein-daily"

data_dir <- paste0(ROOT, "data/", data_name, "/") 
save_dir <- paste0(data_dir, "fits/")
clean_dir <- paste0(data_dir, "clean/")
fig_dir <- paste0(data_dir, "figs/")

ideal_dim <- "a"
covariates <- "all_no_int"
K <- 25
addendum <- 114

name <- paste0("STBS_ideal_",ideal_dim,"_",covariates,addendum,"_K",K)

param_dir <- paste0(save_dir, name, "/params/")
fig_dir <- paste0(fig_dir, name, "/")


### Loading the data
library(readr)
iota_loc <- as.matrix(read_csv(paste0(param_dir, "iota_loc.csv")))
if(joint_varfam){
  iota_scale_tril <- as.matrix(read_csv(paste0(param_dir, "iota_scale_tril.csv")))
  iota_var <- iota_scale_tril %*% t(iota_scale_tril)
}else{
  iota_scale <- as.matrix(read_csv(paste0(param_dir, "iota_scale.csv")))
  iota_var <- diag(c(iota_scale)^2)
}
ideal_loc <- as.matrix(read_csv(paste0(param_dir, "ideal_loc.csv")))
ideal_scl <- as.matrix(read_csv(paste0(param_dir, "ideal_scl.csv")))
L <- dim(iota_loc)[2]

# author info data
author_info <- read_csv(paste0(clean_dir, "author_detailed_info_with_religion114.csv"))
author_info$religion <- author_info$RELIGION
author_info <- as.matrix(author_info)

### Function to obtain p-values
VIpvalue <- function(C, mu, Sigma, mu0 = matrix(rep(0,length(mu)), nrow=1)){
  # C[Z,L] - matrix declaring linear combinations
  # mu[1,L] - vector of estimated means
  # Sigma[L,L] - variance matrix
  # mu0[1,L] - vector for testing hypothesis
  CSigmaC <- C %*% Sigma %*% t(C)
  Cdif <- C %*% t(mu - mu0)
  x <- solve(CSigmaC, Cdif)
  chi <- as.numeric(t(Cdif) %*% x)
  pval <- pchisq(chi, df=dim(C)[1], lower.tail = F)
  return(pval)
}

VIpvalue(matrix(c(1,0,1,1), nrow=2, byrow=T), matrix(rep(0.5,2), nrow=1), diag(2))

Pvalbreaks <- c(0, 0.001, 0.01, 0.05, 0.1, 1)
signif_codes <- c("***", "**", "*", ".", "")

cut(runif(1000), breaks=Pvalbreaks, labels = signif_codes)

### Function to create linear combinations
labels <- list()
labels[["party"]] <- c("Democrats", "Independent", "Republicans")
labels[["gender"]] <- c("Male", "Female")
labels[["region"]] <- c("Northeast", "Midwest", "Southeast", "South", "West")
labels[["generation"]] <- c("Silent", "Boomer", "Gen X")
labels[["exper_cong"]] <- c("(10,100]", "(1,10]", "(0,1]")
labels[["religion"]] <- c("Other", "Catholic", "Presbyterian", "Baptist",
                          "Jewish", "Methodist", "Lutheran", "Mormon")
nicelabels_names <- c("Party", "Gender", "Region", "Generation", "Experience", "Religion")
names(nicelabels_names) <- names(labels)

nicelabels <- list()
nicelabels[["party"]] <- c("Democratic", "Independent", "Republican")
nicelabels[["gender"]] <- c("Male", "Female")
#nicelabels[["region"]] <- c("NE", "MW", "SW", "S", "W")
nicelabels[["region"]] <- c("Northeast", "Midwest", "Southeast", "South", "West")
nicelabels[["generation"]] <- c("Silent", "Boomer", "Gen X")
nicelabels[["exper_cong"]] <- c("Experienced", "Advanced", "Freshman")
nicelabels[["religion"]] <- c("Other", "Catholic", "Presbyterian", "Baptist",
                              "Jewish", "Methodist", "Lutheran", "Mormon")

values <- list()
values[["party"]] <- values[["gender"]] <- values[["region"]] <- 
  values[["generation"]] <- values[["exper_cong"]] <- values[["religion"]] <-list()
values[["party"]][["Democrats"]] <- c("D")
values[["party"]][["Republicans"]] <- c("R")
values[["party"]][["Independent"]] <- c("I")
values[["gender"]][["Male"]] <- c("M")
values[["gender"]][["Female"]] <- c("F")
values[["region"]][["Northeast"]] <- c("Northeast") 
values[["region"]][["Midwest"]] <- c("Midwest")
values[["region"]][["Southeast"]] <- c("Southeast")
values[["region"]][["South"]] <- c("South")
values[["region"]][["West"]] <- c("West")
values[["generation"]][["Silent"]] <- c("Silent") 
values[["generation"]][["Boomer"]] <- c("Boomers")
values[["generation"]][["Gen X"]] <- c("Gen X")
values[["exper_cong"]][["(10,100]"]] <- c("(10, 100]")
values[["exper_cong"]][["(1,10]"]] <- c("(1, 10]")
values[["exper_cong"]][["(0,1]"]] <- c("(0, 1]") 
values[["religion"]][["Other"]] <- c("Congregationalist", "Anglican/Episcopal", 
                                     "Unspecified/Other (Protestant)",
                                     "Nondenominational Christian", 
                                     "Don’t Know/Refused", "Buddhist")  
values[["religion"]][["Catholic"]] <- c("Catholic") 
values[["religion"]][["Presbyterian"]] <- c("Presbyterian") 
values[["religion"]][["Baptist"]] <- c("Baptist") 
values[["religion"]][["Jewish"]] <- c("Jewish") 
values[["religion"]][["Methodist"]] <- c("Methodist") 
values[["religion"]][["Lutheran"]] <- c("Lutheran") 
values[["religion"]][["Mormon"]] <- c("Mormon") 

indices <- list()
indices[["party"]] <- 3:2
indices[["gender"]] <- 4
indices[["region"]] <- 5:8
indices[["generation"]] <- 9:10
indices[["exper_cong"]] <- 11:12
indices[["religion"]] <- 13:19

create_lin_komb <- function(category = "party", L=19){
  Z <- length(indices[[category]])
  C = matrix(0, nrow = Z, ncol = L)
  for(z in 1:Z){
    C[z, indices[[category]][z]] <- 1
  }
  return(C)
}
# try it 
Creg <- create_lin_komb(category = "region")
VIpvalue(Creg, iota_loc, iota_var)
VIpvalue(matrix(Creg[1,], nrow=1), iota_loc, iota_var)
VIpvalue(matrix(Creg[2,], nrow=1), iota_loc, iota_var)
VIpvalue(matrix(Creg[3,], nrow=1), iota_loc, iota_var)
VIpvalue(matrix(Creg[4,], nrow=1), iota_loc, iota_var)
VIpvalue(Creg, iota_loc, iota_var)
specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))
specify_decimal(0.007856, 3)
formatPval <- function(x, k){
  if(x < 10^{-k}){
    return(paste0("<", 10^{-k}))
  }else{
    return(trimws(format(round(x, k), nsmall=k)))
  }
}


### Function for the plot creation
plot_regression_results <- function(iota_loc, iota_var, ideal_loc, 
                                    ideal_lim=c(-1.2,1.2), 
                                    effect_lim = c(-0.8,0.8), 
                                    save = "pdf"){
  fontsize1 = switch(save, "pdf"=0.75, "png"=1)
  fontsize2 = switch(save, "pdf"=0.75, "png"=0.85)
  Ncol <- 1000
  idealgrid <- seq(ideal_lim[1], ideal_lim[2], length.out=Ncol+1) 
  idealcol <- diverge_hsv(Ncol)
  effectgrid <- c(-100, seq(effect_lim[1], effect_lim[2], length.out=Ncol+1), 100) 
  effectcol <- c("blue", diverge_hsv(Ncol), "red")
  
  layout(matrix(c(1,2), nrow = 1, byrow = TRUE), widths=c(8,1))
  par(mar = c(0.1,6.3,0.5,0))
  plot(0,0, xlim=c(0,1), ylim=c(-11.8,3), type="n",
       xaxt="n", yaxt="n", xlab="", ylab = "")
  
  ## Ideological positions - y in [1,4]
  mtext("Ideological", side=2, at=2, las=2, line=0.5, font=2)
  mtext("positions", side=2, at=1.6, las=2, line=0.5, font=2)
  #abline(h=0, col="grey", lty=2)
  for(party in labels[["party"]]){
    value = values[["party"]][[party]]
    ind = (author_info[,"party"] == value)
    ideals = ideal_loc[ind]
    histbreaks = c(-100, seq(ideal_lim[1], ideal_lim[2], by = 0.1), 100)
    plotbreaks = c(ideal_lim[1]-0.1, 
                   seq(ideal_lim[1], ideal_lim[2], by=0.1), 
                   ideal_lim[2]+0.1)
    nhist = length(histbreaks)
    fideal = cut(ideals, breaks=histbreaks)
    sumfideal = summary(fideal)
    rect(xleft=(plotbreaks[1:(nhist-1)]-ideal_lim[1]+0.1)/(ideal_lim[2]-ideal_lim[1]+0.2),
         ybottom=0.35,
         xright=(plotbreaks[2:nhist]-ideal_lim[1]+0.1)/(ideal_lim[2]-ideal_lim[1]+0.2),
         ytop=0.35+sumfideal/11*2.5,
         col = switch(party,
                      "Democrats"=rgb(0,0,1,0.5),
                      "Independent"=rgb(0,0,0,0.5),
                      "Republicans"=rgb(1,0,0,0.5))
    )
    
  }
  #segments(x0=0.5+ifelse(value=="D",-1,0), y0=1.3, y1=4.2, col="grey", lty=2)
  text((c(-1.0,-0.5,0.0,0.5,1.0)-ideal_lim[1]+0.1)/(ideal_lim[2]-ideal_lim[1]+0.2),
       0.4,
       c("-1.0","-0.5", "0.0", "0.5", "1.0"),
       pos=1, cex=0.6)
  legend("top", nicelabels[["party"]], ncol = 3, bty = "n",
         pt.bg = c(rgb(0,0,1,0.5), rgb(0,0,0,0.5), rgb(1,0,0,0.5)),
         pt.cex = 1.2,
         pch = 22, cex=0.85)
  
  ## Other covariates
  for(icat in 1:length(names(labels))){
    cat = names(labels)[icat]
    tab = table(author_info[,cat])
    tab_join = rep(0, length(labels[[cat]]))
    names(tab_join) = labels[[cat]]
    for(j in labels[[cat]]){
      tab_join[j] <- sum(tab[values[[cat]][[j]]])
    }
    mtext(nicelabels_names[cat], side=2, at=-2*icat+1.3, las=2, line=0.5, font=2)
    abline(h=-2*icat+2, col="grey", lty=2)
    catC = create_lin_komb(category=cat)
    
    pvalC = VIpvalue(catC, mu=iota_loc, Sigma=iota_var)
    pvalC_round = formatPval(pvalC, 3)
    pvalC_signif = cut(pvalC, breaks=Pvalbreaks, labels = signif_codes)
    mtext(paste0("All: ", pvalC_round, " (", pvalC_signif, ")"), 
          side=2, at=-2*icat+0.7, las=2, line=0.3)
    
    # rectangle measures
    Z = length(labels[[cat]])
    space = 0.09-Z*0.01
    len = (1-0.02-(Z-1)*space-Z*0.01)/sum(tab_join)
    xL = c(0.01+0:(Z-1)*space + cumsum(c(0,tab_join*len+0.01))[-(Z+1)])
    xR = c(0.01+ 0:(Z-1)*space + cumsum(tab_join*len+0.01))
    xmid = (xL+xR)/2
    
    pval = c(1,sapply(1:dim(catC)[1], function(z){VIpvalue(matrix(catC[z,], nrow=1),
                                                           mu=iota_loc, Sigma=iota_var)}))
    eff = c(0, catC %*% t(iota_loc))
    pval_signif = as.character(cut(pval, breaks=Pvalbreaks, labels = signif_codes))
    
    rect(xleft=xL,
         xright=xR,
         ybottom=-2*icat+0.9, ytop=-2*icat+1.9,
         col=as.character(cut(eff, breaks=effectgrid, labels=effectcol))
    )
    text(xmid, -2*icat+1.4, pval_signif)
    if(Z <= 3){
      label_add = 0.5
    }else{
      label_add = rep(c(0.2,0.6),Z)[1:Z]
    }
    text(xmid, -2*icat+label_add, 
         paste0(nicelabels[[cat]], " (", tab_join, ")"),
         cex = ifelse(Z>5,fontsize2,fontsize1))
    
  }
  
  ## blue-red scale on the right-hand side
  par(mar = c(0.1,0.5,0.5,2.5))
  plot(0,0, xlim=c(0,1), ylim=effect_lim, bty = "n", type="n",
       xaxt="n", yaxt="n", xlab="", ylab="",
       main="")
  axis(4, at=seq(effect_lim[1], effect_lim[2], by=0.2), las=2)
  rect(0,effectgrid[2:(Ncol+1)],1,effectgrid[3:(Ncol+2)],col=effectcol,border=NA)
  
}

cairo_pdf(paste0(fig_dir, "covariate_effects.pdf"),
          width = 7, height = 7)
{
  plot_regression_results(iota_loc, iota_var, ideal_loc, save="pdf")
}
dev.off()

png(paste0(fig_dir, "covariate_effects.png"),
    width = 700, height = 700)
{
  plot_regression_results(iota_loc, iota_var, ideal_loc, save="png")
}
dev.off()

### Transposed version for slides
plot_regression_results_transposed <- function(iota_loc, iota_var, ideal_loc, 
                                               ideal_lim=c(-1.2,1.2), 
                                               effect_lim = c(-0.8,0.8), 
                                               save = "pdf"){
  fontsize1 = switch(save, "pdf"=0.7, "png"=1)
  fontsize2 = switch(save, "pdf"=0.7, "png"=0.85)
  Ncol <- 1000
  idealgrid <- seq(ideal_lim[1], ideal_lim[2], length.out=Ncol+1) 
  idealcol <- diverge_hsv(Ncol)
  effectgrid <- c(-100, seq(effect_lim[1], effect_lim[2], length.out=Ncol+1), 100) 
  effectcol <- c("blue", diverge_hsv(Ncol), "red")
  
  layout(matrix(c(1,2), nrow = 1, byrow = TRUE), widths=c(14,1))
  par(mar = c(2.5,0.1,0.1,0))
  plot(0,0, ylim=c(-0.01,1.01), xlim=c(-3.15,12.1), type="n",
       xaxt="n", yaxt="n", xlab="", ylab = "",
       xaxs="i",yaxs="i")
  
  ## Ideological positions - y in [1,4]
  mtext("Ideological positions", side=1, at=-1.5, line=0.5, font=2)
  #abline(h=0, col="grey", lty=2)
  for(party in labels[["party"]]){
    value = values[["party"]][[party]]
    ind = (author_info[,"party"] == value)
    ideals = ideal_loc[ind]
    histbreaks = c(-100, seq(ideal_lim[1], ideal_lim[2], by = 0.1), 100)
    plotbreaks = c(ideal_lim[1]-0.1, 
                   seq(ideal_lim[1], ideal_lim[2], by=0.1), 
                   ideal_lim[2]+0.1)
    nhist = length(histbreaks)
    fideal = cut(ideals, breaks=histbreaks)
    sumfideal = summary(fideal)
    rect(xleft=3*(plotbreaks[1:(nhist-1)]-ideal_lim[1]+0.1)/(ideal_lim[2]-ideal_lim[1]+0.2)-3.05,
         ybottom=0.05,
         xright=3*(plotbreaks[2:nhist]-ideal_lim[1]+0.1)/(ideal_lim[2]-ideal_lim[1]+0.2)-3.05,
         ytop=0.05+sumfideal/12*0.85,
         col = switch(party,
                      "Democrats"=rgb(0,0,1,0.5),
                      "Independent"=rgb(0,0,0,0.5),
                      "Republicans"=rgb(1,0,0,0.5))
    )
    
  }
  #segments(x0=0.5+ifelse(value=="D",-1,0), y0=1.3, y1=4.2, col="grey", lty=2)
  text(3*(c(-1.0,-0.5,0.0,0.5,1.0)-ideal_lim[1]+0.1)/(ideal_lim[2]-ideal_lim[1]+0.2)-3.05,
       0.05,
       c("-1.0","-0.5", "0.0", "0.5", "1.0"),
       pos=1, cex=0.6)
  legend("topleft", nicelabels[["party"]], ncol = 1, bty = "n",
         pt.bg = c(rgb(0,0,1,0.5), rgb(0,0,0,0.5), rgb(1,0,0,0.5)),
         pt.cex = 1.2,
         pch = 22, cex=0.85)
  
  ## Other covariates
  for(icat in 1:length(names(labels))){
    cat = names(labels)[icat]
    tab = table(author_info[,cat])
    tab_join = rep(0, length(labels[[cat]]))
    names(tab_join) = labels[[cat]]
    for(j in labels[[cat]]){
      tab_join[j] <- sum(tab[values[[cat]][[j]]])
    }
    mtext(nicelabels_names[cat], side=1, at=2*(icat-1)+1.0, line=0.5, font=2)
    abline(v=2*(icat-1), col="grey", lty=2)
    catC = create_lin_komb(category=cat)
    
    pvalC = VIpvalue(catC, mu=iota_loc, Sigma=iota_var)
    pvalC_round = formatPval(pvalC, 3)
    pvalC_signif = cut(pvalC, breaks=Pvalbreaks, labels = signif_codes)
    mtext(paste0("All: ", pvalC_round, " (", pvalC_signif, ")"), 
          side=1, at=2*(icat-1)+1.0, line=1.3)
    
    # rectangle measures
    Z = length(labels[[cat]])
    space = 0.09-Z*0.01
    len = (1-0.02-(Z-1)*space-Z*0.01)/sum(tab_join)
    xL = c(0.01+0:(Z-1)*space + cumsum(c(0,tab_join*len+0.01))[-(Z+1)])
    xR = c(0.01+ 0:(Z-1)*space + cumsum(tab_join*len+0.01))
    xmid = (xL+xR)/2
    
    pval = c(1,sapply(1:dim(catC)[1], function(z){VIpvalue(matrix(catC[z,], nrow=1),
                                                           mu=iota_loc, Sigma=iota_var)}))
    eff = c(0, catC %*% t(iota_loc))
    pval_signif = as.character(cut(pval, breaks=Pvalbreaks, labels = signif_codes))
    
    rect(ybottom=xL,
         ytop=xR,
         xleft=2*(icat-1)+0.05, 
         xright=2*(icat-1)+0.65,
         col=as.character(cut(eff, breaks=effectgrid, labels=effectcol))
    )
    text(2*(icat-1)+0.35, xmid, pval_signif)
    text(2*(icat-1)+0.6, xmid, pos=4, 
         paste0(nicelabels[[cat]], " (", tab_join, ")"),
         cex = ifelse(Z>5,fontsize2,fontsize1))
    
  }
  
  ## blue-red scale on the right-hand side
  par(mar = c(2.5,0.5,0.1,2.5))
  plot(0,0, xlim=c(0,1), ylim=effect_lim, bty = "n", type="n",
       xaxt="n", yaxt="n", xlab="", ylab="",
       main="")
  axis(4, at=seq(effect_lim[1], effect_lim[2], by=0.2), las=2)
  rect(0,effectgrid[2:(Ncol+1)],1,effectgrid[3:(Ncol+2)],col=effectcol,border=NA)
  
}

cairo_pdf(paste0(fig_dir, "covariate_effects_transposed.pdf"),
          width = 11, height = 5)
{
  plot_regression_results_transposed(iota_loc, iota_var, ideal_loc, save="pdf")
}
dev.off()



# ### Application to topic-specific regressions
# fig_dir <- paste0(data_dir, "figs/")
# ideal_dim <- "ak"
# covariates <- "all_no_int"
# addendum <- 114
# K <- 25
# name <- paste0("STBS_ideal_",ideal_dim,"_",covariates,addendum,"_K",K)
# param_dir <- paste0(save_dir, name, "/params/")
# fig_dir <- paste0(fig_dir, name, "/")
# 
# # Loading the data
# iota_loc <- as.matrix(read_csv(paste0(param_dir, "iota_loc.csv")))
# if(joint_varfam){
#   iota_scale_tril <- as.matrix(read_csv(paste0(param_dir, "iota_scale_tril.csv")))
#   iota_var <- iota_scale_tril %*% t(iota_scale_tril)
# }else{
#   iota_scale <- as.matrix(read_csv(paste0(param_dir, "iota_scale.csv")))
#   iota_var <- diag(c(iota_scale)^2)
# }
# ideal_loc <- as.matrix(read_csv(paste0(param_dir, "ideal_loc.csv")))
# ideal_scl <- as.matrix(read_csv(paste0(param_dir, "ideal_scl.csv")))
# L <- dim(iota_loc)[2]
# 
# # Plots
# for(k in 1:K){
#   cairo_pdf(paste0(fig_dir, "covariate_effects_k_", k-1, ".pdf"),
#             width = 7, height = 7)
#   {
#     plot_regression_results(matrix(iota_loc[k,], nrow = 1),
#                             iota_var,
#                             matrix(ideal_loc[,k], ncol = 1),
#                             save = "pdf")
#   }
#   dev.off()
#   
#   cairo_pdf(paste0(fig_dir, "covariate_effects_transposed_k_", k-1, ".pdf"),
#             width = 11, height = 5)
#   {
#     plot_regression_results_transposed(matrix(iota_loc[k,], nrow = 1),
#                                        iota_var,
#                                        matrix(ideal_loc[,k], ncol = 1),
#                                        save = "pdf")
#   }
#   dev.off()
#   
#   png(paste0(fig_dir, "covariate_effects_k_", k-1, ".png"),
#       width = 700, height = 700)
#   {
#     plot_regression_results(matrix(iota_loc[k,], nrow = 1), 
#                             iota_var, 
#                             matrix(ideal_loc[,k], ncol = 1),
#                             save = "png")
#   }
#   dev.off()
# }
# 
# 
# 
