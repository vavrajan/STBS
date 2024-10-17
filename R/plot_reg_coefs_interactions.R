library("colorspace")
library(readr)
ROOT <- "C:/Users/jvavra/PycharmProjects/STBS/"
data_name <- "hein-daily"

data_dir <- paste0(ROOT, "data/", data_name, "/") 
save_dir <- paste0(data_dir, "fits/")
clean_dir <- paste0(data_dir, "clean/")
fig_dir <- paste0(data_dir, "figs/")

ideal_dim <- "a"
covariates <- "all"
ideal_varfam <- TRUE
K <- 25
addendum <- 114

name <- paste0("STBS_ideal_",ideal_dim,"_",covariates,addendum,"_K",K)

param_dir <- paste0(save_dir, name, "/params/")
fig_dir <- paste0(fig_dir, name, "/")
if(!dir.exists(fig_dir)){
  dir.create(fig_dir)
}

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
CCPvalue <- function(C, mu, Sigma, mu0 = matrix(rep(0,length(mu)), nrow=1)){
  # C[Z,L] - matrix declaring linear combinations
  # mu[1,L] - vector of estimated means
  # Sigma[L,L] - variance matrix
  # mu0[1,L] - vector for testing hypothesis
  CSigmaC <- C %*% Sigma %*% t(C)
  Cdif <- C %*% t(mu - mu0)
  x <- solve(CSigmaC, Cdif)
  chi <- as.numeric(t(Cdif) %*% x)
  CCPval <- pchisq(chi, df=dim(C)[1], lower.tail = F)
  return(CCPval)
}

CCPvalue(matrix(c(1,0,1,1), nrow=2, byrow=T), matrix(rep(0.5,2), nrow=1), diag(2))

CCPvalbreaks <- c(0, 0.001, 0.01, 0.05, 0.1, 1)
signif_codes <- c("***", "**", "*", ".", "")

cut(runif(1000), breaks=CCPvalbreaks, labels = signif_codes)

### Function to create linear combinations
labels <- list()
labels[["party"]] <- c("Democrats", "Republicans")
labels[["gender"]] <- c("Male", "Female")
labels[["region"]] <- c("Northeast", "Midwest", "Southeast", "South", "West")
labels[["generation"]] <- c("Silent", "Boomer", "Gen X")
labels[["exper_cong"]] <- c("(10,100]", "(1,10]", "(0,1]")
labels[["religion"]] <- c("Other", "Catholic", "Presbyterian", "Baptist",
                          "Jewish", "Methodist", "Lutheran", "Mormon")
nicelabels_names <- c("Party", "Gender", "Region", "Generation", "Experience", "Religion")
names(nicelabels_names) <- names(labels)

nicelabels <- list()
nicelabels[["party"]] <- c("Democrats", "Republicans")
nicelabels[["gender"]] <- c("Male", "Female")
nicelabels[["region"]] <- c("NE", "MW", "SW", "S", "W")
nicelabels[["generation"]] <- c("Silent", "Boomer", "Gen X")
nicelabels[["exper_cong"]] <- c("Experienced", "Advanced", "Freshman")
nicelabels[["religion"]] <- c("Other", "Catholic", "Presbyterian", "Baptist",
                              "Jewish", "Methodist", "Lutheran", "Mormon")

values <- list()
values[["party"]] <- values[["gender"]] <- values[["region"]] <- 
  values[["generation"]] <- values[["exper_cong"]] <- values[["religion"]] <-list()
values[["party"]][["Democrats"]] <- c("D")
values[["party"]][["Republicans"]] <- c("R")
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
indices[["party"]] <- 1:2
indices[["gender"]] <- 4
indices[["region"]] <- 5:8
indices[["generation"]] <- 9:10
indices[["exper_cong"]] <- 11:12
indices[["religion"]] <- 13:19

create_lin_komb <- function(party = "D", category = "party", L=51){
  if(category == "party"){
    C = matrix(0, nrow = 2, ncol = L)
    C[1,1] = 1
    C[2,1:2] = 1
  }else{
    Z <- length(indices[[category]])
    C = matrix(0, nrow = Z, ncol = L)
    for(z in 1:Z){
      C[z, indices[[category]][z]] <- 1
      if(party == "R"){
        C[z, indices[[category]][z]+16] <- 1
      }
      if(party == "I"){
        C[z, indices[[category]][z]+2*16] <- 1
      }
    }
  }
  return(C)
}
# # try it 
CD <- create_lin_komb(party = "D", category = "region")
CR <- create_lin_komb(party = "R", category = "region")
CCPvalue(CD, iota_loc, iota_var)
CCPvalue(matrix(CD[1,], nrow=1), iota_loc, iota_var)
CCPvalue(matrix(CD[2,], nrow=1), iota_loc, iota_var)
CCPvalue(matrix(CD[3,], nrow=1), iota_loc, iota_var)
CCPvalue(matrix(CD[4,], nrow=1), iota_loc, iota_var)
CCPvalue(CR, iota_loc, iota_var)
specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))
specify_decimal(0.007856, 3)
formatCCPval <- function(x, k){
  if(x < 10^{-k}){
    return(paste0("<", 10^{-k}))
  }else{
    return(trimws(format(round(x, k), nsmall=k)))
  }
}

### Function for the plot creation
plot_regression_results <- function(iota_loc, iota_var, ideal_loc, 
                                    ideal_lim=c(-1,1),
                                    effect_lim=c(-0.8,0.8),
                                    save = "pdf"){
  fontsize1 = switch(save, "pdf"=0.75, "png"=1)
  fontsize2 = switch(save, "pdf"=0.6, "png"=0.85)
  Ncol <- 1000
  idealgrid <- seq(ideal_lim[1], ideal_lim[2], length.out=Ncol+1) 
  idealcol <- diverge_hsv(Ncol)
  effectgrid <- c(-100, seq(effect_lim[1], effect_lim[2], length.out=Ncol+1), 100) 
  effectcol <- c("blue", diverge_hsv(Ncol), "red")
  
  layout(matrix(c(1,2), nrow = 1, byrow = TRUE), widths=c(9,1))
  par(mar = c(0.1,6.3,2,0))
  plot(0,0, xlim=c(-1,1), ylim=c(-9.8,4), type="n",
       xaxt="n", yaxt="n", xlab="", ylab = "")
  mtext("Democrats", side=3, line=0.3, at=-0.5, cex=2, font=2)
  mtext("Republicans", side=3, line=0.3, at=0.5, cex=2, font=2)
  abline(v=0, col="grey", lty=2)
  
  ## Ideological positions - y in [1,4]
  mtext("Ideological", side=2, at=3, las=2, line=0.5, font=2)
  mtext("positions", side=2, at=2.6, las=2, line=0.5, font=2)
  abline(h=1, col="grey", lty=2)
  for(party in labels[["party"]]){
    value = values[["party"]][[party]]
    ind = (author_info[,"party"] == value)
    ideals = ideal_loc[ind]
    histbreaks = c(-100, seq(ideal_lim[1], ideal_lim[2], length.out=21), 100)
    plotbreaks = c(ideal_lim[1]-0.1, 
                   seq(ideal_lim[1], ideal_lim[2], length.out=21), 
                   ideal_lim[2]+0.1)
    fideal = cut(ideals, breaks=histbreaks)
    sumfideal = summary(fideal)
    histcol = c("blue", diverge_hsv(20), "red")
    rect(xleft=(plotbreaks[1:22]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         ybottom=1.35,
         xright=(plotbreaks[2:23]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         ytop=1.35+sumfideal/11*2.7,
         col = histcol
    )
    #segments(x0=0.5+ifelse(value=="D",-1,0), y0=1.3, y1=4.2, col="grey", lty=2)
    text((plotbreaks[c(2,7,12,17,22)]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         1.4,
         c("-1.0","-0.5", "0.0", "0.5", "1.0"),
         pos=1, cex=0.6)
  }
  
  ## Effect of the party
  mtext(paste0(nicelabels_names["party"], 
               ""
               #" effect"
  ), side=2, at=0.8, las=2, line=0.5, font=2)
  # abline(h=0, col="grey", lty=2)
  partyC = create_lin_komb()
  partyR = matrix(partyC[2,], nrow=1)
  partyR[1,1] = 0
  CCPval_party_effect = CCPvalue(partyR, mu=iota_loc, Sigma=iota_var)
  CCPval_party_effect_round = formatCCPval(CCPval_party_effect, 3)
  CCPval_party_effect_signif = cut(CCPval_party_effect, breaks=CCPvalbreaks, labels = signif_codes)
  mtext(paste0(CCPval_party_effect_round, " (", CCPval_party_effect_signif, ")"), 
        side=2, at=0.3, las=2, line=0.3)
  
  CCPval_D = CCPvalue(matrix(partyC[1,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_D_round = formatCCPval(CCPval_D, 3)
  CCPval_D_signif = cut(CCPval_D, breaks=CCPvalbreaks, labels = signif_codes)
  
  CCPval_R = CCPvalue(matrix(partyC[2,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_R_round = formatCCPval(CCPval_R, 3)
  CCPval_R_signif = cut(CCPval_R, breaks=CCPvalbreaks, labels = signif_codes)
  
  D_eff = as.numeric(iota_loc %*% partyC[1,])
  D_col = as.character(cut(D_eff , breaks=effectgrid, labels=effectcol))
  rect(-0.9,0.1,-0.1,0.9,col=D_col)
  text(-0.5, 0.5, CCPval_D_signif)
  
  R_eff = as.numeric(iota_loc %*% partyC[2,])
  R_col = as.character(cut(R_eff , breaks=effectgrid, labels=effectcol))
  rect(0.1,0.1,0.9,0.9,col=R_col)
  text(0.5, 0.5, CCPval_R_signif)
  
  ## Other covariates
  for(icat in 1:length(names(labels)[-1])){
    cat = names(labels)[icat+1]
    tab = table(author_info[,"party"], author_info[,cat])
    tab_join = matrix(0, nrow=2, ncol=length(labels[[cat]]))
    rownames(tab_join) = c("D", "R")
    colnames(tab_join) = labels[[cat]]
    for(j in labels[[cat]]){
      if(length(values[[cat]][[j]]) > 1){
        tab_join[,j] <- apply(tab[c("D", "R"), values[[cat]][[j]]], 1, sum)
      }else{
        tab_join[,j] <- tab[c("D", "R"), values[[cat]][[j]]]
      }
    }
    mtext(nicelabels_names[cat], side=2, at=-2*icat+1.3, las=2, line=0.5, font=2)
    abline(h=-2*icat+2, col="grey", lty=2)
    catD = create_lin_komb(party="D", category=cat)
    catR = create_lin_komb(party="R", category=cat)
    catC = catR - catD
    
    CCPval_int = CCPvalue(catC, mu=iota_loc, Sigma=iota_var)
    CCPval_int_round = formatCCPval(CCPval_int, 3)
    CCPval_int_signif = cut(CCPval_int, breaks=CCPvalbreaks, labels = signif_codes)
    mtext(paste0("Int.: ", CCPval_int_round, " (", CCPval_int_signif, ")"), 
          side=2, at=-2*icat+0.7, las=2, line=0.3)
    
    # rectangle measures
    Z = length(labels[[cat]])
    first = 0.25
    space = 0.09-Z*0.01
    len = (1-first-(Z-1)*space)/(Z-1)
    xL = c(0.02, first + space + 0:(Z-2) * (space+len))
    xR = c(first, first + 1:(Z-1) * (space+len))
    xmid = (xL+xR)/2
    
    for(party in c("R", "D")){
      if(party == "D"){
        xL = xL-1.02
        xR = xR-1.02
        xmid = xmid-1.02
      }
      C = create_lin_komb(party=party, category=cat)
      CCPvalC = CCPvalue(C, mu=iota_loc, Sigma=iota_var)
      CCPval = sapply(1:dim(C)[1], function(z){CCPvalue(matrix(C[z,], nrow=1),
                                                      mu=iota_loc, Sigma=iota_var)})
      eff = C %*% t(iota_loc)
      CCPvalC_signif = as.character(cut(CCPvalC, breaks=CCPvalbreaks, labels = signif_codes))
      CCPval_signif = as.character(cut(CCPval, breaks=CCPvalbreaks, labels = signif_codes))
      
      rect(xleft=xL,
           xright=xR,
           ybottom=-2*icat+0.9, ytop=-2*icat+1.9,
           col=as.character(cut(c(0,eff), breaks=effectgrid, labels=effectcol))
      )
      text(xmid[-1], -2*icat+1.4, CCPval_signif)
      text(xmid[1], -2*icat+1.2, paste0(formatCCPval(CCPvalC, 3)),
           # " (", CCPvalC_signif, ")"), 
           cex=0.8)
      text(xmid[1], -2*icat+1.6, "All cat.", cex = 0.8)
      if(Z <= 5){
        label_add = 0.5
      }else{
        label_add = rep(c(0.2,0.6),Z)[1:Z]
      }
      text(xmid, -2*icat+label_add, 
           paste0(nicelabels[[cat]], " (", tab_join[party,], ")"),
           cex = ifelse(Z>5,fontsize2,fontsize1))
    }
    
  }
  
  ## blue-red scale on the right-hand side
  par(mar = c(0,0.5,2,2.5))
  plot(0,0, xlim=c(0,1), ylim=effect_lim, bty = "n", type="n",
       xaxt="n", yaxt="n", xlab="", ylab="",
       main="")
  axis(4, at=seq(effect_lim[1], effect_lim[2], by=0.2), las=2)
  rect(0,effectgrid[2:(Ncol+1)],1,effectgrid[3:(Ncol+2)],col=effectcol,border=NA)
  
}

### Transposed version - for slides
plot_regression_results_transposed <- function(iota_loc, iota_var, ideal_loc,
                                               ideal_lim=c(-1,1), 
                                               effect_lim=c(-0.8,0.8),
                                               max_hist_ylim=15,
                                               save = "pdf"){
  fontsize1 = switch(save, "pdf"=0.7, "png"=1)
  fontsize2 = switch(save, "pdf"=0.7, "png"=0.85)
  Ncol <- 1000
  idealgrid <- seq(ideal_lim[1], ideal_lim[2], length.out=Ncol+1) 
  idealcol <- diverge_hsv(Ncol)
  effectgrid <- c(-100, seq(effect_lim[1], effect_lim[2], length.out=Ncol+1), 100) 
  effectcol <- c("blue", diverge_hsv(Ncol), "red")
  
  layout(matrix(c(1,2), nrow = 1, byrow = TRUE), widths=c(14,1))
  par(mar = c(2.5,2,0.1,0))
  plot(0,0, ylim=c(-1.02,1.02), xlim=c(-4.2, 10.1), type="n",
       xaxt="n", yaxt="n", xlab="", ylab = "",
       xaxs="i",yaxs="i")
  mtext("Democrats", side=2, line=0.3, at=-0.5, cex=2, font=2)
  mtext("Republicans", side=2, line=0.3, at=0.5, cex=2, font=2)
  abline(h=0, col="grey", lty=2)
  
  ## Ideological positions - y in [1,4]
  mtext("Ideological positions", side=1, at=-2.8, line=0.5, font=2)
  abline(v=-1, col="grey", lty=2)
  for(party in labels[["party"]]){
    value = values[["party"]][[party]]
    ind = (author_info[,"party"] == value)
    ideals = ideal_loc[ind]
    histbreaks = c(-100, seq(ideal_lim[1], ideal_lim[2], length.out=21), 100)
    plotbreaks = c(ideal_lim[1]-0.1, 
                   seq(ideal_lim[1], ideal_lim[2], length.out=21), 
                   ideal_lim[2]+0.1)
    fideal = cut(ideals, breaks=histbreaks)
    sumfideal = summary(fideal)
    histcol = c("blue", diverge_hsv(20), "red")
    rect(xleft=3*(plotbreaks[1:22]+1.1)/2.2-4.15,
         ybottom=ifelse(value=="D",-0.9,0.1),
         xright=3*(plotbreaks[2:23]+1.1)/2.2-4.15,
         ytop=ifelse(value=="D",-0.9,0.1)+sumfideal/max_hist_ylim*0.9,
         col = histcol
    )
    # segments(x0=3*(plotbreaks[12]+1.1)/2.2-4.15, 
    #          y0=-0.9, y1=1.0, col="grey", lty=2)
    text(3*(plotbreaks[c(2,7,12,17,22)]+1.1)/2.2-4.15,
         ifelse(value=="D",-0.9,0.1),
         c("-1.0","-0.5", "0.0", "0.5", "1.0"),
         pos=1, cex=0.6)
  }
  
  ## Effect of the party
  mtext(paste0(nicelabels_names["party"], 
               ""
               #" effect"
  ), side=1, at=-0.5, line=0.5, font=2)
  # abline(h=0, col="grey", lty=2)
  partyC = create_lin_komb()
  partyR = matrix(partyC[2,], nrow=1)
  partyR[1,1] = 0
  CCPval_party_effect = CCPvalue(partyR, mu=iota_loc, Sigma=iota_var)
  CCPval_party_effect_round = formatCCPval(CCPval_party_effect, 3)
  CCPval_party_effect_signif = cut(CCPval_party_effect, breaks=CCPvalbreaks, labels = signif_codes)
  mtext(paste0(CCPval_party_effect_round, " (", CCPval_party_effect_signif, ")"), 
        side=1, at=-0.5, line=1.3)
  
  CCPval_D = CCPvalue(matrix(partyC[1,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_D_round = formatCCPval(CCPval_D, 3)
  CCPval_D_signif = cut(CCPval_D, breaks=CCPvalbreaks, labels = signif_codes)
  
  CCPval_R = CCPvalue(matrix(partyC[2,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_R_round = formatCCPval(CCPval_R, 3)
  CCPval_R_signif = cut(CCPval_R, breaks=CCPvalbreaks, labels = signif_codes)
  
  tab = table(author_info[,"party"])
  D_eff = as.numeric(iota_loc %*% partyC[1,])
  D_col = as.character(cut(D_eff , breaks=effectgrid, labels=effectcol))
  rect(-0.9,-0.01-0.85*tab["D"]/max(tab),-0.1,-0.01,col=D_col)
  text(-0.5, -0.93, paste0("(",tab["D"],")"), cex = fontsize1)
  text(-0.5, -0.5, CCPval_D_signif)
  
  R_eff = as.numeric(iota_loc %*% partyC[2,])
  R_col = as.character(cut(R_eff , breaks=effectgrid, labels=effectcol))
  rect(-0.9,0.99-0.85*tab["R"]/max(tab),-0.1,0.99,col=R_col)
  text(-0.5, 0.07, paste0("(",tab["R"],")"), cex = fontsize1)
  text(-0.5, 0.5, CCPval_R_signif)
  
  ## Other covariates
  for(icat in 1:length(names(labels)[-1])){
    cat = names(labels)[icat+1]
    tab = table(author_info[,"party"], author_info[,cat])
    tab_join = matrix(0, nrow=2, ncol=length(labels[[cat]]))
    rownames(tab_join) = c("D", "R")
    colnames(tab_join) = labels[[cat]]
    for(j in labels[[cat]]){
      if(length(values[[cat]][[j]]) > 1){
        tab_join[,j] <- apply(tab[c("D", "R"), values[[cat]][[j]]], 1, sum)
      }else{
        tab_join[,j] <- tab[c("D", "R"), values[[cat]][[j]]]
      }
    }
    mtext(nicelabels_names[cat], side=1, at=2*(icat-1)+1, line=0.5, font=2)
    abline(v=2*(icat-1), col="grey", lty=2)
    catD = create_lin_komb(party="D", category=cat)
    catR = create_lin_komb(party="R", category=cat)
    catC = catR - catD
    
    CCPval_int = CCPvalue(catC, mu=iota_loc, Sigma=iota_var)
    CCPval_int_round = formatCCPval(CCPval_int, 3)
    CCPval_int_signif = cut(CCPval_int, breaks=CCPvalbreaks, labels = signif_codes)
    mtext(paste0("Int.: ", CCPval_int_round, " (", CCPval_int_signif, ")"), 
          side=1, at=2*(icat-1)+1, line=1.3)
    
    # rectangle measures
    Z = length(labels[[cat]])
    first = 0.1
    space = 0.09-Z*0.01
    
    for(party in c("R", "D")){
      len = (1-0.02-first-Z*space-Z*0.01)/sum(tab_join[party,])
      xL = c(0.01, 0.01+first+ 1:Z*space + cumsum(c(0,tab_join[party,]*len+0.01))[-(Z+1)])
      xR = c(0.01+first, 0.01+first+ 1:Z*space + cumsum(tab_join[party,]*len+0.01))
      if(party == "D"){
        xL = xL-1
        xR = xR-1
      }
      xmid = (xL+xR)/2
      C = create_lin_komb(party=party, category=cat)
      CCPvalC = CCPvalue(C, mu=iota_loc, Sigma=iota_var)
      CCPval = sapply(1:dim(C)[1], function(z){CCPvalue(matrix(C[z,], nrow=1),
                                                      mu=iota_loc, Sigma=iota_var)})
      eff = C %*% t(iota_loc)
      CCPvalC_signif = as.character(cut(CCPvalC, breaks=CCPvalbreaks, labels = signif_codes))
      CCPval_signif = as.character(cut(CCPval, breaks=CCPvalbreaks, labels = signif_codes))
      
      rect(ybottom=xL,
           ytop=xR,
           xleft=2*(icat-1)+0.1, 
           xright=2*(icat-1)+0.7,
           col=as.character(cut(c(0,0,eff), breaks=effectgrid, labels=effectcol))
      )
      text(2*(icat-1)+0.4, xmid, c(CCPvalC_signif, "", CCPval_signif))
      text(2*(icat-1)+0.65, xmid, pos=4,
           c(paste0("All (", sum(tab_join[party,]), ")"),
             paste0(nicelabels[[cat]], " (", tab_join[party,], ")")),
           cex = ifelse(Z>5,fontsize2,fontsize1))
    }
    
  }
  
  ## blue-red scale on the right-hand side
  par(mar = c(2.5,0.5,0.1,2.4))
  plot(0,0, xlim=c(0,1), ylim=effect_lim, bty = "n", type="n",
       xaxt="n", yaxt="n", xlab="", ylab="",
       main="")
  axis(4, at=seq(effect_lim[1], effect_lim[2], by=0.2), las=2)
  rect(0,effectgrid[2:(Ncol+1)],1,effectgrid[3:(Ncol+2)],col=effectcol,border=NA)
  
}


cairo_pdf(paste0(fig_dir, "party_effects_interactions.pdf"),
          width = 8, height = 7)
{
  plot_regression_results(iota_loc, iota_var, ideal_loc, save="pdf")
}
dev.off()

png(paste0(fig_dir, "party_effects_interactions.png"),
    width = 800, height = 700)
{
  plot_regression_results(iota_loc, iota_var, ideal_loc, save="png")
}
dev.off()

cairo_pdf(paste0(fig_dir, "party_effects_interactions_transposed.pdf"),
          width = 11, height = 5)
{
  plot_regression_results_transposed(iota_loc, iota_var, ideal_loc, save="pdf")
}
dev.off()



### Application to topic-specific regressions
fig_dir <- paste0(data_dir, "figs/")
ideal_dim <- "ak"
covariates <- "all"
addendum <- 114
joint_varfam <- TRUE
K <- 25
name <- paste0("STBS_ideal_",ideal_dim,"_",covariates,addendum,"_K",K)
param_dir <- paste0(save_dir, name, "/params/")
fig_dir <- paste0(fig_dir, name, "/")
if(!dir.exists(fig_dir)){
  dir.create(fig_dir)
}

# Loading the data
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

# Plots
for(k in 1:K){
  cairo_pdf(paste0(fig_dir, "party_effects_interactions_k_", k-1, ".pdf"),
            width = 8, height = 7)
  {
    plot_regression_results(matrix(iota_loc[k,], nrow = 1),
                            iota_var,
                            matrix(ideal_loc[,k], ncol = 1),
                            save = "pdf")
  }
  dev.off()
  
  png(paste0(fig_dir, "party_effects_interactions_k_", k-1, ".png"),
      width = 800, height = 700)
  {
    plot_regression_results(matrix(iota_loc[k,], nrow = 1), 
                            iota_var, 
                            matrix(ideal_loc[,k], ncol = 1),
                            save = "png")
  }
  dev.off()
  
  cairo_pdf(paste0(fig_dir, "party_effects_interactions_transposed_k_", k-1, ".pdf"),
            width = 11, height = 5)
  {
    plot_regression_results_transposed(matrix(iota_loc[k,], nrow = 1),
                                       iota_var,
                                       matrix(ideal_loc[,k], ncol = 1),
                                       save = "pdf")
  }
  dev.off()
}


### wide, but not transposed
plot_regression_results_wide <- function(iota_loc, iota_var, ideal_loc, 
                                         ideal_lim=c(-1,1),
                                         effect_lim=c(-0.8,0.8),
                                         save = "pdf"){
  fontsize1 = switch(save, "pdf"=0.75, "png"=1)
  fontsize2 = switch(save, "pdf"=0.6, "png"=0.85)
  Ncol <- 1000
  idealgrid <- seq(ideal_lim[1], ideal_lim[2], length.out=Ncol+1) 
  idealcol <- diverge_hsv(Ncol)
  effectgrid <- c(-100, seq(effect_lim[1], effect_lim[2], length.out=Ncol+1), 100) 
  effectcol <- c("blue", diverge_hsv(Ncol), "red")
  
  layout(matrix(c(1,2), nrow = 1, byrow = TRUE), widths=c(14,1))
  par(mar = c(0.1,6.3,2,0))
  plot(0,0, xlim=c(-1,1), ylim=c(-9.8,4), type="n",
       xaxt="n", yaxt="n", xlab="", ylab = "")
  mtext("Democrats", side=3, line=0.3, at=-0.5, cex=2, font=2)
  mtext("Republicans", side=3, line=0.3, at=0.5, cex=2, font=2)
  abline(v=0, col="grey", lty=2)
  
  ## Ideological positions - y in [1,4]
  mtext("Ideological", side=2, at=3, las=2, line=0.5, font=2)
  mtext("positions", side=2, at=2.6, las=2, line=0.5, font=2)
  abline(h=1, col="grey", lty=2)
  for(party in labels[["party"]]){
    value = values[["party"]][[party]]
    ind = (author_info[,"party"] == value)
    ideals = ideal_loc[ind]
    histbreaks = c(-100, seq(ideal_lim[1], ideal_lim[2], length.out=21), 100)
    plotbreaks = c(ideal_lim[1]-0.1, 
                   seq(ideal_lim[1], ideal_lim[2], length.out=21), 
                   ideal_lim[2]+0.1)
    fideal = cut(ideals, breaks=histbreaks)
    sumfideal = summary(fideal)
    histcol = c("blue", diverge_hsv(20), "red")
    rect(xleft=(plotbreaks[1:22]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         ybottom=1.35,
         xright=(plotbreaks[2:23]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         ytop=1.35+sumfideal/11*2.7,
         col = histcol
    )
    #segments(x0=0.5+ifelse(value=="D",-1,0), y0=1.3, y1=4.2, col="grey", lty=2)
    text((plotbreaks[c(2,7,12,17,22)]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         1.4,
         c("-1.0","-0.5", "0.0", "0.5", "1.0"),
         pos=1, cex=0.6)
  }
  
  ## Effect of the party
  mtext(paste0(nicelabels_names["party"], 
               ""
               #" effect"
  ), side=2, at=0.8, las=2, line=0.5, font=2)
  # abline(h=0, col="grey", lty=2)
  partyC = create_lin_komb()
  partyR = matrix(partyC[2,], nrow=1)
  partyR[1,1] = 0
  CCPval_party_effect = CCPvalue(partyR, mu=iota_loc, Sigma=iota_var)
  CCPval_party_effect_round = formatCCPval(CCPval_party_effect, 3)
  CCPval_party_effect_signif = cut(CCPval_party_effect, breaks=CCPvalbreaks, labels = signif_codes)
  mtext(paste0(CCPval_party_effect_round, " (", CCPval_party_effect_signif, ")"), 
        side=2, at=0.3, las=2, line=0.3)
  
  CCPval_D = CCPvalue(matrix(partyC[1,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_D_round = formatCCPval(CCPval_D, 3)
  CCPval_D_signif = cut(CCPval_D, breaks=CCPvalbreaks, labels = signif_codes)
  
  CCPval_R = CCPvalue(matrix(partyC[2,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_R_round = formatCCPval(CCPval_R, 3)
  CCPval_R_signif = cut(CCPval_R, breaks=CCPvalbreaks, labels = signif_codes)
  
  D_eff = as.numeric(iota_loc %*% partyC[1,])
  D_col = as.character(cut(D_eff , breaks=effectgrid, labels=effectcol))
  rect(-0.9,0.1,-0.1,0.9,col=D_col)
  text(-0.5, 0.5, CCPval_D_signif)
  
  R_eff = as.numeric(iota_loc %*% partyC[2,])
  R_col = as.character(cut(R_eff , breaks=effectgrid, labels=effectcol))
  rect(0.1,0.1,0.9,0.9,col=R_col)
  text(0.5, 0.5, CCPval_R_signif)
  
  ## Other covariates
  for(icat in 1:length(names(labels)[-1])){
    cat = names(labels)[icat+1]
    tab = table(author_info[,"party"], author_info[,cat])
    tab_join = matrix(0, nrow=2, ncol=length(labels[[cat]]))
    rownames(tab_join) = c("D", "R")
    colnames(tab_join) = labels[[cat]]
    for(j in labels[[cat]]){
      if(length(values[[cat]][[j]]) > 1){
        tab_join[,j] <- apply(tab[c("D", "R"), values[[cat]][[j]]], 1, sum)
      }else{
        tab_join[,j] <- tab[c("D", "R"), values[[cat]][[j]]]
      }
    }
    mtext(nicelabels_names[cat], side=2, at=-2*icat+1.3, las=2, line=0.5, font=2)
    abline(h=-2*icat+2, col="grey", lty=2)
    catD = create_lin_komb(party="D", category=cat)
    catR = create_lin_komb(party="R", category=cat)
    catC = catR - catD
    
    CCPval_int = CCPvalue(catC, mu=iota_loc, Sigma=iota_var)
    CCPval_int_round = formatCCPval(CCPval_int, 3)
    CCPval_int_signif = cut(CCPval_int, breaks=CCPvalbreaks, labels = signif_codes)
    mtext(paste0("Int.: ", CCPval_int_round, " (", CCPval_int_signif, ")"), 
          side=2, at=-2*icat+0.7, las=2, line=0.3)
    
    # rectangle measures
    Z = length(labels[[cat]])
    first = 0.1
    space = 0.09-Z*0.01
    
    for(party in c("R", "D")){
      len = (1-0.02-first-Z*space-Z*0.01)/sum(tab_join[party,])
      xL = c(0.01, 0.01+first+ 1:Z*space + cumsum(c(0,tab_join[party,]*len+0.01))[-(Z+1)])
      xR = c(0.01+first, 0.01+first+ 1:Z*space + cumsum(tab_join[party,]*len+0.01))
      xmid = (xL+xR)/2
      if(party == "D"){
        xL = xL-1.02
        xR = xR-1.02
        xmid = xmid-1.02
      }
      C = create_lin_komb(party=party, category=cat)
      CCPvalC = CCPvalue(C, mu=iota_loc, Sigma=iota_var)
      CCPval = sapply(1:dim(C)[1], function(z){CCPvalue(matrix(C[z,], nrow=1),
                                                      mu=iota_loc, Sigma=iota_var)})
      eff = C %*% t(iota_loc)
      CCPvalC_signif = as.character(cut(CCPvalC, breaks=CCPvalbreaks, labels = signif_codes))
      CCPval_signif = as.character(cut(CCPval, breaks=CCPvalbreaks, labels = signif_codes))
      
      rect(xleft=xL,
           xright=xR,
           ybottom=-2*icat+0.9, ytop=-2*icat+1.9,
           col=as.character(cut(c(0,0,eff), breaks=effectgrid, labels=effectcol))
      )
      text(xmid, -2*icat+1.4, c(CCPvalC_signif, "", CCPval_signif))
      if(Z <= 5){
        label_add = 0.5
      }else{
        label_add = rep(c(0.2,0.6),Z)[1:Z]
      }
      text(xmid, -2*icat+label_add, 
           c(paste0("All (", sum(tab_join[party,]), ")"),
             paste0(nicelabels[[cat]], " (", tab_join[party,], ")")),
           cex = ifelse(Z>5,fontsize2,fontsize1))
    }
    
  }
  
  ## blue-red scale on the right-hand side
  par(mar = c(0,0.5,2,2.5))
  plot(0,0, xlim=c(0,1), ylim=effect_lim, bty = "n", type="n",
       xaxt="n", yaxt="n", xlab="", ylab="",
       main="")
  axis(4, at=seq(effect_lim[1], effect_lim[2], by=0.2), las=2)
  rect(0,effectgrid[2:(Ncol+1)],1,effectgrid[3:(Ncol+2)],col=effectcol,border=NA)
  
}

for(k in 1:K){
  cairo_pdf(paste0(fig_dir, "party_effects_interactions_wide_k_", k-1, ".pdf"),
            width = 11, height = 5)
  {
    plot_regression_results_wide(matrix(iota_loc[k,], nrow = 1),
                                 iota_var,
                                 matrix(ideal_loc[,k], ncol = 1),
                                 save = "pdf")
  }
  dev.off()
  
  png(paste0(fig_dir, "party_effects_interactions_wide_k_", k-1, ".png"),
      width = 1200, height = 700)
  {
    plot_regression_results_wide(matrix(iota_loc[k,], nrow = 1), 
                                 iota_var, 
                                 matrix(ideal_loc[,k], ncol = 1),
                                 save = "png")
  }
  dev.off()
}

