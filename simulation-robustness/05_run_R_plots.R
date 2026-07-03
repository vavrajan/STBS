#!/usr/bin/env Rscript
# =============================================================================
# run_R_plots.R
# Generates all plots and tables for the STBS CAVI revision.
# Adapted from the original R scripts in STBS_CAVI/R/
# =============================================================================

library("colorspace")
library("readr")
library("ggplot2")
library("dplyr")

# ---- Paths ----
# Accept results_dir as command-line argument (for multi-seed runs)
# Usage: Rscript run_R_plots.R [results_dir]
args <- commandArgs(trailingOnly = TRUE)

STBS_DIR <- "/Users/paul.hofmarcher/Desktop/PolAn_Revision/STBS_CAVI"

if (length(args) >= 1) {
  results_dir <- args[1]
  cat(sprintf("Using results directory from CLI: %s\n", results_dir))
} else {
  BASE <- "/Users/paul.hofmarcher/Desktop/PolAn_Revision/Revision_code_CAVI"
  results_dir <- file.path(BASE, "stbs_cavi_results")
}

param_dir   <- file.path(results_dir, "params")
fig_dir     <- file.path(results_dir, "figs")
tab_dir     <- file.path(results_dir, "tabs")
clean_dir   <- file.path(STBS_DIR, "data/hein-daily/clean")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

K <- 25
joint_varfam <- TRUE  # iota_coef_jointly=True in Python => MVnormal with scale_tril

# =============================================================================
# Load data
# =============================================================================
cat("Loading parameters...\n")

iota_loc <- as.matrix(read_csv(file.path(param_dir, "iota_loc.csv"),
                                col_names = FALSE, show_col_types = FALSE))
ideal_loc <- as.matrix(read_csv(file.path(param_dir, "ideal_loc.csv"),
                                 col_names = FALSE, show_col_types = FALSE))
ideal_scl <- as.matrix(read_csv(file.path(param_dir, "ideal_scl.csv"),
                                 col_names = FALSE, show_col_types = FALSE))

if(joint_varfam){
  iota_scale_tril <- as.matrix(read_csv(file.path(param_dir, "iota_scale_tril.csv"),
                                         col_names = FALSE, show_col_types = FALSE))
  iota_var <- iota_scale_tril %*% t(iota_scale_tril)
}else{
  iota_scale <- as.matrix(read_csv(file.path(param_dir, "iota_scale.csv"),
                                    col_names = FALSE, show_col_types = FALSE))
  iota_var <- diag(c(iota_scale)^2)
}

L <- dim(iota_loc)[2]
cat(sprintf("  iota_loc: [%d x %d], ideal_loc: [%d x %d], L=%d\n",
            nrow(iota_loc), ncol(iota_loc), nrow(ideal_loc), ncol(ideal_loc), L))

# Author info
author_info <- read_csv(file.path(clean_dir, "author_detailed_info_with_religion114.csv"),
                         show_col_types = FALSE)
author_info$religion <- author_info$RELIGION
author_info <- as.matrix(author_info)

cat(sprintf("  Authors: %d\n", nrow(author_info)))

# =============================================================================
# Common functions and label definitions
# =============================================================================

### CCP-value function (chi-squared test on linear combinations)
CCPvalue <- function(C, mu, Sigma, mu0 = matrix(rep(0,length(mu)), nrow=1)){
  CSigmaC <- C %*% Sigma %*% t(C)
  Cdif <- C %*% t(mu - mu0)
  x <- solve(CSigmaC, Cdif)
  chi <- as.numeric(t(Cdif) %*% x)
  CCPval <- pchisq(chi, df=dim(C)[1], lower.tail = F)
  return(CCPval)
}

EstSECCP <- function(C, mu, Sigma, mu0 = matrix(rep(0,length(mu)), nrow=1)){
  CSigmaC <- t(C) %*% Sigma %*% C
  Cdif <- t(C) %*% t(mu - mu0)
  chi <- Cdif^2 / CSigmaC
  CCPval <- pchisq(chi, df=1, lower.tail = F)
  return(c(Cdif, sqrt(CSigmaC), CCPval))
}

CCPvalbreaks <- c(0, 0.001, 0.01, 0.05, 0.1, 1)
signif_codes <- c("***", "**", "*", ".", "")

specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))
formatCCPval <- function(x, k){
  if(x < 10^{-k}){
    return(paste0("<", 10^{-k}))
  }else{
    return(trimws(format(round(x, k), nsmall=k)))
  }
}

### Labels and category definitions
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

nicelabels_names_table <- c("party", "gender", "region", "generation", "experience", "religion")
names(nicelabels_names_table) <- names(labels)

nicelabels <- list()
nicelabels[["party"]] <- c("Democrats", "Republicans")
nicelabels[["gender"]] <- c("Male", "Female")
nicelabels[["region"]] <- c("NE", "MW", "SW", "S", "W")
nicelabels[["generation"]] <- c("Silent", "Boomer", "Gen X")
nicelabels[["exper_cong"]] <- c("Experienced", "Advanced", "Freshman")
nicelabels[["religion"]] <- c("Other", "Catholic", "Presbyterian", "Baptist",
                              "Jewish", "Methodist", "Lutheran", "Mormon")

# For the non-interaction plot_reg_coefs variant (3-party labels)
nicelabels_3party <- list()
nicelabels_3party[["party"]] <- c("Democratic", "Independent", "Republican")
nicelabels_3party[["gender"]] <- nicelabels[["gender"]]
nicelabels_3party[["region"]] <- c("Northeast", "Midwest", "Southeast", "South", "West")
nicelabels_3party[["generation"]] <- nicelabels[["generation"]]
nicelabels_3party[["exper_cong"]] <- nicelabels[["exper_cong"]]
nicelabels_3party[["religion"]] <- nicelabels[["religion"]]

values <- list()
values[["party"]] <- values[["gender"]] <- values[["region"]] <-
  values[["generation"]] <- values[["exper_cong"]] <- values[["religion"]] <- list()
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
                                     "Don't Know/Refused", "Buddhist")
values[["religion"]][["Catholic"]] <- c("Catholic")
values[["religion"]][["Presbyterian"]] <- c("Presbyterian")
values[["religion"]][["Baptist"]] <- c("Baptist")
values[["religion"]][["Jewish"]] <- c("Jewish")
values[["religion"]][["Methodist"]] <- c("Methodist")
values[["religion"]][["Lutheran"]] <- c("Lutheran")
values[["religion"]][["Mormon"]] <- c("Mormon")

# Covariate indices for the interaction model (L=51):
# 1: intercept (D baseline)
# 2: R main, 3: I main
# 4: Female
# 5-8: Midwest, Southeast, South, West
# 9-10: Boomers, Gen X
# 11-12: Exp (1,10], Exp (0,1]
# 13-19: Catholic, Presbyterian, Baptist, Jewish, Methodist, Lutheran, Mormon
# 20-35: R x (Female, Midwest, ..., Mormon)  => indices 4:19 + 16
# 36-51: I x (Female, Midwest, ..., Mormon)  => indices 4:19 + 32
indices <- list()
indices[["party"]] <- 1:2
indices[["gender"]] <- 4
indices[["region"]] <- 5:8
indices[["generation"]] <- 9:10
indices[["exper_cong"]] <- 11:12
indices[["religion"]] <- 13:19

### Linear combination constructor for interaction model
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

topic_labels <- c("National Security",
                  "Supreme Court",
                  "Coast Guard",
                  "Human Trafficking",
                  "Commemoration and Anniversaries",
                  "Gun Violence",
                  "Middle Class and Small Businesses",
                  "Health Care",
                  "Public Health (Zika)",
                  "Veterans and Health Care",
                  "Drugs and Addiction",
                  "Climate Change",
                  "Natural Resources",
                  "Planned Parenthood and Abortion",
                  "Institutes and Research",
                  "Middle East and Nuclear Weapons",
                  "Immigration and Department of Homeland Security",
                  "Social Security and Taxes",
                  "Rhetorics and Discussion",
                  "Clean Water Act",
                  "Law Enforcement",
                  "Wars and Human Rights",
                  "Education for Children",
                  "Cyber Security",
                  "Export, Import and Business")

# =============================================================================
# 1. INTERACTION PLOTS (from plot_reg_coefs_interactions.R)
# =============================================================================
cat("\n=== Creating interaction plots (D vs R) ===\n")

plot_regression_results_interactions <- function(iota_loc, iota_var, ideal_loc,
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

  ## Ideological positions
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
         col = histcol)
    text((plotbreaks[c(2,7,12,17,22)]+1.1)/2.2+ifelse(value=="D",-1.01,0.01),
         1.4,
         c("-1.0","-0.5", "0.0", "0.5", "1.0"),
         pos=1, cex=0.6)
  }

  ## Effect of the party
  mtext(nicelabels_names["party"], side=2, at=0.8, las=2, line=0.5, font=2)
  partyC = create_lin_komb()
  partyR = matrix(partyC[2,], nrow=1)
  partyR[1,1] = 0
  CCPval_party_effect = CCPvalue(partyR, mu=iota_loc, Sigma=iota_var)
  CCPval_party_effect_round = formatCCPval(CCPval_party_effect, 3)
  CCPval_party_effect_signif = cut(CCPval_party_effect, breaks=CCPvalbreaks, labels = signif_codes)
  mtext(paste0(CCPval_party_effect_round, " (", CCPval_party_effect_signif, ")"),
        side=2, at=0.3, las=2, line=0.3)

  CCPval_D = CCPvalue(matrix(partyC[1,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_D_signif = cut(CCPval_D, breaks=CCPvalbreaks, labels = signif_codes)
  CCPval_R = CCPvalue(matrix(partyC[2,], nrow=1), mu=iota_loc, Sigma=iota_var)
  CCPval_R_signif = cut(CCPval_R, breaks=CCPvalbreaks, labels = signif_codes)

  D_eff = as.numeric(iota_loc %*% partyC[1,])
  D_col = as.character(cut(D_eff, breaks=effectgrid, labels=effectcol))
  rect(-0.9,0.1,-0.1,0.9,col=D_col)
  text(-0.5, 0.5, CCPval_D_signif)

  R_eff = as.numeric(iota_loc %*% partyC[2,])
  R_col = as.character(cut(R_eff, breaks=effectgrid, labels=effectcol))
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
      # Only index columns that actually exist in the cross-tabulation
      vals_present <- intersect(values[[cat]][[j]], colnames(tab))
      if(length(vals_present) == 0){
        tab_join[,j] <- c(0, 0)
      } else if(length(vals_present) > 1){
        tab_join[,j] <- apply(tab[c("D", "R"), vals_present, drop=FALSE], 1, sum)
      } else {
        tab_join[,j] <- tab[c("D", "R"), vals_present]
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

      rect(xleft=xL, xright=xR,
           ybottom=-2*icat+0.9, ytop=-2*icat+1.9,
           col=as.character(cut(c(0,eff), breaks=effectgrid, labels=effectcol)))
      text(xmid[-1], -2*icat+1.4, CCPval_signif)
      text(xmid[1], -2*icat+1.2, paste0(formatCCPval(CCPvalC, 3)), cex=0.8)
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

  ## blue-red scale
  par(mar = c(0,0.5,2,2.5))
  plot(0,0, xlim=c(0,1), ylim=effect_lim, bty = "n", type="n",
       xaxt="n", yaxt="n", xlab="", ylab="", main="")
  axis(4, at=seq(effect_lim[1], effect_lim[2], by=0.2), las=2)
  rect(0,effectgrid[2:(Ncol+1)],1,effectgrid[3:(Ncol+2)],col=effectcol,border=NA)
}

# Topic-specific interaction plots
for(k in 1:K){
  cat(sprintf("  Topic %d/%d\n", k, K))

  pdf(file.path(fig_dir, sprintf("party_effects_interactions_k_%d.pdf", k-1)),
            width = 8, height = 7)
  plot_regression_results_interactions(matrix(iota_loc[k,], nrow = 1),
                                       iota_var,
                                       matrix(ideal_loc[,k], ncol = 1),
                                       save = "pdf")
  dev.off()

  png(file.path(fig_dir, sprintf("party_effects_interactions_k_%d.png", k-1)),
      width = 800, height = 700)
  plot_regression_results_interactions(matrix(iota_loc[k,], nrow = 1),
                                       iota_var,
                                       matrix(ideal_loc[,k], ncol = 1),
                                       save = "png")
  dev.off()
}


# =============================================================================
# 2. TABLE: Regression coefficients with interactions
#    (from table_reg_coefs_interactions.R)
# =============================================================================
cat("\n=== Creating regression coefficient tables ===\n")

table_regression_coefs <- function(topics){
  digits <- c(3, 3, 3, 3)
  nk <- length(topics)
  LTAB <- paste0("\\begin{tabular}{ll", paste(rep("|rrrr",nk), collapse = ""), "}\n")
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n\\toprule\n")
  LTAB <- paste0(LTAB, "\\multirow{2}{*}{Coefficient} & \\multirow{2}{*}{Category} ",
                 paste(paste0(" & \\multicolumn{4}{c}{Topic ", topics-1, "}"), collapse = ""),
                 "\\\\\n")
  LTAB <- paste0(LTAB, " & ",
                 paste(rep(" & Estimate & SE & CCP & CCP (all)", nk), collapse = ""),
                 "\\\\\n")
  LTAB <- paste0(LTAB, "\\midrule\n\\noalign{\\smallskip}\n")

  Ltab = 51
  C = diag(Ltab)
  l = 1
  # Intercept
  LTAB <- paste0(LTAB, "\\texttt{intercept} & ")
  for(k in topics){
    row <- EstSECCP(C[l,], mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
    LTAB <- paste0(LTAB,
                   " & $", specify_decimal(row[1], digits[1]),
                   "$ & $", specify_decimal(row[2], digits[2]),
                   "$ & $", formatCCPval(row[3], digits[3]),
                   "$ & ")
  }
  LTAB <- paste0(LTAB, "\\\\\n")
  # Party main effects
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n\\midrule\n\\noalign{\\smallskip}\n")
  LTAB <- paste0(LTAB, "\\multirow{2}{*}{\\texttt{party}}")
  for(p in c("Republican", "Independent")){
    l = l+1
    LTAB <- paste0(LTAB, " & \\texttt{",p,"}")
    for(k in topics){
      row <- EstSECCP(C[l,], mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
      LTAB <- paste0(LTAB,
                     " & $", specify_decimal(row[1], digits[1]),
                     "$ & $", specify_decimal(row[2], digits[2]),
                     "$ & $", formatCCPval(row[3], digits[3]),
                     "$ & ")
      if(p == "Republican"){
        ccpall <- CCPvalue(C[c(l,l+1),],
                           mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
        LTAB <- paste0(LTAB, "\\multirow{2}{*}{$",formatCCPval(ccpall, digits[4]),"$}")
      }
    }
    LTAB <- paste0(LTAB, "\\\\\n")
  }
  # Other main effects
  for(icat in 1:length(names(labels)[-1])){
    cat_name = names(labels)[icat+1]
    nlev = length(labels[[cat_name]])
    LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n\\midrule\n\\noalign{\\smallskip}\n")
    LTAB <- paste0(LTAB, "\\multirow{",nlev-1,"}{*}{\\texttt{",
                   nicelabels_names_table[cat_name],"}}")
    for(j in labels[[cat_name]][-1]){
      l = l+1
      LTAB <- paste0(LTAB, " & \\texttt{",j,"}")
      for(k in topics){
        row <- EstSECCP(C[l,], mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
        LTAB <- paste0(LTAB,
                       " & $", specify_decimal(row[1], digits[1]),
                       "$ & $", specify_decimal(row[2], digits[2]),
                       "$ & $", formatCCPval(row[3], digits[3]),
                       "$ & ")
        if((j == labels[[cat_name]][2]) & (nlev > 2)){
          ccpall <- CCPvalue(C[seq(l,l+nlev-2),],
                             mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
          LTAB <- paste0(LTAB, "\\multirow{",nlev-1,"}{*}{$",formatCCPval(ccpall, digits[4]),"$}")
        }
      }
      LTAB <- paste0(LTAB, "\\\\\n")
    }
  }
  # Interaction terms
  for(icat in 1:length(names(labels)[-1])){
    cat_name = names(labels)[icat+1]
    nlev = length(labels[[cat_name]])
    LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n\\midrule\n\\noalign{\\smallskip}\n")
    LTAB <- paste0(LTAB, "\\multirow{",nlev-1,"}{*}{\\texttt{party\\_Republican:",
                   nicelabels_names_table[cat_name],"}}")
    for(j in labels[[cat_name]][-1]){
      l = l+1
      LTAB <- paste0(LTAB, " & \\texttt{",j,"}")
      for(k in topics){
        row <- EstSECCP(C[l,], mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
        LTAB <- paste0(LTAB,
                       " & $", specify_decimal(row[1], digits[1]),
                       "$ & $", specify_decimal(row[2], digits[2]),
                       "$ & $", formatCCPval(row[3], digits[3]),
                       "$ & ")
        if((j == labels[[cat_name]][2]) & (nlev > 2)){
          ccpall <- CCPvalue(C[seq(l,l+nlev-2),],
                             mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
          LTAB <- paste0(LTAB, "\\multirow{",nlev-1,"}{*}{$",formatCCPval(ccpall, digits[4]),"$}")
        }
      }
      LTAB <- paste0(LTAB, "\\\\\n")
    }
  }
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n\\bottomrule\n\\noalign{\\medskip}\n")
  LTAB <- paste0(LTAB, "\\end{tabular}\n")
  return(LTAB)
}

# Per-topic tables
for(k in 1:K){
  LTAB <- table_regression_coefs(k)
  con <- file(file.path(tab_dir, sprintf("regression_coefs_k_%d.tex", k-1)),
              open = "wt", encoding = "UTF-8")
  sink(con); cat(LTAB); sink(); close(con)
}
cat("  25 per-topic LaTeX tables written.\n")

# Selected topics combined
topics <- c(5, 10, 12, 14, 17, 25)
LTAB <- table_regression_coefs(topics)
con <- file(file.path(tab_dir, sprintf("regression_coefs_k_%s.tex",
                                        paste(topics-1, collapse = "_"))),
            open = "wt", encoding = "UTF-8")
sink(con); cat(LTAB); sink(); close(con)
cat(sprintf("  Combined table for topics %s written.\n", paste(topics-1, collapse=", ")))


# =============================================================================
# 3. R-squared of party predicting ideal points (from ideal_party_R2.R)
# =============================================================================
cat("\n=== Creating R-squared analysis ===\n")

# Build ideal_data with party info
ideal_data <- data.frame(ideal_loc)
colnames(ideal_data) <- paste0("X", 0:(K-1))
ideal_data$avg <- rowMeans(ideal_loc)
ideal_data$party <- author_info[,"party"]

cols <- c(paste0("X", 0:(K-1)), "avg")
R2 <- R2adj <- numeric(length(cols))
names(R2) <- names(R2adj) <- cols

for(y in cols){
  fit <- lm(as.formula(paste0(y, " ~ factor(party)")), ideal_data)
  sumfit <- summary(fit)
  R2[y] <- sumfit$r.squared
  R2adj[y] <- sumfit$adj.r.squared
}

ideal_data_R <- data.frame(Topic = cols, R2 = R2, R2adj = R2adj)
write.csv(ideal_data_R, file.path(results_dir, "ideal_data_R2.csv"), row.names = FALSE)

# Bar plot of R2
r2_df <- data.frame(
  Topic = factor(0:(K-1)),
  R2 = R2[paste0("X", 0:(K-1))]
)
r2_plot <- ggplot(r2_df, aes(x = Topic, y = R2)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = R2["avg"], linetype = "dashed", color = "red") +
  annotate("text", x = K-1, y = R2["avg"] + 0.02,
           label = sprintf("avg R2 = %.3f", R2["avg"]), color = "red", hjust = 1) +
  theme_bw() +
  labs(y = expression(R^2), title = "R-squared: Party predicting ideal points per topic") +
  theme(axis.text.x = element_text(angle = 0, size = 7))
ggsave(file.path(fig_dir, "r2_party_per_topic.pdf"), r2_plot, width = 8, height = 4)
ggsave(file.path(fig_dir, "r2_party_per_topic.png"), r2_plot, width = 8, height = 4, dpi = 150)
cat("  R2 plot saved.\n")


# =============================================================================
# 4. Eta / Ideal point variability (from barplot_eta_ideal_variability.R)
# =============================================================================
cat("\n=== Creating eta/ideal variability barplot ===\n")

# Compute variability of ideal points per topic
ip_var_per_topic <- apply(ideal_loc, 2, var)
ip_df <- data.frame(
  Topic = factor(paste0(topic_labels, ifelse(0:(K-1) < 10, "     ", "   "), 0:(K-1)),
                 levels = paste0(topic_labels, ifelse(0:(K-1) < 10, "     ", "   "), 0:(K-1))),
  Variance = ip_var_per_topic
)

ip_var_plot <- ggplot(ip_df, aes(x = Topic, y = Variance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_bw() +
  labs(y = "Variance of ideal points", x = "") +
  ggtitle("Variability of topic-specific ideal points")
ggsave(file.path(fig_dir, "ideal_point_variability.pdf"), ip_var_plot, width = 8, height = 6)
ggsave(file.path(fig_dir, "ideal_point_variability.png"), ip_var_plot, width = 8, height = 6, dpi = 150)
cat("  Ideal point variability plot saved.\n")

# =============================================================================
# DONE
# =============================================================================
cat(sprintf("\n============================================================\n"))
cat(sprintf("All R plots and tables saved to:\n  Figs: %s\n  Tabs: %s\n", fig_dir, tab_dir))
cat(sprintf("============================================================\n"))
