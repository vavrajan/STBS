ROOT <- "C:/Users/jvavra/PycharmProjects/STBS/"
data_name <- "hein-daily"

data_dir <- paste0(ROOT, "data/", data_name, "/") 
save_dir <- paste0(data_dir, "fits/")
clean_dir <- paste0(data_dir, "clean/")
fig_dir <- paste0(data_dir, "figs/")
tab_dir <- paste0(data_dir, "tabs/")

ideal_dim <- "a"
covariates <- "all_no_int"
K <- 25
joint_varfam <- TRUE
addendum <- 114

name <- paste0("STBS_ideal_",ideal_dim,"_",covariates,addendum,"_K",K)
param_dir <- paste0(save_dir, name, "/params/")
fig_dir <- paste0(fig_dir, name, "/")
tab_dir <- paste0(tab_dir, name, "/")

if(!dir.exists(tab_dir)){
  dir.create(tab_dir)
}

# Loading the data
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

### Function to obtain CCP-values
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

EstSECCP <- function(C, mu, Sigma, mu0 = matrix(rep(0,length(mu)), nrow=1)){
  # C[L] - matrix declaring linear combinations
  # mu[1,L] - vector of estimated means
  # Sigma[L,L] - variance matrix
  # mu0[1,L] - vector for testing hypothesis
  CSigmaC <- t(C) %*% Sigma %*% C
  Cdif <- t(C) %*% t(mu - mu0)
  chi <- Cdif^2 / CSigmaC
  CCPval <- pchisq(chi, df=1, lower.tail = F)
  return(c(Cdif, sqrt(CSigmaC), CCPval))
}

CCPvalue(matrix(c(1,0,1,1), nrow=2, byrow=T), matrix(rep(0.5,2), nrow=1), diag(2))
EstSECCP(c(1,1), matrix(rep(0.5,2), nrow=1), diag(2))

CCPvalbreaks <- c(0, 0.001, 0.01, 0.05, 0.1, 1)
signif_codes <- c("***", "**", "*", ".", "")

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
specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))
specify_decimal(0.007856, 3)

formatCCPval <- function(x, k){
  if(x < 10^{-k}){
    return(paste0("<", 10^{-k}))
  }else{
    return(trimws(format(round(x, k), nsmall=k)))
  }
}


### Table with regression coefficients - main effects + interaction with Republican party
table_regression_coefs <- function(topics){
  digits <- c(3, 3, 3, 3)
  nk <- length(topics)
  ## Header
  LTAB <- paste0("\\begin{tabular}{ll", paste(rep("|rrrr",nk), collapse = ""), "}\n")
  # LTAB <- paste0(LTAB, "\\renewcommand{\\arraystretch}{0.75}")
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n\\toprule\n")
  LTAB <- paste0(LTAB, "\\multirow{2}{*}{Coefficient} & \\multirow{2}{*}{Category} ",
                 paste(paste0(" & \\multicolumn{4}{c}{Topic ", topics-1, "}"), collapse = ""),
                 "\\\\\n")
  LTAB <- paste0(LTAB, " & ",
                 paste(rep(" & Estimate & SE & CCP & CCP (all)", nk), collapse = ""),
                 "\\\\\n")
  LTAB <- paste0(LTAB, "\\midrule\n")
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n")
  
  ## Body
  L = 19
  C = diag(L)
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
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n")
  LTAB <- paste0(LTAB, "\\midrule\n")
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n")
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
    cat = names(labels)[icat+1]
    nlev = length(labels[[cat]])
    LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n")
    LTAB <- paste0(LTAB, "\\midrule\n")
    LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n")
    LTAB <- paste0(LTAB, "\\multirow{",nlev-1,"}{*}{\\texttt{",
                   nicelabels_names_table[cat],"}}")
    for(j in labels[[cat]][-1]){
      l = l+1
      LTAB <- paste0(LTAB, " & \\texttt{",j,"}")
      for(k in topics){
        row <- EstSECCP(C[l,], mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
        LTAB <- paste0(LTAB, 
                       " & $", specify_decimal(row[1], digits[1]),
                       "$ & $", specify_decimal(row[2], digits[2]),
                       "$ & $", formatCCPval(row[3], digits[3]),
                       "$ & ")
        if((j == labels[[cat]][2]) & (nlev > 2)){
          ccpall <- CCPvalue(C[seq(l,l+nlev-2),],
                             mu=matrix(iota_loc[k,], nrow = 1), Sigma=iota_var)
          LTAB <- paste0(LTAB, "\\multirow{",nlev-1,"}{*}{$",formatCCPval(ccpall, digits[4]),"$}")
        }
      }
      LTAB <- paste0(LTAB, "\\\\\n")
    }
  }
  ## End
  LTAB <- paste0(LTAB, "\\noalign{\\smallskip}\n")
  LTAB <- paste0(LTAB, "\\bottomrule\n")
  LTAB <- paste0(LTAB, "\\noalign{\\medskip}\n")
  LTAB <- paste0(LTAB, "\\end{tabular}\n")
  
  return(LTAB)
}

topics <- c(1)
{
  LTAB <- table_regression_coefs(topics)
  con <- file(paste0(tab_dir , "regression_coefs.tex"), 
              open = "wt", encoding = "UTF-8")
  sink(con)
  cat(LTAB)
  sink()
  close(con)
}
