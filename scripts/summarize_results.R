#!/usr/bin/env Rscript
# Stratified classification metrics and confusion cells from evaluation result CSV.
args<-commandArgs(trailingOnly=TRUE); opt<-function(flag,default){i<-match(flag,args);if(is.na(i)||i==length(args))default else args[i+1]}
input<-opt('--input',if(length(args))args[1] else 'evaluation_results.csv'); summary_out<-opt('--output','evaluation_summary.csv'); cm_out<-opt('--confusion-output','confusion_matrices.csv')
df<-read.csv(input,check.names=FALSE,stringsAsFactors=FALSE,fileEncoding='UTF-8-BOM')
canon<-function(x)gsub('[^a-z0-9]+','_',tolower(trimws(x)))
find<-function(opts){j<-match(canon(opts),canon(names(df)));j<-j[!is.na(j)];if(!length(j))stop(paste('Missing',paste(opts,collapse='/')));j[1]}
truthcol<-find(c('ground_truth','mapped_label','label')); predcol<-find(c('prediction','predicted_label'))
norm<-function(x){s<-tolower(trimws(as.character(x)));if(grepl('decept|lie|yalan|aldat',s))'Deceptive' else if(grepl('truth|doğru|dogru|honest',s))'Truthful' else stop(paste('Unknown label',x))}
y<-vapply(df[[truthcol]],norm,''); p<-vapply(df[[predcol]],norm,'')
metric<-function(ix,breakdown,condition){a<-y[ix];b<-p[ix];tp<-sum(a=='Deceptive'&b=='Deceptive');fp<-sum(a!='Deceptive'&b=='Deceptive');fn<-sum(a=='Deceptive'&b!='Deceptive');tn<-sum(a!='Deceptive'&b!='Deceptive');pr<-if(tp+fp)tp/(tp+fp)else 0;re<-if(tp+fn)tp/(tp+fn)else 0;data.frame(breakdown=breakdown,condition=as.character(condition),n=length(ix),correct=sum(a==b),accuracy=if(length(ix))mean(a==b)else 0,accuracy_pct=if(length(ix))100*mean(a==b)else 0,TP=tp,FP=fp,FN=fn,TN=tn,precision=pr,recall=re,f1=if(pr+re)2*pr*re/(pr+re)else 0)}
rows<-list(metric(seq_along(y),'overall','ALL')); cmrows<-list()
addcm<-function(ix,kind,cond){for(a in c('Truthful','Deceptive'))for(b in c('Truthful','Deceptive'))cmrows[[length(cmrows)+1]]<<-data.frame(breakdown=kind,condition=as.character(cond),actual=a,prediction=b,count=sum(y[ix]==a&p[ix]==b))}
addcm(seq_along(y),'overall','ALL')
for(field in c('label_type','source_column')) {j<-match(field,canon(names(df))); if(!is.na(j)){v<-as.character(df[[j]]);v[is.na(v)]<-'(missing)'; for(cond in unique(v)){ix<-which(v==cond);rows[[length(rows)+1]]<-metric(ix,field,cond);addcm(ix,field,cond)}}}
write.csv(do.call(rbind,rows),summary_out,row.names=FALSE);write.csv(do.call(rbind,cmrows),cm_out,row.names=FALSE)
cat(sprintf('Wrote %d summary strata and %d confusion cells\n',length(rows),length(cmrows)))
