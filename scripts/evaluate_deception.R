#!/usr/bin/env Rscript
# Same normalized-label, zero/few-shot, simulation/API evaluation flow as Python.
# Base R only for simulation/CSV; real API requires jsonlite (and curl through httr).
args <- commandArgs(trailingOnly=TRUE)
getopt <- function(flag, default=NULL) { i <- match(flag,args); if(is.na(i)||i==length(args)) default else args[i+1] }
has <- function(flag) flag %in% args
input <- getopt('--input'); prompts <- getopt('--prompts'); output <- getopt('--output','evaluation_results.csv'); summary_file <- getopt('--summary','evaluation_summary.json')
real <- has('--real'); template <- getopt('--template','zero_shot'); limit <- as.integer(getopt('--limit','Inf'))
model <- getopt('--model',Sys.getenv('LLM_MODEL','gpt-4o-mini')); endpoint <- getopt('--endpoint',Sys.getenv('LLM_BASE_URL','https://api.openai.com/v1/chat/completions')); key <- getopt('--api-key',Sys.getenv('OPENAI_API_KEY'))
if(is.null(input)&&is.null(prompts)) stop('Use --input CSV or --prompts key:value text file')
norm <- function(x) { s<-tolower(trimws(as.character(x))); if(grepl('decept|lie|yalan|aldat|^false$',s)) 'Deceptive' else if(grepl('truth|doğru|dogru|honest|^true$',s)) 'Truthful' else stop(paste('Unknown label:',x)) }
parse_pred <- function(x) { s<-tolower(x); m<-regexpr('deceptive|deception|truthful|truth|yalan|aldatıcı|doğru|dogru',s,perl=TRUE); if(m[1]<0) stop(paste('Unparseable response:',x)); z<-regmatches(s,m); if(z %in% c('deceptive','deception','yalan','aldatıcı')) 'Deceptive' else 'Truthful' }
if(!is.null(prompts)) {
 lines<-readLines(prompts,warn=FALSE,encoding='UTF-8'); rec<-list(); cur<-list()
 for(line in c(lines,'')) { if(!nzchar(trimws(line))) { if(length(cur)) rec[[length(rec)+1]]<-as.data.frame(cur,stringsAsFactors=FALSE); cur<-list() } else if(grepl(':',line,fixed=TRUE)) { z<-strsplit(line,':',fixed=TRUE)[[1]]; nm<-tolower(gsub('[^a-z0-9]+','_',trimws(z[1]))); cur[[nm]]<-paste(z[-1],collapse=':') } }
 if(!length(rec)) stop('No prompt records parsed'); df<-do.call(rbind,rec)
} else df<-read.csv(input,check.names=FALSE,stringsAsFactors=FALSE,fileEncoding='UTF-8-BOM')
canon<-function(x) gsub('[^a-z0-9]+','_',tolower(trimws(x)))
findcol<-function(options,required=TRUE) { j<-match(canon(options),canon(names(df))); j<-j[!is.na(j)]; if(!length(j)){if(required) stop(paste('Missing column:',paste(options,collapse='/'))) else return(NA_integer_)}; j[1] }
sentc<-findcol(c('sentence','sentence_tr','statement','turkish_sentence','text')); gtc<-findcol(c('ground_truth','mapped_label','label','target_label')); n<-nrow(df); if(is.finite(limit)) { n<-min(n,max(0,limit)); df<-df[seq_len(n),,drop=FALSE] }
sent<-as.character(df[[sentc]]); truth<-vapply(df[[gtc]],norm,''); if(any(!nzchar(sent))) stop('Empty sentence')
getval<-function(options,default='') { j<-findcol(options,FALSE); if(is.na(j)) rep(default,n) else as.character(df[[j]]) }
src<-getval('source_column')
zero<-'Aşağıdaki Türkçe cümlenin doğru mu yoksa aldatıcı mı olduğunu belirleyin. Yalnızca Truthful veya Deceptive yazın.\n\nCümle: %s\nKarar:'
few<-'Aşağıdaki Türkçe cümlenin doğru mu yoksa aldatıcı mı olduğunu belirleyin. Yalnızca Truthful veya Deceptive yazın.\n\nÖrnek: Ali evde. => Truthful\nÖrnek: Ali evde değilken evde olduğunu söyledi. => Deceptive\n\nCümle: %s\nKarar:'
simulate<-function(i) if(grepl('l_dir|l_ind|lie|decept',tolower(src[i]))) 'Deceptive' else if(grepl('yalan|aldatıcı|aldatmak',tolower(sent[i]))) 'Deceptive' else 'Truthful'
call_api<-function(prompt) { if(!requireNamespace('httr',quietly=TRUE)||!requireNamespace('jsonlite',quietly=TRUE)) stop('API mode requires httr and jsonlite packages'); if(!nzchar(key)) stop('Set OPENAI_API_KEY or --api-key'); body<-list(model=model,messages=list(list(role='system',content='Classify the Turkish statement. Reply with exactly Truthful or Deceptive.'),list(role='user',content=prompt)),temperature=0,max_tokens=12); res<-httr::POST(endpoint,httr::add_headers(Authorization=paste('Bearer',key)),body=body,encode='json',timeout(60)); httr::stop_for_status(res); jsonlite::fromJSON(httr::content(res,'text',encoding='UTF-8'))$choices[[1]]$message$content }
pred<-character(n); raw<-character(n)
for(i in seq_len(n)) { prompt<-sprintf(if(template=='few_shot') few else zero,sent[i]); raw[i]<-if(real) call_api(prompt) else simulate(i); pred[i]<-parse_pred(raw[i]) }
out<-df; out$ground_truth<-truth; out$prediction<-pred; out$raw_response<-raw; out$is_correct<-truth==pred; write.csv(out,output,row.names=FALSE,fileEncoding='UTF-8')
classes<-c('Truthful','Deceptive'); cm<-table(factor(truth,levels=classes),factor(pred,levels=classes)); per<-lapply(classes,function(cl){tp<-cm[cl,cl]; fp<-sum(cm[,cl])-tp; fn<-sum(cm[cl,])-tp; p<-if(tp+fp)tp/(tp+fp) else 0; r<-if(tp+fn)tp/(tp+fn) else 0; list(precision=p,recall=r,f1=if(p+r)2*p*r/(p+r) else 0) }); names(per)<-classes
summary<-list(n=n,accuracy=if(n)mean(truth==pred) else 0,confusion_matrix=as.data.frame.matrix(cm),per_class=per,macro_f1=mean(vapply(per,function(x)x$f1,numeric(1))))
if(requireNamespace('jsonlite',quietly=TRUE)) jsonlite::write_json(summary,summary_file,pretty=TRUE,auto_unbox=TRUE) else writeLines(c(paste0('n: ',n),paste0('accuracy: ',summary$accuracy)),summary_file)
cat(sprintf('Wrote %d rows to %s; accuracy=%.4f\n',n,output,summary$accuracy))
