# Instructions
- bertSupervised , file con classi utilizzate per il modello nuovo (chiamato bert)
- ExternalTagsIntegration, file dove eseguo il preprocess dei vertical e tag esterni
- Pipeline_predict , file con il metodo per generare i label con match (generatepseudolabel()) per i vertical, ignora il resto
- sqlPrediction , main dove modificare le chiamate sql
- SubBigrams , ignora
- TagPredictions, file con il match dei tag
- VertPicker, criterio di selezione dei vertical più importanti una volta ottenuto l'output dei modelli

uncomment riga 63,64,65 per scaricare i file da blob_storage
per lanciare il processo esegui sql_predictions