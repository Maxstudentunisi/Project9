Il progetto è organizzato in vari file .py. Il file run.py è lo script principale per eseguire gli attacchi. Durante l’esecuzione viene creata automaticamente una cartella outputs/, dove vengono salvate le immagini generate con i risultati.

Attenzione: ad ogni esecuzione dello script la cartella viene sostituita!

I valori di epsilon e il numero di sample da testare sono definiti direttamente nel codice e alcuni sono commentati. Per provare configurazioni diverse è necessario decommentare manualmente i valori desiderati.

Questa scelta è stata fatta perché effettuiamo 3 attacchi su 6 modelli: eseguire automaticamente tutte le combinazioni possibili è troppo oneroso dal punto di vista computazionale per il nostro hardware, quindi i test vengono selezionati manualmente in base alle esigenze.

Per eseguire il run.py è strettamente necessaria la cartella "checkpoints" dove sono caricati i pesi dei vari modelli preaddestrati ed il file models.py il quale ha al suo interno la funzione che ritorna i modelli utilizzati.

Gli altri due file "data.py" e train.py", sono stati utilizzati per il training dei modelli, il run.py scarica le 10000 immagini di test.

